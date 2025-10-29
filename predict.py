"""Run DEIMv2 inference on images and export YOLO-format detections.

This script loads a trained DEIMv2 checkpoint and produces predictions for a
single image or for all images inside a directory.  For every processed image
we emit a ``.txt`` file containing one detection per line in the format::

    cls x y h w conf

Where ``x``/``y`` are the normalized centre coordinates and ``h``/``w`` are the
normalized height/width (YOLO convention).  The confidence score is appended as
an extra column so downstream tooling can apply custom thresholds.

The implementation reuses the project ``YAMLConfig`` to build the model and
postprocessor, mirrors the evaluation-time resize/normalization pipeline, and
supports overriding key runtime settings from the command line.
"""
from __future__ import annotations

import argparse
import math
import os
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

def _mitigate_duplicate_openmp():
    """Avoid Windows libiomp duplication crashes after CUDA-enabled installs.

    Recent PyTorch + CUDA builds pulled from the PyTorch/NVIDIA channels may
    ship both Intel OpenMP (via MKL) and LLVM OpenMP runtimes.  When Python
    packages import them in different orders, Windows raises the libiomp
    duplication error shown by users trying to re-install CUDA builds.  The
    safest long-term solution is to align the environment so that only one
    OpenMP runtime is present, but to keep the training script usable we
    optimistically allow duplicates and surface a warning.
    """

    if os.name != 'nt':
        return

    if os.environ.get('KMP_DUPLICATE_LIB_OK'):
        return

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    warnings.warn(
        'Detected Windows environment without KMP_DUPLICATE_LIB_OK. '
        'Temporarily allowing duplicate OpenMP runtimes to avoid libiomp '
        'initialization errors.  Consider reinstalling PyTorch/NumPy with a '
        'single OpenMP runtime for maximum stability.',
        RuntimeWarning,
        stacklevel=2,
    )


_mitigate_duplicate_openmp()

import torch
from PIL import Image
from torch import nn
from torchvision import transforms as T

from engine.core import YAMLConfig

# Common image file extensions that we accept when iterating over directories.
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _mitigate_duplicate_openmp() -> None:
    """Allow duplicate OpenMP runtimes on Windows to avoid libiomp crashes."""

    if os.name != "nt":
        return

    if os.environ.get("KMP_DUPLICATE_LIB_OK"):
        return

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    warnings.warn(
        "Detected Windows environment without KMP_DUPLICATE_LIB_OK. "
        "Temporarily allowing duplicate OpenMP runtimes to avoid libiomp "
        "initialization errors.  Consider reinstalling PyTorch/NumPy with a "
        "single OpenMP runtime for maximum stability.",
        RuntimeWarning,
        stacklevel=2,
    )


def _strip_module_prefix(state_dict: dict) -> dict:
    """Remove ``module.`` prefixes that appear after DDP training."""

    if not isinstance(state_dict, dict):
        return state_dict

    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v
    return out


def _select_model_state(state: dict) -> dict:
    """Extract the model weights from a checkpoint structure."""

    if "ema" in state and isinstance(state["ema"], dict) and "module" in state["ema"]:
        return state["ema"]["module"]
    if "model" in state:
        return state["model"]
    return state


def _normalization_from_config(cfg: YAMLConfig) -> Optional[Tuple[Sequence[float], Sequence[float]]]:
    """Inspect the validation transforms and return (mean, std) if normalize is used."""

    val_dataset = cfg.yaml_cfg.get("val_dataloader", {}).get("dataset", {})
    transforms_cfg = val_dataset.get("transforms", {})
    ops: Sequence[dict] = transforms_cfg.get("ops", [])  # type: ignore[assignment]

    for op in ops:
        if isinstance(op, dict) and op.get("type") == "Normalize":
            mean = op.get("mean")
            std = op.get("std")
            if mean is not None and std is not None:
                return mean, std
    return None


def _extract_feat_strides(cfg: YAMLConfig) -> Sequence[int]:
    """Fetch the decoder feature strides declared in the YAML config."""

    for key in ("DEIMTransformer", "RTDETRDecoder", "DEIMDecoder", "RTDETRTransformer"):
        strides = cfg.yaml_cfg.get(key, {}).get("feat_strides")
        if strides:
            return tuple(int(s) for s in strides)
    return ()


def _anchor_count_for_size(height: int, width: int, strides: Sequence[int]) -> Optional[int]:
    """Return the expected number of anchors for the provided resolution."""

    if not strides:
        return None

    total = 0
    for stride in strides:
        if height % stride != 0 or width % stride != 0:
            return None
        total += (height // stride) * (width // stride)
    return total


def _infer_square_size_from_anchors(anchor_count: int, strides: Sequence[int]) -> Optional[int]:
    """Infer a square evaluation size whose anchor grid matches ``anchor_count``."""

    if not strides or anchor_count <= 0:
        return None

    lcm_stride = strides[0]
    for stride in strides[1:]:
        lcm_stride = math.lcm(lcm_stride, stride)

    # Search over reasonable multiples of the LCM (up to 4096 pixels).
    for multiplier in range(1, (4096 // lcm_stride) + 1):
        size = lcm_stride * multiplier
        total = _anchor_count_for_size(size, size, strides)
        if total == anchor_count:
            return size
    return None


def _resolve_eval_size(cfg: YAMLConfig, override: Optional[int]) -> Tuple[int, int]:
    """Determine the inference resolution, respecting optional CLI overrides."""

    if override is not None:
        return int(override), int(override)

    size = cfg.yaml_cfg.get("eval_spatial_size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        w, h = int(size[0]), int(size[1])
        return w, h

    # Fallback to the validation resize if eval_spatial_size is absent.
    val_dataset = cfg.yaml_cfg.get("val_dataloader", {}).get("dataset", {})
    transforms_cfg = val_dataset.get("transforms", {})
    ops: Sequence[dict] = transforms_cfg.get("ops", [])  # type: ignore[assignment]
    for op in ops:
        if isinstance(op, dict) and op.get("type") == "Resize" and "size" in op:
            sz = op["size"]
            if isinstance(sz, (list, tuple)) and len(sz) == 2:
                return int(sz[0]), int(sz[1])

    return 640, 640


def _build_preprocess(size_hw: Tuple[int, int], norm: Optional[Tuple[Sequence[float], Sequence[float]]]) -> T.Compose:
    """Create the torchvision preprocessing pipeline."""

    transforms: List[T.Transform] = [T.Resize(size_hw)]
    transforms.append(T.ToTensor())
    if norm is not None:
        mean, std = norm
        transforms.append(T.Normalize(mean=mean, std=std))
    return T.Compose(transforms)


class _DeployedModel(nn.Module):
    """Wrapper around ``cfg.model``/``cfg.postprocessor`` in deploy mode."""

    def __init__(self, cfg: YAMLConfig, device: torch.device) -> None:
        super().__init__()
        self.model = cfg.model.deploy().to(device)
        self.postprocessor = cfg.postprocessor.deploy().to(device)
        self.model.eval()
        self.postprocessor.eval()

    def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


class Predictor:
    """High-level helper that loads the network and processes images."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: torch.device,
        image_size: Optional[int] = None,
    ) -> None:
        self.cfg = YAMLConfig(config_path)

        # Avoid redundant backbone downloads during inference.
        if "HGNetv2" in self.cfg.yaml_cfg:
            self.cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        if "DINOv3STAs" in self.cfg.yaml_cfg:
            weights_key = "weights_path"
            if weights_key in self.cfg.yaml_cfg["DINOv3STAs"] and not Path(
                self.cfg.yaml_cfg["DINOv3STAs"][weights_key]
            ).exists():
                warnings.warn(
                    "DINOv3 pretrained weights path does not exist; ensure the file is available before inference.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _strip_module_prefix(_select_model_state(state))

        strides = _extract_feat_strides(self.cfg)
        anchor_tensor = state_dict.get("decoder.anchors")
        anchor_count = int(anchor_tensor.shape[1]) if isinstance(anchor_tensor, torch.Tensor) else None

        desired_size: Optional[int] = None
        effective_override: Optional[int] = image_size
        if image_size is not None:
            desired_size = int(image_size)
            if anchor_count is not None:
                expected = _anchor_count_for_size(desired_size, desired_size, strides)
                if expected is None or expected != anchor_count:
                    warnings.warn(
                        "Requested --image-size does not match anchor grid stored in the checkpoint; "
                        "using checkpoint-derived resolution instead.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    desired_size = None
                    effective_override = None

        if desired_size is None and anchor_count is not None:
            inferred = _infer_square_size_from_anchors(anchor_count, strides)
            if inferred is not None:
                desired_size = inferred
                effective_override = desired_size

        if desired_size is not None:
            resolved = [desired_size, desired_size]
            if self.cfg.yaml_cfg.get("eval_spatial_size") != resolved:
                print(
                    f"[Override] eval_spatial_size: {self.cfg.yaml_cfg.get('eval_spatial_size')} -> {resolved}"
                )
            self.cfg.yaml_cfg["eval_spatial_size"] = resolved
            self.cfg.eval_spatial_size = resolved

        load_result = self.cfg.model.load_state_dict(state_dict, strict=False)
        missing, unexpected = load_result.missing_keys, load_result.unexpected_keys
        if missing:
            print(f"Warning: missing keys when loading checkpoint: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys when loading checkpoint: {unexpected}")

        self.device = device
        self.model = _DeployedModel(self.cfg, device)

        eval_w, eval_h = _resolve_eval_size(self.cfg, effective_override)
        self.input_size = (eval_h, eval_w)
        self.preprocess = _build_preprocess(self.input_size, _normalization_from_config(self.cfg))

    def predict(self, image_path: Path, score_threshold: float) -> List[Tuple[int, float, float, float, float, float]]:
        """Run inference on a single image and return YOLO-format detections."""

        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        inputs = self.preprocess(image).unsqueeze(0).to(self.device)
        orig_sizes = torch.tensor([[orig_w, orig_h]], dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            labels, boxes, scores = self.model(inputs, orig_sizes)

        labels = labels[0].cpu()
        boxes = boxes[0].cpu()
        scores = scores[0].cpu()

        results: List[Tuple[int, float, float, float, float, float]] = []
        for label, box, score in zip(labels, boxes, scores):
            conf = float(score.item())
            if conf < score_threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            box_w = max(x2 - x1, 0.0)
            box_h = max(y2 - y1, 0.0)
            cx = x1 + box_w / 2.0
            cy = y1 + box_h / 2.0

            # Normalize to [0, 1] following YOLO conventions.
            norm_cx = min(max(cx / orig_w, 0.0), 1.0) if orig_w > 0 else 0.0
            norm_cy = min(max(cy / orig_h, 0.0), 1.0) if orig_h > 0 else 0.0
            norm_h = min(max(box_h / orig_h, 0.0), 1.0) if orig_h > 0 else 0.0
            norm_w = min(max(box_w / orig_w, 0.0), 1.0) if orig_w > 0 else 0.0

            results.append((int(label.item()), norm_cx, norm_cy, norm_h, norm_w, conf))
        return results


def _gather_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]

    files: List[Path] = []
    for ext in _IMAGE_EXTENSIONS:
        files.extend(path.glob(f"**/*{ext}"))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No images found under {path}")
    return files


def _write_yolo_file(output_path: Path, detections: Iterable[Tuple[int, float, float, float, float, float]]) -> None:
    lines = [
        f"{cls_id} {cx:.6f} {cy:.6f} {h:.6f} {w:.6f} {conf:.6f}"
        for cls_id, cx, cy, h, w, conf in detections
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DEIMv2 inference and export YOLO txt files.")
    parser.add_argument("--config", required=True, help="Path to the YAML config used for training.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained checkpoint (.pth).")
    parser.add_argument(
        "--input",
        required=True,
        help="Image file or directory containing images to run inference on.",
    )
    parser.add_argument(
        "--output-dir",
        default="predictions",
        help="Directory where YOLO-format txt files will be written (default: predictions/).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional square inference size that overrides eval_spatial_size from the config.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.25,
        help="Detections below this confidence will be filtered out (default: 0.25).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string (e.g., cuda, cuda:0, cpu). Defaults to CUDA if available.",
    )
    return parser.parse_args()


def main() -> None:
    _mitigate_duplicate_openmp()
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    predictor = Predictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        image_size=args.image_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = _gather_images(input_path)
    print(f"Found {len(images)} image(s) under {input_path}.")

    for image_path in images:
        detections = predictor.predict(image_path, score_threshold=args.score_threshold)
        output_path = output_dir / f"{image_path.stem}.txt"
        _write_yolo_file(output_path, detections)
        print(f"Processed {image_path.name}: {len(detections)} detections -> {output_path}")


if __name__ == "__main__":
    main()
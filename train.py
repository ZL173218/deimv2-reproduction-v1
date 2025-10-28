"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
import warnings


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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
from typing import Dict, Any, Optional

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

debug=False

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

def _merge_cli_kwargs(args) -> Dict[str, Any]:
    """Merge generic CLI keyword arguments for YAMLConfig construction.

    Some runtime overrides (batch size, image size, multi-scale control) are
    handled later because they require mutating nested dictionaries that would
    otherwise violate `total_batch_size`/`batch_size` assertions.  We therefore
    exclude them from the initial merge here.
    """

    skip_keys = {
        'update',
        'train_batch_size',
        'train_total_batch_size',
        'val_batch_size',
        'val_total_batch_size',
        'train_image_size',
        'eval_image_size',
        'disable_train_multiscale',
    }

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({
        k: v for k, v in vars(args).items()
        if k not in skip_keys and v is not None
    })
    return update_dict


def _apply_runtime_overrides(cfg: YAMLConfig, args) -> None:
    """Apply memory-conscious overrides after loading the YAML config."""

    def _set_eval_spatial_size_if_needed(target_size: Optional[int]) -> None:
        """Normalize the eval spatial size setting across config views."""

        if target_size is None:
            return

        resolved = [int(target_size), int(target_size)]

        if cfg.yaml_cfg.get('eval_spatial_size') != resolved:
            print(f'[Override] eval_spatial_size: {cfg.yaml_cfg.get("eval_spatial_size")} -> {resolved}')

        cfg.yaml_cfg['eval_spatial_size'] = resolved
        # Keep the attribute view in sync so later prints/debugging reflect the override.
        cfg.eval_spatial_size = resolved

    def _override_loader(
        loader_key: str,
        *,
        batch_size: Optional[int] = None,
        total_batch_size: Optional[int] = None,
        image_size: Optional[int] = None,
        disable_multiscale: bool = False,
    ) -> None:
        loader_cfg = cfg.yaml_cfg.get(loader_key)
        if loader_cfg is None:
            return

        if total_batch_size is not None and batch_size is not None:
            raise ValueError('Specify either batch_size or total_batch_size, not both.')

        if total_batch_size is not None:
            loader_cfg.pop('batch_size', None)
            loader_cfg['total_batch_size'] = total_batch_size
            print(f'[Override] {loader_key}.total_batch_size -> {total_batch_size}')
        elif batch_size is not None:
            loader_cfg.pop('total_batch_size', None)
            loader_cfg['batch_size'] = batch_size
            print(f'[Override] {loader_key}.batch_size -> {batch_size}')

        collate_cfg = loader_cfg.get('collate_fn')
        if collate_cfg is not None:
            if disable_multiscale:
                if collate_cfg.get('base_size_repeat') is not None:
                    print(f'[Override] {loader_key}.collate_fn.base_size_repeat -> None (disable multi-scale)')
                collate_cfg['base_size_repeat'] = None
            if image_size is not None:
                base_before = collate_cfg.get('base_size')
                collate_cfg['base_size'] = image_size
                print(f'[Override] {loader_key}.collate_fn.base_size: {base_before} -> {image_size}')

        if image_size is not None:
            dataset_cfg = loader_cfg.get('dataset')
            if dataset_cfg is not None:
                transforms = dataset_cfg.get('transforms')
                if isinstance(transforms, dict):
                    ops = transforms.get('ops', [])
                    for op in ops:
                        if isinstance(op, dict) and op.get('type') == 'Resize' and 'size' in op:
                            old_size = op['size']
                            op['size'] = [image_size, image_size]
                            print(f'[Override] {loader_key}.dataset.transforms.Resize.size: {old_size} -> {op["size"]}')

    _override_loader(
        'train_dataloader',
        batch_size=args.train_batch_size,
        total_batch_size=args.train_total_batch_size,
        image_size=args.train_image_size,
        disable_multiscale=args.disable_train_multiscale,
    )

    _override_loader(
        'val_dataloader',
        batch_size=args.val_batch_size,
        total_batch_size=args.val_total_batch_size,
        image_size=args.eval_image_size,
    )

    if args.eval_image_size is not None:
        _set_eval_spatial_size_if_needed(args.eval_image_size)
    else:
        # Fall back to the chosen validation base size or training override so evaluation
        # does not retain a stale resolution when the loaders change.
        val_collate = (
            cfg.yaml_cfg
            .get('val_dataloader', {})
            .get('collate_fn', {})
        )
        candidate = val_collate.get('base_size')
        if candidate is None and args.train_image_size is not None and cfg.yaml_cfg.get('eval_spatial_size') is None:
            candidate = args.train_image_size
        _set_eval_spatial_size_if_needed(candidate)


def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    update_dict = _merge_cli_kwargs(args)

    cfg = YAMLConfig(args.config, **update_dict)
    _apply_runtime_overrides(cfg, args)

    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, default='')
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, default=0, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--train-batch-size', type=int, help='override train per-device batch size')
    parser.add_argument('--train-total-batch-size', type=int, help='override total train batch size across ranks')
    parser.add_argument('--val-batch-size', type=int, help='override validation per-device batch size')
    parser.add_argument('--val-total-batch-size', type=int, help='override total validation batch size across ranks')
    parser.add_argument('--train-image-size', type=int, help='override training image size / base size')
    parser.add_argument('--eval-image-size', type=int, help='override evaluation image size')
    parser.add_argument('--disable-train-multiscale', action='store_true', help='disable train-time multi-scale resizing')

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)

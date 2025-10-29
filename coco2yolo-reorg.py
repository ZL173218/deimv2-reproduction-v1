import argparse
import json
import logging
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, List, Set
from PIL import Image
from tqdm import tqdm

def parse_args():
    ap = argparse.ArgumentParser(
        description="Convert a standard COCO dataset to YOLO format. Outputs ABSOLUTE paths in train.txt/val.txt."
    )
    ap.add_argument("--root", type=str, required=True,
                    help="Dataset root containing images/{train,val} and annotations/instances_{train,val}.json")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--copy", action="store_true", help="Copy images into images/")
    mode.add_argument("--link", action="store_true", help="Hardlink images into images/")
    mode.add_argument("--symlink", action="store_true", help="Symlink images into images/")
    ap.add_argument("--overwrite", action="store_true",
                    help="Remove existing images/, labels/, train.txt, val.txt before writing")
    ap.add_argument("--write-classes", action="store_true",
                    help="Write classes.txt with YOLO class names (one per line)")
    ap.add_argument("--quiet", action="store_true", help="Less logging")
    return ap.parse_args()

def setup_logging(quiet=False):
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="[%(levelname)s] %(message)s"
    )

def ensure_dirs(root: Path, overwrite: bool):
    images_flat = root / "images"
    labels_dir  = root / "labels"
    if overwrite:
        for d in [images_flat, labels_dir]:
            if d.exists():
                shutil.rmtree(d)
        for f in [root/"train.txt", root/"val.txt", root/"classes.txt"]:
            if f.exists():
                f.unlink()
    images_flat.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_flat, labels_dir

def load_coco_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"COCO json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco.get("images", [])}
    ann_by_img = defaultdict(list)
    for ann in coco.get("annotations", []):
        ann_by_img[ann["image_id"]].append(ann)
    cats = {c["id"]: c.get("name", str(c["id"])) for c in coco.get("categories", [])}
    return coco, images, ann_by_img, cats

def build_global_cat_mapping(train_cats: Dict[int,str], val_cats: Dict[int,str]) -> Tuple[Dict[int,int], List[str]]:
    all_ids = sorted(set(train_cats.keys()).union(val_cats.keys()))
    cat2yolo = {cid: i for i, cid in enumerate(all_ids)}
    names = []
    for cid in all_ids:
        name = train_cats.get(cid, val_cats.get(cid, f"class_{cat2yolo[cid]}"))
        names.append(name)
    return cat2yolo, names

def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    x = max(0.0, min(x, img_w))
    y = max(0.0, min(y, img_h))
    w = max(0.0, min(w, img_w - x))
    h = max(0.0, min(h, img_h - y))
    if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
        return None
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    def clamp01(v): return max(0.0, min(1.0, v))
    return (clamp01(cx), clamp01(cy), clamp01(nw), clamp01(nh))

def write_yolo_label(label_path: Path, lines: List[str]):
    if lines:
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        label_path.write_text("", encoding="utf-8")  # 显式空标签文件

def process_split(
    split_name: str,
    coco_json: Path,
    split_img_dir: Path,
    images_flat: Path,
    labels_dir: Path,
    cat2yolo: Dict[int,int],
    method: str
) -> List[str]:
    coco, images, ann_by_img, _ = load_coco_json(coco_json)
    abs_list = []
    missing: Set[str] = set()

    for img_id, im in tqdm(images.items(), desc=f"{split_name}: images", ncols=80):
        fname = im["file_name"]
        src_img = split_img_dir / fname
        if not src_img.is_file():
            candidates = list(split_img_dir.rglob(fname))
            if candidates:
                src_img = candidates[0]
            else:
                missing.add(fname)
                continue

        dst_img = images_flat / src_img.name
        if method == "copy":
            shutil.copy2(src_img, dst_img)
        elif method == "link":
            try:
                if dst_img.exists():
                    dst_img.unlink()
                dst_img.hardlink_to(src_img)
            except Exception:
                shutil.copy2(src_img, dst_img)
        elif method == "symlink":
            if dst_img.exists():
                dst_img.unlink()
            dst_img.symlink_to(src_img)
        else:
            raise ValueError("Unknown method")

        w = im.get("width", None)
        h = im.get("height", None)
        if not w or not h:
            with Image.open(src_img) as pil:
                w, h = pil.size

        anns = ann_by_img.get(img_id, [])
        yolo_lines = []
        for ann in anns:
            if ann.get("iscrowd", 0) == 1:
                continue
            cid = ann["category_id"]
            if cid not in cat2yolo:
                continue
            yid = cat2yolo[cid]
            y = coco_bbox_to_yolo(ann["bbox"], w, h)
            if y is None:
                continue
            cx, cy, bw, bh = y
            if bw <= 0 or bh <= 0:
                continue
            yolo_lines.append(f"{yid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path = labels_dir / (dst_img.stem + ".txt")
        write_yolo_label(label_path, yolo_lines)

        # 关键：写入绝对路径
        abs_list.append(str(dst_img.resolve()))

    if missing:
        logging.warning(f"[{split_name}] Missing {len(missing)} images in annotations, e.g. {list(missing)[:3]}")

    return sorted(abs_list)

def main():
    args = parse_args()
    setup_logging(args.quiet)

    root = Path(args.root).resolve()
    imgs_train = root / "images" / "train"
    imgs_val   = root / "images" / "val"
    ann_train  = root / "annotations" / "instances_train.json"
    ann_val    = root / "annotations" / "instances_val.json"

    if not imgs_train.is_dir() or not imgs_val.is_dir():
        raise SystemExit("Expected images/train and images/val under dataset root.")
    if not ann_train.is_file() or not ann_val.is_file():
        raise SystemExit("Expected annotations/instances_train.json and instances_val.json under dataset root.")

    images_flat, labels_dir = ensure_dirs(root, args.overwrite)

    # 统一类别映射（train+val）
    _, _, _, cats_train = load_coco_json(ann_train)
    _, _, _, cats_val   = load_coco_json(ann_val)
    cat2yolo, yolo_names = build_global_cat_mapping(cats_train, cats_val)

    method = "copy"
    if args.link: method = "link"
    if args.symlink: method = "symlink"

    train_abs = process_split("train", ann_train, imgs_train, images_flat, labels_dir, cat2yolo, method)
    val_abs   = process_split("val",   ann_val,   imgs_val,   images_flat, labels_dir, cat2yolo, method)

    # 写绝对路径
    (root / "train.txt").write_text("\n".join(train_abs) + "\n", encoding="utf-8")
    (root / "val.txt").write_text("\n".join(val_abs) + "\n", encoding="utf-8")

    if args.write_classes:
        (root / "classes.txt").write_text("\n".join(yolo_names) + "\n", encoding="utf-8")

    logging.info("Done.")
    logging.info(f"images/: {images_flat}")
    logging.info(f"labels/: {labels_dir}")
    logging.info(f"train.txt / val.txt written with ABSOLUTE paths")
    if args.write_classes:
        logging.info(f"classes.txt written with {len(yolo_names)} classes")

if __name__ == "__main__":
    main()

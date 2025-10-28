import argparse
import json
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging

def parse_args():
    ap = argparse.ArgumentParser(
        description="Reorganize YOLO dataset by train/val lists and convert labels to COCO."
    )
    ap.add_argument("--root", type=str, required=True,
                    help="Dataset root dir containing images/, labels/, train.txt, val.txt")
    ap.add_argument("--train-list", type=str, default="train.txt",
                    help="Relative path to train list file under root (default: train.txt)")
    ap.add_argument("--val-list", type=str, default="val.txt",
                    help="Relative path to val list file under root (default: val.txt)")
    ap.add_argument("--images-dir", type=str, default="images",
                    help="Images directory name under root (default: images)")
    ap.add_argument("--labels-dir", type=str, default="labels",
                    help="Labels directory name under root (default: labels)")
    ap.add_argument("--to-jpg", action="store_true",
                    help="Convert/copy images as .jpg into target split folders")
    ap.add_argument("--classes-file", type=str, default=None,
                    help="Optional: a text file with one class name per line")
    ap.add_argument("--class-names", type=str, default=None,
                    help="Optional: comma-separated class names (overrides classes-file)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite target images/ and annotations/ if exist")
    ap.add_argument("--quiet", action="store_true", help="Less logging")
    return ap.parse_args()

def setup_logging(quiet: bool):
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="[%(levelname)s] %(message)s"
    )

def read_list(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"List file not found: {file_path}")
    lines = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            lines.append(s)
    return lines

def rel_to_stem(rel_or_abs: str):
    """
    train.txt/val.txt 里可能是绝对路径或相对路径。
    我们只取文件名（不含扩展名）的 stem 作为锚点，去 images/ 与 labels/ 下找对应文件。
    """
    return Path(rel_or_abs).stem

def find_image_by_stem(images_dir: Path, stem: str):
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
    for ext in exts:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def read_class_names(args, discovered_ids):
    # 优先顺序：--class-names > --classes-file > 自动生成 class_{id}
    if args.class_names:
        names = [x.strip() for x in args.class_names.split(",") if x.strip()]
        return {i: n for i, n in enumerate(names)}
    if args.classes_file:
        path = Path(args.classes_file)
        if not path.is_file():
            raise FileNotFoundError(f"classes-file not found: {path}")
        names = [line.strip() for line in path.open("r", encoding="utf-8") if line.strip()]
        return {i: n for i, n in enumerate(names)}

    # 自动：按发现到的类别 id 生成占位名称
    return {i: f"class_{i}" for i in sorted(discovered_ids)}

def yolo_label_path(labels_dir: Path, stem: str):
    return labels_dir / f"{stem}.txt"

def load_yolo_boxes(lbl_path: Path):
    """
    返回列表[(cls, cx, cy, w, h)]，均为 float，YOLO 归一化格式。
    若无文件则返回空列表。
    """
    if not lbl_path.exists():
        return []
    out = []
    with lbl_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            out.append((cls, cx, cy, w, h))
    return out

def yolo_to_coco_xywh(cx, cy, w, h, img_w, img_h):
    """
    YOLO(归一化) -> COCO [x, y, w, h]，像素坐标
    """
    x = (cx - w / 2.0) * img_w
    y = (cy - h / 2.0) * img_h
    return [max(0.0, x), max(0.0, y), max(0.0, w * img_w), max(0.0, h * img_h)]

def ensure_dirs(root: Path, overwrite: bool):
    images_train = root / "images" / "train"
    images_val   = root / "images" / "val"
    ann_dir      = root / "annotations"

    for d in [images_train, images_val, ann_dir]:
        if d.exists() and overwrite:
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    return images_train, images_val, ann_dir

def copy_or_convert_image(src: Path, dst: Path, to_jpg: bool):
    dst = dst.with_suffix(".jpg") if to_jpg else dst.with_suffix(src.suffix.lower())
    dst.parent.mkdir(parents=True, exist_ok=True)

    if to_jpg:
        # 转成 JPG（RGB）
        with Image.open(src) as im:
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            else:
                if im.mode == "L":
                    im = im.convert("RGB")
            im.save(dst, format="JPEG", quality=95)
    else:
        # 直接拷贝
        shutil.copy2(src, dst)
    return dst

def make_coco_template():
    return {
        "info": {"description": "Converted from YOLO by yolo2coco_reorg.py"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

def build_split(
    split_name: str,
    list_file: Path,
    images_dir: Path,
    labels_dir: Path,
    out_images_dir: Path,
    to_jpg: bool,
    image_start_id: int,
    annot_start_id: int,
    class_name_map: dict,
    discovered_ids_global: set
):
    coco = make_coco_template()
    img_id = image_start_id
    ann_id = annot_start_id

    lines = read_list(list_file)
    # 记录发现到的类 id（仅用于无类别名时的回填；最终以 class_name_map 决定）
    discovered_ids_local = set()

    for line in tqdm(lines, desc=f"Processing {split_name}", ncols=80):
        stem = rel_to_stem(line)
        src_img = find_image_by_stem(images_dir, stem)
        if src_img is None:
            logging.warning(f"[{split_name}] Image not found for stem '{stem}' in {images_dir}")
            continue

        # 复制/转换图像
        dst_img_candidate = out_images_dir / f"{stem}"
        dst_img = copy_or_convert_image(src_img, dst_img_candidate, to_jpg)

        # 打开以获取尺寸
        with Image.open(dst_img) as im:
            w, h = im.size

        coco["images"].append({
            "id": img_id,
            "file_name": dst_img.name,  # 仅文件名，COCO 常见做法
            "width": w,
            "height": h
        })

        # 读取 YOLO 标签
        lbl_path = yolo_label_path(labels_dir, stem)
        boxes = load_yolo_boxes(lbl_path)
        for (cls, cx, cy, bw, bh) in boxes:
            discovered_ids_local.add(cls)
            discovered_ids_global.add(cls)
            bbox = yolo_to_coco_xywh(cx, cy, bw, bh, w, h)
            area = bbox[2] * bbox[3]
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls,
                "bbox": [round(x, 2) for x in bbox],
                "area": round(area, 2),
                "iscrowd": 0,
                "segmentation": []  # 无语义分割，留空
            })
            ann_id += 1

        img_id += 1

    # 类别填充（以 class_name_map 为准；若缺少某些 id，自动占位）
    max_cls = max(discovered_ids_global) if discovered_ids_global else -1
    for cid in range(max_cls + 1):
        name = class_name_map.get(cid, f"class_{cid}")
        coco["categories"].append({"id": cid, "name": name})

    return coco, img_id, ann_id

def main():
    args = parse_args()
    setup_logging(args.quiet)

    root = Path(args.root).resolve()
    images_dir = root / args.images_dir
    labels_dir = root / args.labels_dir
    train_list = root / args.train_list
    val_list   = root / args.val_list

    if not images_dir.is_dir():
        raise SystemExit(f"Images dir not found: {images_dir}")
    if not labels_dir.is_dir():
        logging.warning(f"Labels dir not found: {labels_dir} (images will still be reorganized)")
    if not train_list.is_file():
        raise SystemExit(f"Train list not found: {train_list}")
    if not val_list.is_file():
        raise SystemExit(f"Val list not found: {val_list}")

    images_train_dir, images_val_dir, ann_dir = ensure_dirs(root, args.overwrite)

    # 首轮扫描以发现所有可能出现的类别 id（仅当未提供类别名时用于构建占位）
    discovered_ids_global = set()

    # 若用户已指定类别名，先构造映射；否则等两边处理后再统一补齐
    if args.class_names:
        class_name_map = {i: n for i, n in enumerate([x.strip() for x in args.class_names.split(",") if x.strip()])}
    elif args.classes_file:
        names = [line.strip() for line in Path(args.classes_file).open("r", encoding="utf-8") if line.strip()]
        class_name_map = {i: n for i, n in enumerate(names)}
    else:
        class_name_map = {}  # 暂空，稍后用发现的 id 自动填充

    # 处理 train
    coco_train, next_img_id, next_ann_id = build_split(
        "train",
        train_list,
        images_dir,
        labels_dir,
        images_train_dir,
        args.to_jpg,
        image_start_id=1,
        annot_start_id=1,
        class_name_map=class_name_map,
        discovered_ids_global=discovered_ids_global
    )

    # 处理 val
    coco_val, _, _ = build_split(
        "val",
        val_list,
        images_dir,
        labels_dir,
        images_val_dir,
        args.to_jpg,
        image_start_id=next_img_id,
        annot_start_id=next_ann_id,
        class_name_map=class_name_map,
        discovered_ids_global=discovered_ids_global
    )

    # 若未提供类别名，则依据发现的 id 自动回填
    if not class_name_map:
        class_name_map = read_class_names(args, discovered_ids_global)

        # 用最终 class_name_map 重写 categories（保持 id 对齐）
        def rebuild_categories(coco_obj):
            coco_obj["categories"] = [
                {"id": cid, "name": class_name_map.get(cid, f"class_{cid}")}
                for cid in range(max(discovered_ids_global) + 1) if discovered_ids_global
            ]
        rebuild_categories(coco_train)
        rebuild_categories(coco_val)

    # 保存
    (ann_dir / "instances_train.json").write_text(
        json.dumps(coco_train, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ann_dir / "instances_val.json").write_text(
        json.dumps(coco_val, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logging.info("Done.")
    logging.info(f"Images (train): {images_train_dir}")
    logging.info(f"Images (val)  : {images_val_dir}")
    logging.info(f"Annotations   : {ann_dir/'instances_train.json'}, {ann_dir/'instances_val.json'}")

if __name__ == "__main__":
    main()

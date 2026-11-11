"""
Fine-tune YOLOv8n for firearm detection on Open Images V7.

Downloads the Handgun / Rifle / Shotgun subset of Open Images V7 (via FiftyOne),
exports it in YOLO format, fine-tunes the pretrained COCO yolov8n.pt, then copies
the best weights to edge/models/weapon_model.pt with a JSON card describing the run.

Usage (from the edge/ folder, using the training venv):
    python training/train_firearms.py                 # full run
    python training/train_firearms.py --smoke         # tiny run to check the pipeline

Dataset citation:
    Kuznetsova, A. et al. (2020). The Open Images Dataset V4. IJCV 128, 1956–1981.
"""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

HERE     = Path(__file__).resolve().parent
EDGE     = HERE.parent
DATA_DIR = HERE / "data" / "openimages_firearms"
RUNS_DIR = HERE / "runs"
OUT_PT   = EDGE / "models" / "weapon_model.pt"

# Open Images label → class name used by the edge THREAT_MAP
CLASSES = {"Handgun": "handgun", "Rifle": "rifle", "Shotgun": "shotgun"}
NEGATIVES_ADDED = {"train": 0, "val": 0}   # filled by add_negatives(), recorded in the card


def build_dataset(max_per_split: dict, seed: int) -> Path:
    import fiftyone as fo
    import fiftyone.zoo as foz

    yaml_path = DATA_DIR / "dataset.yaml"
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    for split, oi_split in (("train", "train"), ("val", "validation"), ("test", "test")):
        ds = foz.load_zoo_dataset(
            "open-images-v7",
            split=oi_split,
            label_types=["detections"],
            classes=list(CLASSES),
            max_samples=max_per_split[split],
            shuffle=True,
            seed=seed,
            dataset_name=f"oi-firearms-{split}-{max_per_split[split]}",
        )
        # Keep only the firearm boxes and rename to the edge's lowercase labels
        view = ds.filter_labels("ground_truth", fo.ViewField("label").is_in(list(CLASSES)))
        view = view.map_labels("ground_truth", CLASSES)
        view.export(
            export_dir=str(DATA_DIR),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            split=split,
            classes=list(CLASSES.values()),
        )
    return yaml_path


def add_negatives(n_train: int, n_val: int, seed: int):
    """Add background images (people, no firearm label) with empty label files.

    Without them the model never sees a gun-free scene and fires on dark shapes
    such as shorts, signs and handlebars. Drawn from Open Images so COCO stays
    an untouched test set for false alarms.
    """
    import fiftyone.zoo as foz

    positives = {p.stem for p in (DATA_DIR / "images").rglob("*.jpg")}
    for split, oi_split, n in (("train", "train", n_train), ("val", "validation", n_val)):
        if not n:
            continue
        ds = foz.load_zoo_dataset(
            "open-images-v7", split=oi_split, label_types=["detections"],
            classes=["Person"], max_samples=int(n * 1.3), shuffle=True, seed=seed + 1,
            dataset_name=f"oi-person-negatives-{split}-{n}",
        )
        added = 0
        for s in ds:
            stem = Path(s.filepath).stem
            if stem in positives:
                continue
            shutil.copy(s.filepath, DATA_DIR / "images" / split / f"neg_{stem}.jpg")
            (DATA_DIR / "labels" / split / f"neg_{stem}.txt").write_text("")
            added += 1
            if added >= n:
                break
        print(f"{split}: added {added} negative images")
        NEGATIVES_ADDED[split] = added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke",   action="store_true", help="tiny dataset, 1 epoch")
    ap.add_argument("--epochs",  type=int, default=50)
    ap.add_argument("--imgsz",   type=int, default=640)
    ap.add_argument("--batch",   type=int, default=16)
    ap.add_argument("--train-n", type=int, default=None, help="max train images (default: all)")
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--negatives", type=int, default=0,
                    help="background images (people, no firearms) added to train; 1/8 as many to val")
    ap.add_argument("--prepare-only", action="store_true", help="build the dataset and stop")
    args = ap.parse_args()

    if args.smoke:
        limits, epochs = {"train": 40, "val": 10, "test": 10}, 1
    else:
        limits, epochs = {"train": args.train_n, "val": None, "test": None}, args.epochs

    yaml_path = build_dataset(limits, args.seed)
    if args.negatives:
        add_negatives(args.negatives, args.negatives // 8, args.seed)
    if args.prepare_only:
        print(f"Dataset ready at {DATA_DIR}")
        return

    from ultralytics import YOLO
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    run_name = "smoke" if args.smoke else f"firearms_{datetime.now():%Y%m%d_%H%M}"
    model = YOLO(str(EDGE.parent / "yolov8n.pt"))
    model.train(data=str(yaml_path), epochs=epochs, imgsz=args.imgsz, batch=args.batch,
                device=device, project=str(RUNS_DIR), name=run_name, exist_ok=True,
                seed=args.seed, plots=True)

    best = RUNS_DIR / run_name / "weights" / "best.pt"
    test = YOLO(str(best)).val(data=str(yaml_path), split="test", device=device,
                               project=str(RUNS_DIR), name=f"{run_name}_test", exist_ok=True)

    card = {
        "base_model":   "yolov8n.pt (Ultralytics, COCO-pretrained)",
        "dataset":      "Open Images V7 — Handgun, Rifle, Shotgun",
        "classes":      list(CLASSES.values()),
        "epochs":       epochs,
        "imgsz":        args.imgsz,
        "seed":         args.seed,
        "device":       device,
        "smoke_test":   args.smoke,
        "negative_images": dict(NEGATIVES_ADDED),
        "test_metrics": {
            "precision": round(float(test.box.mp), 4),
            "recall":    round(float(test.box.mr), 4),
            "mAP50":     round(float(test.box.map50), 4),
            "mAP50_95":  round(float(test.box.map), 4),
            "per_class_mAP50_95": {n: round(float(v), 4)
                                   for n, v in zip(CLASSES.values(), test.box.maps)},
        },
        "trained_at":   datetime.now().isoformat(timespec="seconds"),
    }

    if args.smoke:
        print(json.dumps(card, indent=2))
        print("Smoke run complete — weights NOT installed.")
        return

    OUT_PT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, OUT_PT)
    OUT_PT.with_suffix(".json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))
    print(f"Installed → {OUT_PT}")


if __name__ == "__main__":
    main()

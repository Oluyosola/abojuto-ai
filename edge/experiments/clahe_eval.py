"""
Experiment: does CLAHE low-light enhancement improve person detection?

Uses COCO val2017 images that contain people. Each image is evaluated as:
  original           — normal lighting
  dark_<level>       — synthetic low light (gamma + gain + sensor noise)
  dark_<level>+clahe — the same dark image passed through the edge's own
                       _enhance_low_light() (so the tested code is the deployed code)

Metrics for the person class (IoU ≥ 0.5):
  precision / recall / F1 at the deployed confidence threshold
  AP50 (all-point interpolated) over all confidences

Run from the edge/ folder:
    python experiments/clahe_eval.py               # all person images
    python experiments/clahe_eval.py --limit 200   # quick run
Outputs → experiments/results/clahe/
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

EDGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EDGE))
from detector import _enhance_low_light, LOW_LIGHT_THRESHOLD  # noqa: E402

DATA    = EDGE / "experiments" / "data"
IMAGES  = DATA / "val2017"
LABELS  = DATA / "coco" / "labels" / "val2017"
OUT     = EDGE / "experiments" / "results" / "clahe"

# gain k, gamma g, noise sigma (pixel units)
DARK_LEVELS = {
    "mild":     (0.50, 1.2, 3.0),
    "moderate": (0.30, 1.5, 5.0),
    "severe":   (0.15, 2.0, 8.0),
}
IOU_MATCH = 0.5


def darken(img: np.ndarray, k: float, g: float, sigma: float, rng) -> np.ndarray:
    x = 255.0 * k * np.power(img.astype(np.float32) / 255.0, g)
    x += rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(x, 0, 255).astype(np.uint8)


def load_person_boxes(label_file: Path, w: int, h: int) -> np.ndarray:
    boxes = []
    for line in label_file.read_text().splitlines():
        p = line.split()
        if not p or p[0] != "0":
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between one box a [4] and many boxes b [N,4]."""
    ix = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def match(preds: np.ndarray, confs: np.ndarray, gts: np.ndarray):
    """Greedy match by confidence. Returns TP flag per prediction (sorted by conf)."""
    order = np.argsort(-confs)
    used  = np.zeros(len(gts), dtype=bool)
    tp    = np.zeros(len(preds), dtype=bool)
    for rank, i in enumerate(order):
        if len(gts) == 0:
            break
        ious = iou(preds[i], gts)
        ious[used] = 0
        j = int(np.argmax(ious))
        if ious[j] >= IOU_MATCH:
            used[j] = True
            tp[rank] = True
    return confs[order], tp


def ap_all_point(confs: np.ndarray, tps: np.ndarray, n_gt: int) -> float:
    order = np.argsort(-confs)
    tp = tps[order].astype(np.float64)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall    = ctp / max(n_gt, 1)
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--conf",  type=float, default=0.35, help="deployed confidence threshold")
    ap.add_argument("--model", default=str(EDGE.parent / "yolov8n.pt"))
    ap.add_argument("--seed",  type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model  = YOLO(args.model)
    rng    = np.random.default_rng(args.seed)

    files = sorted(f for f in LABELS.glob("*.txt")
                   if any(l.startswith("0 ") for l in f.read_text().splitlines()))
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} images with people")

    conditions = ["original"] + [f"dark_{l}" for l in DARK_LEVELS] + [f"dark_{l}+clahe" for l in DARK_LEVELS]
    acc = {c: {"confs": [], "tps": [], "n_gt": 0, "tp": 0, "fp": 0, "brightness": [],
               "clahe_applied": 0} for c in conditions}
    clahe_ms = []
    samples  = {}   # a few images saved for the thesis figure

    for n, lf in enumerate(files):
        img = cv2.imread(str(IMAGES / (lf.stem + ".jpg")))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_person_boxes(lf, w, h)

        variants = {"original": img}
        for level, (k, g, s) in DARK_LEVELS.items():
            dark = darken(img, k, g, s, rng)
            variants[f"dark_{level}"] = dark
            t0 = time.perf_counter()
            enh = _enhance_low_light(dark)
            clahe_ms.append((time.perf_counter() - t0) * 1000)
            variants[f"dark_{level}+clahe"] = enh
            if enh is not dark:
                acc[f"dark_{level}+clahe"]["clahe_applied"] += 1

        names = list(variants)
        results = model.predict([variants[c] for c in names], conf=0.001, classes=[0],
                                device=device, verbose=False)
        for c, r in zip(names, results):
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            sc, tp = match(boxes, confs, gts)
            a = acc[c]
            a["confs"].append(sc); a["tps"].append(tp); a["n_gt"] += len(gts)
            keep = sc >= args.conf
            a["tp"] += int(tp[keep].sum()); a["fp"] += int((~tp[keep]).sum())
            a["brightness"].append(float(cv2.cvtColor(variants[c], cv2.COLOR_BGR2GRAY).mean()))

        if len(samples) < 6 and 2 <= len(gts) <= 6 and n % 7 == 0:
            samples[lf.stem] = {c: (variants[c], results[i]) for i, c in enumerate(names)}

        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(files)}")

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for c, a in acc.items():
        tp, fp, n_gt = a["tp"], a["fp"], a["n_gt"]
        p = tp / max(tp + fp, 1)
        r = tp / max(n_gt, 1)
        summary[c] = {
            "images":           len(a["brightness"]),
            "person_instances": n_gt,
            "mean_brightness":  round(float(np.mean(a["brightness"])), 1),
            "precision":        round(p, 4),
            "recall":           round(r, 4),
            "f1":               round(2 * p * r / max(p + r, 1e-9), 4),
            "ap50":             round(ap_all_point(np.concatenate(a["confs"]),
                                                   np.concatenate(a["tps"]), n_gt), 4),
            "tp": tp, "fp": fp, "fn": n_gt - tp,
        }
        if c.endswith("+clahe"):
            summary[c]["clahe_applied_pct"] = round(100 * a["clahe_applied"] / max(len(a["brightness"]), 1), 1)

    meta = {
        "dataset":          "COCO val2017, images containing at least one person",
        "model":            Path(args.model).name,
        "device":           device,
        "conf_threshold":   args.conf,
        "iou_match":        IOU_MATCH,
        "low_light_threshold": LOW_LIGHT_THRESHOLD,
        "dark_levels":      {k: {"gain": v[0], "gamma": v[1], "noise_sigma": v[2]} for k, v in DARK_LEVELS.items()},
        "clahe_ms":         {"mean": round(float(np.mean(clahe_ms)), 2),
                             "median": round(float(np.median(clahe_ms)), 2)},
        "seed":             args.seed,
    }
    (OUT / "summary.json").write_text(json.dumps({"meta": meta, "results": summary}, indent=2))

    with (OUT / "summary.csv").open("w") as f:
        cols = ["condition", "images", "person_instances", "mean_brightness", "precision",
                "recall", "f1", "ap50", "tp", "fp", "fn", "clahe_applied_pct"]
        f.write(",".join(cols) + "\n")
        for c, s in summary.items():
            f.write(",".join(str(s.get(k, "")) if k != "condition" else c for k in cols) + "\n")

    # Sample figure images: dark vs dark+CLAHE with detections drawn
    sd = OUT / "samples"
    sd.mkdir(exist_ok=True)
    for stem, vs in samples.items():
        for c, (img, res) in vs.items():
            vis = img.copy()
            for b, cf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                if cf >= args.conf:
                    x1, y1, x2, y2 = map(int, b)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 80), 2)
                    cv2.putText(vis, f"{cf:.2f}", (x1, max(12, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 80), 1)
            cv2.imwrite(str(sd / f"{stem}__{c}.jpg"), vis)

    print(json.dumps({"meta": meta, "results": summary}, indent=2))


if __name__ == "__main__":
    main()

"""
Functional check: held-weapon escalation (knife/scissors + person → CRITICAL).

Picks COCO val2017 images labelled with both a person and a knife or scissors,
runs the edge's own detect_frame() on each, and writes a review sheet where a
human marks whether the weapon is actually held. Agreement between the system's
decision and the human label gives the escalation accuracy for Chapter Four.

Run from the edge/ folder:
    python experiments/held_weapon_review.py --n 40
Then open experiments/results/held_weapon/review.html, mark each image, and
press "Export CSV". Re-run with --score <exported.csv> to compute accuracy.
"""
import argparse
import csv
import html
import json
import sys
from pathlib import Path

import cv2

EDGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EDGE))

DATA   = EDGE / "experiments" / "data"
IMAGES = DATA / "val2017"
LABELS = DATA / "coco" / "labels" / "val2017"
OUT    = EDGE / "experiments" / "results" / "held_weapon"
WEAPON_IDS = {"43": "knife", "76": "scissors"}


def build(n: int, conf: float):
    import detector as det
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)

    picked = []
    for lf in sorted(LABELS.glob("*.txt")):
        cls = {l.split()[0] for l in lf.read_text().splitlines() if l.strip()}
        if "0" in cls and cls & WEAPON_IDS.keys():
            picked.append(lf.stem)
        if len(picked) >= n:
            break

    rows = []
    for stem in picked:
        img = cv2.imread(str(IMAGES / f"{stem}.jpg"))
        annotated, dets, threat = det.detect_frame(img, confidence_threshold=conf)
        weapons = [d for d in dets if d["label"] in ("knife", "scissors")]
        cv2.imwrite(str(OUT / "images" / f"{stem}.jpg"), annotated)
        rows.append({
            "image":            stem,
            "weapon_detected":  bool(weapons),
            "weapon_labels":    "|".join(d["label"] for d in weapons),
            "system_held":      any(d["threat"] == "CRITICAL" for d in weapons),
            "frame_threat":     threat,
        })

    with (OUT / "system_decisions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["human_held"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "human_held": ""})

    cards = "\n".join(f"""
      <figure data-image="{r['image']}">
        <img src="images/{r['image']}.jpg" loading="lazy" alt="COCO {r['image']}">
        <figcaption>
          <b>{r['image']}</b> · frame: <span class="t {r['frame_threat']}">{r['frame_threat']}</span><br>
          system: {html.escape(r['weapon_labels'] or 'no weapon detected')}
          → <b>{'HELD (CRITICAL)' if r['system_held'] else ('UNATTENDED (HIGH)' if r['weapon_detected'] else '—')}</b>
          <div class="q">Actually held by a person?
            <label><input type="radio" name="{r['image']}" value="Y"> Yes</label>
            <label><input type="radio" name="{r['image']}" value="N"> No</label>
            <label><input type="radio" name="{r['image']}" value="X"> No weapon visible</label>
          </div>
        </figcaption>
      </figure>""" for r in rows)

    (OUT / "review.html").write_text(f"""<!doctype html><meta charset="utf-8">
<title>Held-weapon review</title>
<style>
 body{{font:14px system-ui;margin:16px;background:#fafafa;color:#222}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}}
 figure{{margin:0;background:#fff;border:1px solid #ddd;border-radius:8px;overflow:hidden}}
 img{{width:100%;display:block}} figcaption{{padding:8px;line-height:1.5}}
 .q{{margin-top:6px;padding-top:6px;border-top:1px solid #eee}}
 .t{{padding:1px 6px;border-radius:4px;color:#fff;background:#888}}
 .CRITICAL{{background:#c62828}} .HIGH{{background:#ef6c00}} .MEDIUM{{background:#f9a825}} .LOW{{background:#2e7d32}}
 button{{font:inherit;padding:8px 14px;margin:8px 0 16px}}
</style>
<h1>Held-weapon escalation — manual review</h1>
<p>For each image, mark whether a knife/scissors is <b>actually held by a person</b>. Then export.</p>
<button onclick="exportCsv()">Export CSV</button> <span id="n"></span>
<div class="grid">{cards}</div>
<script>
const sys = {json.dumps({r['image']: r for r in rows})};
function exportCsv(){{
  const lines=["image,weapon_detected,system_held,frame_threat,human_held"];
  for(const [k,r] of Object.entries(sys)){{
    const v=(document.querySelector(`input[name="${{k}}"]:checked`)||{{}}).value||"";
    lines.push([k,r.weapon_detected,r.system_held,r.frame_threat,v].join(","));
  }}
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([lines.join("\\n")],{{type:"text/csv"}}));
  a.download="held_weapon_labels.csv"; a.click();
}}
document.addEventListener("change",()=>{{
  document.getElementById("n").textContent=
    document.querySelectorAll("input:checked").length+" / {len(rows)} marked";
}});
</script>""")
    print(f"{len(rows)} images → {OUT/'review.html'}")
    print(f"weapon detected in {sum(r['weapon_detected'] for r in rows)}, "
          f"escalated to CRITICAL in {sum(r['system_held'] for r in rows)}")


def score(path: Path):
    rows = [r for r in csv.DictReader(path.open()) if r["human_held"] in ("Y", "N")]
    tp = sum(r["system_held"] == "True" and r["human_held"] == "Y" for r in rows)
    fp = sum(r["system_held"] == "True" and r["human_held"] == "N" for r in rows)
    fn = sum(r["system_held"] != "True" and r["human_held"] == "Y" for r in rows)
    tn = sum(r["system_held"] != "True" and r["human_held"] == "N" for r in rows)
    p = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
    result = {"labelled": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
              "accuracy": round((tp + tn) / max(len(rows), 1), 3),
              "precision": round(p, 3), "recall": round(r_, 3)}
    (OUT / "score.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--score", type=Path)
    a = ap.parse_args()
    score(a.score) if a.score else build(a.n, a.conf)

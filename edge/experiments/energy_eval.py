"""
Experiment: energy use of the baseline vs proposed edge configuration.

The same recorded footage (CAVIAR surveillance clips, played at native frame rate)
is processed by each configuration in a fresh process:

  capture_only        no inference at all — overhead floor (decode, encode, stream)
  baseline            detector on every frame, motion gating off, CLAHE off   (thesis baseline)
  interval_1s         detector once per second, motion gating off, CLAHE off
  gating_every_frame  detector on every frame only when motion detected, CLAHE on
  proposed            detector once per second only when motion detected, CLAHE on (as deployed)

Measured per run:
  CodeCarbon energy (process tracking mode; CPU via cpu_load estimate, RAM via power model)
  Process CPU time (user + system seconds, measured by the OS)
  Inferences run, frames skipped by motion gating, CLAHE activations
  Person-presence seconds detected (for checking that gating does not miss people)

Run from the edge/ folder (stop the live edge service and other heavy work first):
    python experiments/energy_eval.py --reps 2
Outputs → experiments/results/energy/
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

EDGE    = Path(__file__).resolve().parent.parent
DATA    = EDGE / "experiments" / "data" / "caviar"
OUT     = EDGE / "experiments" / "results" / "energy"
VIDEO   = DATA / "caviar_scenario.mp4"

CAVIAR = "https://homepages.inf.ed.ac.uk/rbf"
CLIPS = [
    f"{CAVIAR}/CAVIARDATA1/Walk1/Walk1.mpg",
    f"{CAVIAR}/CAVIARDATA1/Browse1/Browse1.mpg",
    f"{CAVIAR}/CAVIARDATA1/Rest_InChair/Rest_InChair.mpg",
    f"{CAVIAR}/CAVIARDATA1/LeftBag/LeftBag.mpg",
    f"{CAVIAR}/CAVIARDATA1/Browse_WhileWaiting1/Browse_WhileWaiting1.mpg",
    f"{CAVIAR}/CAVIARDATA2/WalkByShop1cor/WalkByShop1cor.mpg",
    f"{CAVIAR}/CAVIARDATA2/OneLeaveShop1cor/OneLeaveShop1cor.mpg",
    f"{CAVIAR}/CAVIARDATA2/EnterExitCrossingPaths1cor/EnterExitCrossingPaths1cor.mpg",
]

CONFIGS = {
    "capture_only":       {"DETECTION_INTERVAL": "1e9", "MOTION_FILTER": "false", "LOW_LIGHT_THRESHOLD": "0"},
    "baseline":           {"DETECTION_INTERVAL": "0",   "MOTION_FILTER": "false", "LOW_LIGHT_THRESHOLD": "0"},
    "interval_1s":        {"DETECTION_INTERVAL": "1.0", "MOTION_FILTER": "false", "LOW_LIGHT_THRESHOLD": "0"},
    "gating_every_frame": {"DETECTION_INTERVAL": "0",   "MOTION_FILTER": "true",  "LOW_LIGHT_THRESHOLD": "80"},
    "proposed":           {"DETECTION_INTERVAL": "1.0", "MOTION_FILTER": "true",  "LOW_LIGHT_THRESHOLD": "80"},
}


# ── Footage ───────────────────────────────────────────────────────────────────

def build_video():
    import cv2
    DATA.mkdir(parents=True, exist_ok=True)
    if VIDEO.exists():
        return
    writer, size, fps, manifest = None, None, 25.0, []
    for url in CLIPS:
        local = DATA / url.rsplit("/", 1)[-1]
        if not local.exists():
            print(f"downloading {local.name}")
            urllib.request.urlretrieve(url, local)
        cap = cv2.VideoCapture(str(local))
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                size = (frame.shape[1], frame.shape[0])
                writer = cv2.VideoWriter(str(VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            writer.write(frame)
            n += 1
        manifest.append({"clip": local.name, "frames": n, "seconds": round(n / fps, 1)})
    writer.release()
    (DATA / "manifest.json").write_text(json.dumps(
        {"fps": fps, "size": size, "clips": manifest,
         "total_seconds": round(sum(c["frames"] for c in manifest) / fps, 1),
         "source": "EC Funded CAVIAR project/IST 2001 37540 (CC BY-SA)"}, indent=2))


# ── Worker: one configuration, one run ────────────────────────────────────────

def worker(config: str, out_file: Path):
    import psutil
    os.chdir(EDGE)
    sys.path.insert(0, str(EDGE))
    import numpy as np
    import detector as det

    start = time.time()
    presence = set()        # video seconds in which a person was detected
    clahe_hits = [0]

    real_detect, real_enhance = det.detect_frame, det._enhance_low_light

    def counting_enhance(frame):
        out = real_enhance(frame)
        if out is not frame:
            clahe_hits[0] += 1
        return out

    firearm_secs = set()    # CAVIAR has no guns, so any firearm detection is a false alarm
    fps = json.loads((DATA / "manifest.json").read_text())["fps"]

    def video_second():
        # Position in the footage, not wall time: a config that falls behind
        # real time must still be compared clip-for-clip.
        return int(cam._frames_read / fps)

    def logging_detect(frame, **kw):
        res = real_detect(frame, **kw)
        if any(d["label"] in ("person", "loitering") for d in res[1]):
            presence.add(video_second())
        if any(d["label"] in ("handgun", "rifle", "shotgun") for d in res[1]):
            firearm_secs.add(video_second())
        return res

    det._enhance_low_light = counting_enhance
    det.detect_frame       = logging_detect

    cam = det.CameraDetector(
        camera_index=str(VIDEO),
        detection_interval=float(os.environ["DETECTION_INTERVAL"]),
        alert_cooldown=15.0,
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.35")),
        loitering_seconds=30.0,
        motion_filter=os.environ["MOTION_FILTER"] == "true",
    )
    proc = psutil.Process()
    cpu0 = proc.cpu_times()
    start = time.time()
    cam.start(asyncio.new_event_loop())
    while not cam.finished:
        if not cam._thread.is_alive():
            raise RuntimeError(f"Capture thread stopped early — could not read {VIDEO}")
        time.sleep(0.5)
    cpu1 = proc.cpu_times()

    stats = cam.get_energy_stats()
    lat   = cam.get_latency_stats()
    out_file.write_text(json.dumps({
        "config":           config,
        "settings":         CONFIGS[config],
        "cpu_seconds":      round((cpu1.user - cpu0.user) + (cpu1.system - cpu0.system), 2),
        "clahe_activations": clahe_hits[0],
        "presence_seconds": sorted(presence),
        "false_firearm_seconds": sorted(firearm_secs),
        "weapons_model_loaded": det._weapons_model is not None,
        "energy":           stats,
        "inference_ms":     lat["inference"],
    }, indent=2))
    os._exit(0)   # skip the slow CodeCarbon shutdown; figures are already frozen


# ── Orchestrator ──────────────────────────────────────────────────────────────

def clip_windows(manifest: dict) -> list:
    """(name, start_s, end_s) of each clip within the concatenated scenario."""
    t, out = 0.0, []
    for c in manifest["clips"]:
        out.append((c["clip"], t, t + c["seconds"]))
        t += c["seconds"]
    return out


def detection_checks(presence: set, ref: set, clips: list, tol: int = 2) -> dict:
    """Compare a run's person-seconds with the baseline's.

    recall_tolerant: share of baseline person-seconds with a detection within ±tol s
    clips_detected:  clips in which a person was detected at least once
    first_detection: seconds from clip start to first detection, per clip
    """
    hit = sum(1 for s in ref if any(abs(s - p) <= tol for p in presence))
    first = {}
    for name, a, b in clips:
        inside = [p for p in presence if a <= p < b]
        first[name] = round(min(inside) - a, 1) if inside else None
    return {
        "recall_tolerant": hit / max(len(ref), 1),
        "clips_detected":  sum(1 for v in first.values() if v is not None),
        "first_detection": first,
    }


def summarise(runs: list, manifest: dict) -> dict:
    import statistics as st
    ref = set()
    for r in runs:
        if r["config"] == "baseline":
            ref |= set(r["presence_seconds"])
    clips = clip_windows(manifest)
    for r in runs:
        r["checks"] = detection_checks(set(r["presence_seconds"]), ref, clips)
    out = {}
    for cfg in CONFIGS:
        rs = [r for r in runs if r["config"] == cfg]
        if not rs:
            continue

        def m(fn):
            vals = [fn(r) for r in rs]
            return {"mean": round(st.mean(vals), 6), "sd": round(st.stdev(vals), 6) if len(vals) > 1 else 0.0}

        found = set().union(*(set(r["presence_seconds"]) for r in rs))
        per_run_recall = [len(set(r["presence_seconds"]) & ref) / max(len(ref), 1) for r in rs]
        out[cfg] = {
            "runs":               len(rs),
            "settings":           CONFIGS[cfg],
            "duration_s":         m(lambda r: r["energy"]["session_seconds"]),
            "energy_wh":          m(lambda r: r["energy"]["energy_wh"]),
            "cpu_energy_wh":      m(lambda r: r["energy"]["cpu_kwh"] * 1000),
            "ram_energy_wh":      m(lambda r: r["energy"]["ram_kwh"] * 1000),
            "co2_g":              m(lambda r: r["energy"]["co2_grams"]),
            "cpu_seconds":        m(lambda r: r["cpu_seconds"]),
            "inferences":         m(lambda r: r["energy"]["inferences_run"]),
            "frames_read":        m(lambda r: r["energy"]["frames_read"]),
            "skipped_by_motion":  m(lambda r: r["energy"]["frames_skipped_motion"]),
            "clahe_activations":  m(lambda r: r["clahe_activations"]),
            "inference_ms_median": m(lambda r: r["inference_ms"]["median_ms"] or 0),
            "person_seconds_detected": m(lambda r: len(r["presence_seconds"])),
            "presence_recall_vs_baseline": {"mean": round(st.mean(per_run_recall), 4),
                                            "sd": round(st.stdev(per_run_recall), 4) if len(rs) > 1 else 0.0},
            "presence_recall_tolerant_2s": m(lambda r: r["checks"]["recall_tolerant"]),
            "clips_with_person_detected": m(lambda r: r["checks"]["clips_detected"]),
            "false_firearm_seconds": m(lambda r: len(r["false_firearm_seconds"])),
            "weapons_model_loaded": all(r["weapons_model_loaded"] for r in rs),
            "first_detection_s_per_clip": {
                name: [r["checks"]["first_detection"][name] for r in rs] for name, _, _ in clips},
        }
    base = out.get("baseline")
    if base:
        for cfg, s in out.items():
            for k in ("energy_wh", "cpu_energy_wh", "cpu_seconds", "co2_g"):
                b = base[k]["mean"]
                s[f"{k}_change_vs_baseline_pct"] = round(100 * (s[k]["mean"] - b) / b, 1) if b else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--out",    help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        worker(args.worker, Path(args.out))
        return

    build_video()
    manifest = json.loads((DATA / "manifest.json").read_text())
    print(f"Scenario: {manifest['total_seconds']} s of CAVIAR footage, {len(manifest['clips'])} clips")

    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for rep in range(args.reps):                      # interleave configs to spread drift
        for cfg in args.configs:
            f = OUT / f"run_{cfg}_{rep + 1}.json"
            env = {**os.environ, **CONFIGS[cfg], "CODECARBON_TRACKING_MODE": "process"}
            print(f"rep {rep + 1}/{args.reps}  {cfg} …", flush=True)
            subprocess.run([sys.executable, __file__, "--worker", cfg, "--out", str(f)],
                           env=env, check=True, cwd=EDGE)
            runs.append(json.loads(f.read_text()))
            time.sleep(5)                             # let the machine settle

    summary = {"scenario": manifest, "results": summarise(runs, manifest)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()

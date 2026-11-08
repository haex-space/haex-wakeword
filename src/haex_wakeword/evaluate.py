"""Recall on held-out speakers and false accepts per hour on held-out speech.

False accepts are counted the way haex would act on them: the detector fires when a window's
score reaches the threshold, and stays quiet for the next 2 s (it is listening then). Every
recording of the test sources (MLS test, FLEURS dev in six languages) is scored as a stream.
SC-004 of holzi spec 066 asks for at most 1 false accept in 8 hours at medium sensitivity; the
threshold for "medium" is the lowest one that stays within 0.125 per hour here.

What this cannot measure: SC-003 (9 of 10 at up to 3 m with real voices). That needs real
recordings and a check in the app (T065 in holzi).
"""

import json
from pathlib import Path

import numpy as np
import torch

from . import config, negatives
from .model import Classifier
from .train import _scores, load_clips

THRESHOLDS = [round(t, 3) for t in np.concatenate([np.arange(0.3, 0.9, 0.05), np.arange(0.9, 1.0, 0.01), [0.995, 0.998, 0.999]])]
REFRACTORY_FRAMES = 25  # 2 s
SC004_PER_HOUR = 1 / 8


def triggers(scores: np.ndarray, threshold: float) -> int:
    count, quiet_until = 0, -1
    for i in np.flatnonzero(scores >= threshold):
        if i > quiet_until:
            count += 1
            quiet_until = i + REFRACTORY_FRAMES
    return count


def evaluate(paths: config.Paths, run: str) -> dict:
    run_dir = paths.runs / run
    model = Classifier()
    model.load_state_dict(torch.load(run_dir / "model.pt"))
    model.eval()

    pos = _scores(model, load_clips(paths, "test", "positive"))
    conf = _scores(model, load_clips(paths, "test", "confusable"))
    speech = negatives.speech(paths, "test")
    hours = speech.seconds / 3600

    # Score every window once, then count triggers per recording.
    per_recording = []
    for s, e in speech.recordings:
        if e - s < config.WINDOW_FRAMES:
            continue
        starts = np.arange(s, e - config.WINDOW_FRAMES + 1)
        per_recording.append(_scores(model, speech.windows(starts)))

    rows = []
    for t in THRESHOLDS:
        fa = sum(triggers(r, t) for r in per_recording)
        rows.append(
            {
                "threshold": t,
                "recall": float((pos >= t).mean()),
                "confusable_accept": float((conf >= t).mean()),
                "false_accepts": fa,
                "false_accepts_per_hour": fa / hours,
            }
        )
    medium = next((r for r in rows if r["false_accepts_per_hour"] <= SC004_PER_HOUR), None)
    result = {
        "test_hours": hours,
        "test_positive_windows": int(len(pos)),
        "test_confusable_windows": int(len(conf)),
        "rows": rows,
        "medium": medium,
    }
    (run_dir / "evaluation.json").write_text(json.dumps(result, indent=2))
    print(f"{hours:.1f} h held-out speech, {len(pos)} positive windows of held-out speakers")
    print(f"{'thr':>6} {'recall':>7} {'conf':>6} {'FA':>5} {'FA/h':>7}")
    for r in rows:
        print(
            f"{r['threshold']:6.3f} {r['recall']:7.3f} {r['confusable_accept']:6.3f} "
            f"{r['false_accepts']:5d} {r['false_accepts_per_hour']:7.3f}"
        )
    print("medium:", json.dumps(medium))
    return result

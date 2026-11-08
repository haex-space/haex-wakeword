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
from .train import _scores, load_clips, positives_of, positives_of_lone

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


def per_group(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """The highest score of each group."""
    out = np.full(groups.max() + 1, -1.0, np.float32)
    np.maximum.at(out, groups, scores)
    return out[out >= 0]


def evaluate(paths: config.Paths, run: str) -> dict:
    run_dir = paths.runs / run
    model = Classifier()
    model.load_state_dict(torch.load(run_dir / "model.pt"))
    model.eval()

    phrases_file = run_dir / "phrases.txt"
    phrases = phrases_file.read_text() if phrases_file.exists() else "all"
    pos_windows, pos_groups = positives_of(paths, "test", phrases)
    pos = _scores(model, pos_windows)
    conf = _scores(model, load_clips(paths, "test", "confusable"))
    # Per spoken clip: the detector fires if any of the clip's windows reaches the threshold.
    pos = per_group(pos, pos_groups)
    conf = per_group(conf, np.load(paths.clips / "test-confusable-groups.npy"))
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
    lone_accept = None
    if phrases == "two-words":
        lone_windows, lone_groups = positives_of_lone(paths, "test")
        lone = per_group(_scores(model, lone_windows), lone_groups)
        lone_accept = {str(r["threshold"]): float((lone >= r["threshold"]).mean()) for r in rows}
    result = {
        "phrases": phrases,
        "lone_hex_accept": lone_accept,
        "test_hours": hours,
        "test_positive_clips": int(len(pos)),
        "test_confusable_clips": int(len(conf)),
        "rows": rows,
        "medium": medium,
    }
    (run_dir / "evaluation.json").write_text(json.dumps(result, indent=2))
    print(f"{hours:.1f} h held-out speech, {len(pos)} spoken positives of held-out speakers")
    print(f"{'thr':>6} {'recall':>7} {'conf':>6} {'FA':>5} {'FA/h':>7}")
    for r in rows:
        print(
            f"{r['threshold']:6.3f} {r['recall']:7.3f} {r['confusable_accept']:6.3f} "
            f"{r['false_accepts']:5d} {r['false_accepts_per_hour']:7.3f}"
        )
    print("medium:", json.dumps(medium))
    return result

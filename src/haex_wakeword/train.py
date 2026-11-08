"""Training: positives against confusables, speech in six languages and noise, with hard negatives.

Every batch mixes positives, confusables, random speech and noise windows, and from the first
mining round on a share of the windows the model scored highest so far ("hard negatives").
Negatives weigh more than positives, and more as training goes on: a false accept in a quiet
room is worse than a repeated "Hey Hex" (holzi spec 066 SC-004 asks for at most one in 8 hours).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import config, negatives
from .model import Classifier

BATCH = {"positive": 192, "confusable": 96, "speech": 512, "noise": 64, "hard": 192}
MINE_EVERY = 2000
MINE_POOL = 1_000_000
HARD_KEEP = 50_000


def load_clips(paths: config.Paths, split: str, kind: str) -> np.ndarray:
    file = paths.features / "clips" / f"{split}-{kind}.npy"
    if not file.exists():
        raise FileNotFoundError(f"run `hw clips` first: {file}")
    return np.load(file)


def _scores(model: Classifier, windows: np.ndarray, batch: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            out.append(torch.sigmoid(model(torch.from_numpy(windows[i : i + batch].astype(np.float32)))).numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def positives_of(paths: config.Paths, split: str, phrases: str) -> tuple[np.ndarray, np.ndarray]:
    """Positive windows and their groups; `phrases` "two-words" keeps only "Hey Hex" and its forms."""
    from .clips import group_texts, two_words

    windows = load_clips(paths, split, "positive")
    groups = np.load(paths.features / "clips" / f"{split}-positive-groups.npy")
    if phrases == "all":
        return windows, groups
    keep_group = np.array([two_words(t) for t in group_texts(paths, split, "positive")])
    keep = keep_group[groups]
    return windows[keep], groups[keep]


def positives_of_lone(paths: config.Paths, split: str) -> tuple[np.ndarray, np.ndarray]:
    from .clips import group_texts, two_words

    windows = load_clips(paths, split, "positive")
    groups = np.load(paths.features / "clips" / f"{split}-positive-groups.npy")
    lone = ~np.array([two_words(t) for t in group_texts(paths, split, "positive")])[groups]
    return windows[lone], groups[lone]


def train(paths: config.Paths, run: str, steps: int = 20_000, seed: int = 0, phrases: str = "all") -> Path:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    run_dir = paths.runs / run
    run_dir.mkdir(parents=True, exist_ok=True)

    positives, _ = positives_of(paths, "train", phrases)
    confusables = load_clips(paths, "train", "confusable")
    test_positives, _ = positives_of(paths, "test", phrases)
    if phrases == "two-words":
        # The lone "Hex" must not fire then: it becomes a negative.
        lone, _ = positives_of_lone(paths, "train")
        confusables = np.concatenate([confusables, lone])
    speech = negatives.speech(paths, "train")
    noise = negatives.noise(paths)
    print(
        f"train: {len(positives)} positive, {len(confusables)} confusable windows, "
        f"{speech.seconds / 3600:.1f} h speech, {noise.seconds / 3600:.1f} h noise"
    )

    model = Classifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    hard = np.zeros((0, config.WINDOW_FRAMES, config.EMBEDDING_SIZE), np.float16)
    log = []
    started = time.time()

    for step in range(1, steps + 1):
        parts = [
            positives[rng.integers(0, len(positives), BATCH["positive"])],
            confusables[rng.integers(0, len(confusables), BATCH["confusable"])],
            speech.sample(rng, BATCH["speech"]),
            noise.sample(rng, BATCH["noise"]),
        ]
        if len(hard):
            parts.append(hard[rng.integers(0, len(hard), BATCH["hard"])])
        x = torch.from_numpy(np.concatenate(parts).astype(np.float32))
        n_pos = BATCH["positive"]
        y = torch.zeros(len(x))
        y[:n_pos] = 1
        # Negatives weigh 1 at the start and up to 5 at the end.
        neg_weight = 1 + 4 * step / steps
        weight = torch.full((len(x),), neg_weight)
        weight[:n_pos] = 1
        loss = (loss_fn(model(x), y) * weight).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()

        if step % MINE_EVERY == 0 or step == steps:
            pool_speech = speech.sample(rng, MINE_POOL)
            pool = np.concatenate([pool_speech, confusables])
            scores = _scores(model, pool)
            top = np.argsort(scores)[-HARD_KEEP:]
            hard = pool[top]
            recall = float((_scores(model, test_positives) >= 0.5).mean())
            accepted = float((scores[: len(pool_speech)] >= 0.5).mean())
            entry = {
                "step": step,
                "loss": float(loss.detach()),
                "test_recall@0.5": recall,
                "speech_windows_over_0.5": accepted,
                "hardest_score": float(scores[top[0]]),
                "minutes": (time.time() - started) / 60,
            }
            log.append(entry)
            print(json.dumps(entry))

    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "phrases.txt").write_text(phrases)
    (run_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    return run_dir

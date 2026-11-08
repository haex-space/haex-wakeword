"""Negative windows cut from embedded long recordings, never across two recordings."""

from pathlib import Path

import numpy as np

from . import audio, config, data, features


class Frames:
    """The frames of several embedded sources with the window starts that stay in one recording."""

    def __init__(self, files: list[Path]):
        frames, valid, offset, seconds = [], [], 0, 0.0
        self.recordings: list[tuple[int, int]] = []
        for file in files:
            stored = np.load(file)
            f = stored["frames"]
            starts = list(stored["starts"]) + [len(f)]
            seconds += float(stored["seconds"])
            for s, e in zip(starts[:-1], starts[1:]):
                self.recordings.append((offset + s, offset + e))
                if e - s >= config.WINDOW_FRAMES:
                    valid.append(np.arange(offset + s, offset + e - config.WINDOW_FRAMES + 1))
            frames.append(f)
            offset += len(f)
        self.frames = np.concatenate(frames) if frames else np.zeros((0, config.EMBEDDING_SIZE), np.float16)
        self.starts = np.concatenate(valid) if valid else np.zeros(0, np.int64)
        self.seconds = seconds

    def __len__(self) -> int:
        return len(self.starts)

    def windows(self, starts: np.ndarray) -> np.ndarray:
        index = starts[:, None] + np.arange(config.WINDOW_FRAMES)[None, :]
        return self.frames[index]

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.windows(self.starts[rng.integers(0, len(self.starts), n)])


def speech(paths: config.Paths, split: str) -> Frames:
    files = [paths.features / "speech" / f"{s.name}.npz" for s in data.SOURCES if s.split == split]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"run `hw negatives` first: {missing[0]}")
    return Frames(files)


def noise(paths: config.Paths) -> Frames:
    """The noises of openSLR 28, embedded once. Used for training only: they also go into the
    backgrounds of the positives, so they are no fair test."""
    out = paths.features / "noise.npz"
    if not out.exists():
        embedder = features.Embedder(features.model_path(paths.models))
        frames, starts, total, seconds = [], [], 0, 0.0
        for file in data.noise_files(data.rirs_noises(paths)):
            clip = audio.read(file)
            seconds += len(clip) / config.SAMPLE_RATE
            emb = embedder.embed(clip)
            if len(emb):
                starts.append(total)
                frames.append(emb.astype(np.float16))
                total += len(emb)
        np.savez(out, frames=np.concatenate(frames), starts=np.asarray(starts), seconds=seconds)
    return Frames([out])

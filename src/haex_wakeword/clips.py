"""Augmented, embedded windows of the synthesized clips.

Each clip is placed in a stretch of background (room noise or just microphone hiss), optionally
convolved with a room impulse response and set to a random level, then embedded like haex does.
The windows that end when the clip ends (and up to two frames later) are what the detector must
fire on for a positive; for a confusable every window around its end is a negative.

Output, per split (speakers decide, see synth.split_of): `features/clips/<split>-<kind>.npy`,
float16 [n, 16, 96], and `<split>-<kind>-groups.npy`, the spoken clip each window belongs to.
"""

import json
import random
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
from tqdm import tqdm

from . import audio, config, data, features, synth

PRE_ROLL = int(2.2 * config.SAMPLE_RATE)  # the clip ends after at least 2 s: a full window fits
POST_ROLL = int(0.5 * config.SAMPLE_RATE)
POSITIVE_ENDS = (0, 1, 2, 3, 4)  # windows ending this many frames after the clip's end
CONFUSABLE_ENDS = (-2, 0, 2, 4)

_state: dict = {}


def _init(root: str, seed: int) -> None:
    paths = config.Paths(Path(root))
    noise_root = data.rirs_noises(paths)
    _state["paths"] = paths
    _state["noises"] = data.noise_files(noise_root)
    _state["rirs"] = data.rir_files(noise_root)
    _state["embedder"] = features.Embedder(features.model_path(paths.models), threads=1)
    _state["seed"] = seed


@lru_cache(maxsize=256)
def _load(path: str) -> np.ndarray:
    return audio.read(Path(path))


def _background(rng: random.Random, length: int, level: float) -> np.ndarray:
    hiss = np.random.default_rng(rng.randrange(2**32)).standard_normal(length).astype(np.float32)
    hiss *= 10 ** (rng.uniform(-70, -50) / 20)
    if rng.random() < 0.75:
        noise = _load(str(rng.choice(_state["noises"])))
        if len(noise) < length:
            noise = np.tile(noise, length // max(1, len(noise)) + 1)
        start = rng.randrange(len(noise) - length + 1)
        noise = noise[start : start + length]
        rms = np.sqrt(np.mean(noise**2)) + 1e-9
        snr = rng.uniform(3, 30)
        hiss += noise / rms * level / 10 ** (snr / 20)
    return hiss


def augment(clip: np.ndarray, rng: random.Random) -> tuple[np.ndarray, int]:
    """The clip in a room: returns the audio and the sample where the clip ends."""
    if rng.random() < 0.5:
        rir = _load(str(rng.choice(_state["rirs"])))
        rir = rir[np.argmax(np.abs(rir)) :][: config.SAMPLE_RATE]  # from the direct sound on
        wet = fftconvolve(clip, rir)[: len(clip) + config.SAMPLE_RATE // 4]
        clip = (wet / (np.max(np.abs(wet)) + 1e-9)).astype(np.float32)
    else:
        clip = clip / (np.max(np.abs(clip)) + 1e-9)
    peak = 10 ** (rng.uniform(-32, -3) / 20)
    clip = clip * peak
    level = float(np.sqrt(np.mean(clip**2)))
    start = PRE_ROLL + rng.randrange(int(0.4 * config.SAMPLE_RATE))
    total = start + len(clip) + POST_ROLL
    out = _background(rng, total, level)
    out[start : start + len(clip)] += clip
    return np.clip(out, -1, 1), start + len(clip)


def windows_at(embeddings: np.ndarray, end_sample: int, offsets) -> list[np.ndarray]:
    """Windows whose last frame is the first frame covering `end_sample`, shifted by `offsets`."""
    last = -(-end_sample // config.HOP) - (config.FIRST_REAL_FRAME + 1)
    out = []
    for offset in offsets:
        k = last + offset
        if k - config.WINDOW_FRAMES + 1 >= 0 and k < len(embeddings):
            out.append(embeddings[k - config.WINDOW_FRAMES + 1 : k + 1])
    return out


def _job(job: tuple[str, str, int]) -> list[np.ndarray]:
    path, kind, index = job
    rng = random.Random(f"{_state['seed']}:{path}:{index}")
    clip = _load(path)
    mixed, end = augment(clip, rng)
    emb = _state["embedder"].embed(mixed)
    offsets = POSITIVE_ENDS if kind == "positive" else CONFUSABLE_ENDS
    return [w.astype(np.float16) for w in windows_at(emb, end, offsets)]


def build(paths: config.Paths, variants: int = 3, seed: int = 0, workers: int = 12) -> None:
    out_dir = paths.features / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, directory in (("positive", paths.positives), ("confusable", paths.confusables)):
        clips = synth.read_index(directory)
        for split in ("train", "test"):
            chosen = [c for c in clips if c.split == split]
            jobs = [(str(directory / c.path), kind, v) for c in chosen for v in range(variants)]
            result: list[np.ndarray] = []
            groups: list[int] = []
            with Pool(workers, initializer=_init, initargs=(str(paths.root), seed)) as pool:
                for group, windows in enumerate(
                    tqdm(pool.imap(_job, jobs, chunksize=32), total=len(jobs), desc=f"{split} {kind}")
                ):
                    result.extend(windows)
                    groups.extend([group] * len(windows))
            array = np.stack(result) if result else np.zeros((0, config.WINDOW_FRAMES, config.EMBEDDING_SIZE), np.float16)
            np.save(out_dir / f"{split}-{kind}.npy", array)
            # Which windows belong to one spoken clip: the detector fires if any of them does.
            np.save(out_dir / f"{split}-{kind}-groups.npy", np.asarray(groups, np.int64))
            texts = [chosen[i // variants].text for i in range(len(jobs))]
            (out_dir / f"{split}-{kind}-texts.json").write_text(json.dumps(texts, ensure_ascii=False))
            print(f"{split} {kind}: {len(chosen)} clips -> {len(array)} windows")


def group_texts(paths: config.Paths, split: str, kind: str, variants: int = 3) -> list[str]:
    """The text of every group (spoken clip) of a split. Built features before texts were saved
    are matched through the index: clips are only ever appended to it."""
    out_dir = paths.features / "clips"
    saved = out_dir / f"{split}-{kind}-texts.json"
    if saved.exists():
        return json.loads(saved.read_text())
    groups = np.load(out_dir / f"{split}-{kind}-groups.npy")
    directory = paths.positives if kind == "positive" else paths.confusables
    chosen = [c for c in synth.read_index(directory) if c.split == split]
    return [chosen[g // variants].text for g in range(int(groups.max()) + 1)]


def two_words(text: str) -> bool:
    """"Hey Hex" and its forms in the other languages, not the lone "Hex"."""
    return len(text.replace(",", " ").split()) >= 2

"""Reading, writing and resampling mono audio at the pipeline's 16 kHz."""

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import SAMPLE_RATE


def to_rate(audio: np.ndarray, rate: int, target: int = SAMPLE_RATE) -> np.ndarray:
    if rate == target:
        return audio.astype(np.float32, copy=False)
    g = gcd(rate, target)
    return resample_poly(audio, target // g, rate // g).astype(np.float32)


def read(path: Path) -> np.ndarray:
    """Mono float32 at 16 kHz."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return to_rate(audio.mean(axis=1), rate)


def write(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")


def trim(audio: np.ndarray, threshold: float = 0.01, margin: int = 800) -> np.ndarray:
    """Cuts leading and trailing silence, keeping `margin` samples on each side."""
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if len(loud) == 0:
        return audio[:0]
    return audio[max(0, loud[0] - margin) : loud[-1] + margin]

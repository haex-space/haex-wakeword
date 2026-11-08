"""Embeddings of Google's speech_embedding/1, computed the way haex computes them at run time.

The graph puts 11,120 zeros in front of whatever it gets and returns one embedding per 1,280
samples. Output frame j sees input samples [j*1280 - 11120, (j+1)*1280), 12,400 samples, so it
depends on real audio only from j = 9 on. haex runs a stream: every 80 ms it feeds the last
10 * 1,280 samples and keeps the last frame, which is frame 9 and sees nothing but real audio.

Over a long recording, every frame from 9 on sees exactly the same samples as that streaming frame
did, so [`embed`] over a clip yields the run-time embeddings of the clip's frames from 9 on.
[`stream`] computes them one by one, as haex does; the tests check that both agree.
"""

import hashlib
from functools import cached_property
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from . import config

STREAM_SAMPLES = (config.FIRST_REAL_FRAME + 1) * config.HOP


def model_path(cache: Path) -> Path:
    """Downloads the pinned model once and checks its hash."""
    path = Path(
        hf_hub_download(
            config.EMBEDDING_REPO,
            config.EMBEDDING_FILE,
            revision=config.EMBEDDING_REVISION,
            cache_dir=cache,
        )
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != config.EMBEDDING_SHA256:
        raise RuntimeError(f"{path}: sha256 {digest}, expected {config.EMBEDDING_SHA256}")
    return path


class Embedder:
    def __init__(self, path: Path, threads: int = 0):
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )

    @cached_property
    def _input(self) -> str:
        return self._session.get_inputs()[0].name

    def raw(self, audio: np.ndarray) -> np.ndarray:
        """The graph's output for `audio` (mono 16 kHz float32 in [-1, 1]): [len // 1280, 96]."""
        if audio.ndim != 1:
            raise ValueError("mono audio expected")
        if len(audio) < config.HOP:
            return np.zeros((0, config.EMBEDDING_SIZE), np.float32)
        (out,) = self._session.run(None, {self._input: audio.astype(np.float32, copy=False)})
        return out

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Run-time embeddings of `audio`: frame k here is the frame haex computes once the stream
        has reached sample (k + 10) * 1280 of `audio`. Empty for audio shorter than 12,800 samples.
        """
        return self.raw(audio)[config.FIRST_REAL_FRAME :]

    def stream(self, audio: np.ndarray) -> np.ndarray:
        """The same embeddings, one call per 80 ms like haex. Slow; for tests and evaluation."""
        frames = []
        for end in range(STREAM_SAMPLES, len(audio) + 1, config.HOP):
            frames.append(self.raw(audio[end - STREAM_SAMPLES : end])[-1])
        if not frames:
            return np.zeros((0, config.EMBEDDING_SIZE), np.float32)
        return np.stack(frames)


def windows(embeddings: np.ndarray, frames: int = config.WINDOW_FRAMES) -> np.ndarray:
    """Every run of `frames` consecutive embeddings: [n - frames + 1, frames, 96]."""
    if len(embeddings) < frames:
        return np.zeros((0, frames, embeddings.shape[1]), np.float32)
    view = np.lib.stride_tricks.sliding_window_view(embeddings, frames, axis=0)
    return np.ascontiguousarray(view.transpose(0, 2, 1))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

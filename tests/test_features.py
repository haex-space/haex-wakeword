"""The batch embeddings the training uses must equal the streaming ones haex computes."""

from pathlib import Path

import numpy as np
import pytest

from haex_wakeword import config, features

CACHE = Path(__file__).resolve().parent.parent / "models"


@pytest.fixture(scope="module")
def embedder():
    return features.Embedder(features.model_path(CACHE))


def speechlike(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * config.SAMPLE_RATE)) / config.SAMPLE_RATE
    voiced = 0.3 * np.sin(2 * np.pi * (120 + 40 * np.sin(2 * np.pi * 3 * t)) * t)
    return (voiced * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) + 0.02 * rng.standard_normal(len(t))).astype(
        np.float32
    )


def test_the_graph_returns_one_embedding_per_hop(embedder):
    audio = speechlike(2.0)
    assert embedder.raw(audio).shape == (len(audio) // config.HOP, config.EMBEDDING_SIZE)


def test_batch_embeddings_equal_the_streaming_ones(embedder):
    audio = speechlike(3.0, seed=1)
    batch = embedder.embed(audio)
    stream = embedder.stream(audio)
    assert batch.shape == stream.shape
    assert batch.shape[0] == len(audio) // config.HOP - config.FIRST_REAL_FRAME
    np.testing.assert_allclose(batch, stream, atol=1e-4)


def test_a_frame_does_not_see_audio_before_its_receptive_field(embedder):
    audio = speechlike(2.0, seed=2)
    changed = audio.copy()
    # Frame 9 sees samples [400, 12800); a change before 400 must not reach it.
    changed[:400] = 0.9
    np.testing.assert_allclose(embedder.raw(audio)[9], embedder.raw(changed)[9], atol=1e-5)


def test_windows_are_consecutive_runs():
    embeddings = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
    w = features.windows(embeddings, frames=16)
    assert w.shape == (5, 16, 3)
    np.testing.assert_array_equal(w[2], embeddings[2:18])

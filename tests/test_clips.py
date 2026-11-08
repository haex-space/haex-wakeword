"""Window placement and the plausibility gate, without models."""

import numpy as np

from haex_wakeword import clips, config, synth


def test_the_positive_window_is_the_first_that_covers_the_end():
    emb = np.arange(60, dtype=np.float32)[:, None].repeat(config.EMBEDDING_SIZE, axis=1)
    end = 30 * config.HOP + 5  # frame k covers up to (k + 10) * HOP; first covering: k = 21
    (w,) = clips.windows_at(emb, end, (0,))
    assert w[-1, 0] == 21 and w[0, 0] == 6


def test_windows_that_do_not_fit_are_left_out():
    emb = np.zeros((20, config.EMBEDDING_SIZE), np.float32)
    assert clips.windows_at(emb, 12 * config.HOP, (0, 1)) == []


def test_a_clip_with_a_long_pause_inside_is_not_plausible():
    rate = config.SAMPLE_RATE
    voiced = 0.3 * np.sin(np.arange(rate // 3) * 0.2).astype(np.float32)
    ok = np.concatenate([voiced, np.zeros(rate // 10, np.float32), voiced])
    babble = np.concatenate([voiced, np.zeros(rate // 2, np.float32), voiced])
    assert synth.plausible(ok, "Hey Hex")
    assert not synth.plausible(babble, "Hey Hex")


def test_a_clip_much_longer_than_its_words_is_not_plausible():
    long = 0.3 * np.sin(np.arange(3 * config.SAMPLE_RATE) * 0.2).astype(np.float32)
    assert not synth.plausible(long, "Hex")


def test_the_split_is_stable_and_holds_out_some_speakers():
    speakers = [f"piper:x:{i}" for i in range(2000)]
    test = sum(synth.split_of(s) == "test" for s in speakers)
    assert 200 < test < 400
    assert synth.split_of("pocket:abc") == synth.split_of("pocket:abc")

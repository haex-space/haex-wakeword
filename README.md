# haex-wakeword

Training pipeline for the default wake word of [haex](https://github.com/haex-space/haex-vault)'s
voice assistant: **"Hex"** and **"Hey Hex"**, said [hɛks] (the product name *Haex*).

The detector follows the architecture of [openWakeWord](https://github.com/dscripka/openWakeWord):
a frozen speech embedding model turns 16 kHz audio into one 96-value embedding every 80 ms, and a
small classifier decides on the last 16 embeddings (1.28 s) whether the wake word just ended.
This repository trains only that classifier.

## Pieces and licences

| Piece | Source | Licence |
|---|---|---|
| Pipeline code | this repository | Apache-2.0 |
| `tools/piper_worker.py` | runs Piper as its own process | GPL-3.0-or-later (Piper is GPL-3.0; nothing else imports it) |
| Embedding model | Google `speech_embedding/1` ([Lin et al. 2020](https://arxiv.org/abs/2002.01322)) as ONNX: [`androiddrew/google-speech-embedding-onnx`](https://huggingface.co/androiddrew/google-speech-embedding-onnx) @ `e38c9f38` | Apache-2.0 |
| Positives and confusables, Piper | voices trained on Thorsten-Voice (CC0), LibriTTS-R, VCTK (CC-BY-4.0), single-speaker voices marked CC0, public domain or Apache-2.0 (`config.PIPER_VOICES`) | as listed per voice |
| Positives and confusables, Pocket TTS | Kyutai Pocket TTS (CC-BY-4.0) through the sherpa-onnx export [`haex-space/pocket-tts-sherpa-onnx`](https://huggingface.co/haex-space/pocket-tts-sherpa-onnx), voices: Kyutai's [voice donations](https://huggingface.co/kyutai/tts-voices) | CC-BY-4.0 / CC0 |
| Speech negatives | [Multilingual LibriSpeech](https://www.openslr.org/94/) (de, fr, es, it, pt; English via `parler-tts/mls_eng`), [FLEURS](https://huggingface.co/datasets/google/fleurs) | CC-BY-4.0 |
| Noise and room impulse responses | [openSLR 28 RIRS_NOISES](https://www.openslr.org/28/) | Apache-2.0 |

Not used, on purpose:
- openWakeWord's own `melspectrogram.onnx` and `embedding_model.onnx`. Its README puts all included models under CC BY-NC-SA, and whether that covers these two is an open question in issues #313, #338, #346, #348 and #357.
- Its pre-trained wake word models and ACAV100M, both CC BY-NC-SA.
- Piper's MLS voices. In a trial they babbled for seconds instead of saying two words.
- Any voice with an NC, SA or unclear licence.

No voice is cloned from a person who did not give it for that use.

## Steps

```sh
uv sync --extra pocket
uv run hw noise                                   # 1.3 GB: impulse responses and noises
uv run hw negatives                               # about 4 GB of speech, embedded to features/
uv run hw synth positive   --engine piper  --count 1500
uv run hw synth positive   --engine pocket --count 1500
uv run hw synth confusable --engine piper  --count 700
uv run hw synth confusable --engine pocket --count 700
uv run hw clips                                   # augment and embed the clips
uv run hw train --run run1
uv run hw evaluate --run run1
uv run hw export --run run1                       # runs/run1/hex.onnx and card.json
```

Every step can be stopped and started again; finished work is kept.

## How haex runs it

Every 80 ms haex feeds the last 12,800 samples (16 kHz mono, float32 in [-1, 1]) to the embedding
model and keeps the last output frame. That frame sees only real audio: the graph puts 11,120 zeros
in front of its input, so frames 0–8 of any call see some of those zeros. The last 16 such frames
go into `hex.onnx`. `tests/test_features.py` checks that the batch embeddings the training uses
equal these streaming ones.

## Splits and measurement

- **Split by speaker:** 15 % of all synthesized speakers are held out, chosen by a hash of the speaker id.
- **False accepts:** counted on MLS test and FLEURS dev audio, whose speakers the training never heard. The audio is scored as a stream, with 2 s of quiet after every trigger.
- **Target:** at most one false accept in 8 hours at medium sensitivity (haex spec 066, SC-004). `card.json` records the threshold that meets it on this data.
- **Not measured here:** recall with real voices at a distance (SC-003). That needs real recordings and is measured in the app.

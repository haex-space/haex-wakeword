# SPDX-License-Identifier: GPL-3.0-or-later
"""Runs Piper (GPL-3.0) as its own process for the Apache-2.0 pipeline.

Run with `uv run --no-project --with piper-tts==1.8.0 python tools/piper_worker.py`. Reads one
JSON job per line on stdin:

    {"model": "...onnx", "text": "Hey Hex", "speaker": 12, "length_scale": 1.1,
     "noise_scale": 0.667, "noise_w_scale": 0.8, "out": ".../clip.wav"}

writes the WAV at the voice's own sample rate and answers each job with one JSON line
{"out": ..., "ok": true} or {"out": ..., "ok": false, "error": "..."}.
"""

import json
import sys
import wave

from piper import PiperVoice, SynthesisConfig

voices: dict[str, PiperVoice] = {}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    job = json.loads(line)
    try:
        voice = voices.get(job["model"])
        if voice is None:
            voice = voices[job["model"]] = PiperVoice.load(job["model"])
        config = SynthesisConfig(
            speaker_id=job.get("speaker"),
            length_scale=job.get("length_scale"),
            noise_scale=job.get("noise_scale"),
            noise_w_scale=job.get("noise_w_scale"),
        )
        with wave.open(job["out"], "wb") as out:
            voice.synthesize_wav(job["text"], out, syn_config=config)
        print(json.dumps({"out": job["out"], "ok": True}), flush=True)
    except Exception as error:  # noqa: BLE001 - one bad job must not end the worker
        print(json.dumps({"out": job.get("out"), "ok": False, "error": str(error)}), flush=True)

"""The ONNX model haex runs, and what haex needs to know about it."""

import json

import numpy as np
import onnxruntime as ort
import torch

from . import config, features
from .model import Classifier, Exported


def embedding_for_tract(paths: config.Paths, out) -> None:
    """The embedding model as haex ships it: tract (pure-Rust ONNX, 0.22) rejects a Pad node whose
    optional constant value is left out, so the value 0 (the default) is written in. Checked to
    give exactly the original's output."""
    import onnx
    from onnx import numpy_helper

    original = features.model_path(paths.models)
    model = onnx.load(str(original))
    for node in model.graph.node:
        if node.op_type == "Pad" and len(node.input) >= 3 and node.input[2] == "":
            name = f"{node.name.strip('/')}_constant_value"
            model.graph.initializer.append(numpy_helper.from_array(np.array(0.0, np.float32), name))
            node.input[2] = name
    onnx.checker.check_model(model)
    onnx.save(model, str(out))
    audio = np.random.default_rng(1).standard_normal(4 * 12_800).astype(np.float32) * 0.1
    (a,) = ort.InferenceSession(str(out)).run(None, {"waveform": audio})
    (b,) = ort.InferenceSession(str(original)).run(None, {"waveform": audio})
    np.testing.assert_array_equal(a, b)


def export(paths: config.Paths, run: str) -> None:
    run_dir = paths.runs / run
    classifier = Classifier()
    classifier.load_state_dict(torch.load(run_dir / "model.pt"))
    exported = Exported(classifier).eval()
    out = run_dir / "hex.onnx"
    example = torch.zeros(1, config.WINDOW_FRAMES, config.EMBEDDING_SIZE)
    torch.onnx.export(
        exported,
        (example,),
        str(out),
        input_names=["embeddings"],
        output_names=["probability"],
        dynamic_axes={"embeddings": {0: "batch"}, "probability": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    # The ONNX graph must give what the PyTorch model gives.
    probe = np.random.default_rng(0).standard_normal((8, config.WINDOW_FRAMES, config.EMBEDDING_SIZE)).astype(np.float32)
    (onnx_out,) = ort.InferenceSession(str(out)).run(None, {"embeddings": probe})
    with torch.no_grad():
        torch_out = exported(torch.from_numpy(probe)).numpy()
    np.testing.assert_allclose(onnx_out, torch_out, atol=1e-5)

    embedding = run_dir / "speech_embedding.onnx"
    embedding_for_tract(paths, embedding)

    evaluation_file = run_dir / "evaluation.json"
    evaluation = json.loads(evaluation_file.read_text()) if evaluation_file.exists() else {}
    rules_file = run_dir / "rules.json"
    rules = json.loads(rules_file.read_text()) if rules_file.exists() else {}
    card = {
        "file": out.name,
        "sha256": features.sha256(out),
        "bytes": out.stat().st_size,
        "input": f"embeddings float32 [batch, {config.WINDOW_FRAMES}, {config.EMBEDDING_SIZE}]",
        "output": "probability float32 [batch, 1]",
        "embedding_model": {
            "repo": config.EMBEDDING_REPO,
            "revision": config.EMBEDDING_REVISION,
            "file": config.EMBEDDING_FILE,
            "sha256": config.EMBEDDING_SHA256,
            "shipped_file": embedding.name,
            "shipped_sha256": features.sha256(embedding),
            "shipped_change": "Pad node: the omitted constant value is written in as 0 (tract 0.22 needs it); output unchanged",
            "stream": "every 80 ms feed the last 12,800 samples (16 kHz mono) and keep the last frame",
        },
        "phrases": ["Hex", "Hey Hex"],
        "pronunciation": "[hɛks]",
        "window_frames": config.WINDOW_FRAMES,
        "refractory_frames": config.REFRACTORY_FRAMES,
        "sensitivity": rules.get("sensitivity", config.SENSITIVITY),
        "evaluation": {"test_hours": rules.get("test_hours", evaluation.get("test_hours"))},
        "licence": "Apache-2.0",
    }
    (run_dir / "card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    (run_dir / "NOTICE").write_text(NOTICE)
    (run_dir / "README.md").write_text(model_card(card))
    print(json.dumps(card, indent=2, ensure_ascii=False))


NOTICE = """haex wake word "Hex" / "Hey Hex"
Copyright haex-space. Licensed under the Apache License, Version 2.0.

speech_embedding.onnx is Google speech_embedding/1 (TensorFlow1 / speech-embedding / version 1),
published by Google under the Apache License, Version 2.0:
https://www.kaggle.com/models/google/speech-embedding/TensorFlow1/speech-embedding/1
Lin, Kilgour, Roblek, Sharifi: "Training Keyword Spotters with Limited and Synthesized Speech
Data", arXiv:2002.01322.
It comes from the ONNX conversion androiddrew/google-speech-embedding-onnx (revision
e38c9f3880a56ba46e5094af7b69e78d3d2c4b7b, Apache-2.0, by Drew Bednar), whose NOTICE lists the
conversion's modifications. Modification here: the omitted constant value of the one Pad node is
written in as 0, which leaves the output unchanged.

hex.onnx was trained by haex-space with https://github.com/haex-space/haex-wakeword on speech
synthesized with Piper voices (CC0, CC-BY-4.0, public domain, Apache-2.0 data) and Kyutai Pocket
TTS (CC-BY-4.0) with Kyutai's CC0 voice donations, and on Multilingual LibriSpeech and FLEURS
(CC-BY-4.0) and openSLR 28 noises (Apache-2.0).
"""


def model_card(card: dict) -> str:
    rows = "\n".join(
        f"| {name} | {r['windows']} | {r['threshold']} | "
        f"{r.get('recall_hey_hex', float('nan')):.1%} | {r.get('recall_hex', float('nan')):.1%} | "
        f"{r.get('false_accepts_per_hour', float('nan')):.2f} |"
        for name, r in card["sensitivity"].items()
    )
    hours = card["evaluation"]["test_hours"]
    return f"""---
license: apache-2.0
library_name: onnx
pipeline_tag: audio-classification
tags:
  - wake-word
  - keyword-spotting
  - onnx
---

# haex wake word: "Hex" / "Hey Hex"

The default wake word of [haex](https://github.com/haex-space/haex-vault)'s voice assistant, said
[hɛks]. Trained with [haex-wakeword](https://github.com/haex-space/haex-wakeword).

## Files

- `speech_embedding.onnx`: Google `speech_embedding/1` (Apache-2.0) with its log-mel front end, 16 kHz
  mono float32 in, one 96-value embedding per 1,280 samples out. See `NOTICE`.
- `hex.onnx`: the classifier. Input `embeddings` float32 `[batch, 16, 96]`, output `probability`
  float32 `[batch, 1]`.
- `card.json`: hashes, the streaming contract and the sensitivities below.

## Running it

Every 80 ms, feed the last 12,800 samples to `speech_embedding.onnx` and keep the last output frame
(it sees only real audio). Give the last 16 frames to `hex.onnx`. Fire when `windows` consecutive
probabilities reach `threshold`, then stay quiet for 2 s.

## Measured

On {hours:.1f} h of held-out speech (MLS test, FLEURS dev; six languages; speakers never heard in
training) and held-out synthetic speakers saying the wake word:

| Sensitivity | Windows | Threshold | "Hey Hex" | "Hex" alone | False accepts per hour |
|---|---|---|---|---|---|
{rows}

"medium" is haex's default. These figures come from synthesized voices. Recall with real voices
at a distance is measured in the app.

## Licence

Apache-2.0. Training data and their licences: see `NOTICE` and the pipeline's README.
"""

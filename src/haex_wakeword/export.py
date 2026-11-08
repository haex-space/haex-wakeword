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
    medium = evaluation.get("medium")
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
        "threshold_medium": medium["threshold"] if medium else None,
        "evaluation": {
            "test_hours": evaluation.get("test_hours"),
            "medium": medium,
        },
        "licence": "Apache-2.0",
    }
    (run_dir / "card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    print(json.dumps(card, indent=2, ensure_ascii=False))

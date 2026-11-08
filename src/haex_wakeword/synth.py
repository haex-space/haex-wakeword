"""Synthesized clips: the wake word (positives) and words that sound close (confusables).

Two engines, many speakers, none of them imitated without consent:
- Piper voices trained on CC0/CC-BY corpora with hundreds of speakers each. Piper is GPL-3.0, so
  it runs as its own process (tools/piper_worker.py) and this package never imports it.
- Pocket TTS (the sherpa-onnx export haex ships) with Kyutai's CC0 voice donations as references.

Every clip is written at 16 kHz with its silence trimmed, and listed in `index.jsonl` next to it
with engine, voice, speaker, language and text. The speaker decides the train/test split.
"""

import hashlib
import json
import random
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

from . import audio, config

PIPER_REVISION = "e7229f7d33a10e145e109b886d661984201ac3c4"
PIPER_PACKAGE = "piper-tts==1.8.0"
POCKET_REPO = "haex-space/pocket-tts-sherpa-onnx"
POCKET_REVISION = "d8f5915ed8bd94e1aa3dbd5a219360b3bb966190"
POCKET_FILES = [
    "lm_flow.int8.onnx",
    "lm_main.int8.onnx",
    "encoder.onnx",
    "decoder.int8.onnx",
    "text_conditioner.onnx",
    "vocab.json",
    "token_scores.json",
]
POCKET_VOICES_REVISION = "323332d33f997de8394f24a193e1a76df720e01a"
# Share of speakers held out for evaluation.
TEST_SHARE = 0.15


@dataclass
class Clip:
    path: str
    kind: str  # "positive" | "confusable"
    engine: str
    voice: str
    speaker: str
    language: str
    text: str

    @property
    def split(self) -> str:
        return split_of(self.speaker)


def split_of(speaker: str) -> str:
    """"test" for a stable TEST_SHARE of speakers, "train" for the rest."""
    bucket = int(hashlib.sha256(speaker.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < TEST_SHARE else "train"


def read_index(directory: Path) -> list[Clip]:
    index = directory / "index.jsonl"
    if not index.exists():
        return []
    return [Clip(**json.loads(line)) for line in index.read_text().splitlines() if line]


def _append(directory: Path, clip: Clip) -> None:
    with (directory / "index.jsonl").open("a") as f:
        f.write(json.dumps(asdict(clip), ensure_ascii=False) + "\n")


def texts(kind: str, language: str) -> list[str]:
    if kind == "positive":
        return config.PHRASES[language]
    # Confusables exist in German and English; the other languages' speakers use the English list.
    return config.CONFUSABLES.get(language, config.CONFUSABLES["en"])


def plausible(clip: np.ndarray, text: str) -> bool:
    """Whether a synthesized clip can be what was asked for: not too short, not much longer than
    the words need, and no long silence inside (a voice that babbles on after a pause)."""
    words = len(text.replace(",", " ").split())
    if len(clip) < config.SAMPLE_RATE // 10 or len(clip) > config.SAMPLE_RATE * (0.5 + 0.7 * words):
        return False
    block = config.SAMPLE_RATE // 50  # 20 ms
    loud = [
        np.sqrt(np.mean(clip[i : i + block] ** 2)) > 0.01 for i in range(0, len(clip) - block, block)
    ]
    quiet = longest = 0
    for is_loud in loud:
        quiet = 0 if is_loud else quiet + 1
        longest = max(longest, quiet)
    return longest * block <= 0.45 * config.SAMPLE_RATE


# ---------------------------------------------------------------- Piper


def piper_model(paths: config.Paths, voice: config.PiperVoice) -> tuple[Path, int]:
    """The voice's model file and its number of speakers."""
    files = [
        Path(
            hf_hub_download(
                config.PIPER_REPO,
                f"{voice.path}/{voice.key}{suffix}",
                revision=PIPER_REVISION,
                cache_dir=paths.models,
            )
        )
        for suffix in (".onnx", ".onnx.json")
    ]
    speakers = int(json.loads(files[1].read_text()).get("num_speakers", 1))
    return files[0], speakers


def piper(paths: config.Paths, kind: str, per_voice: int, seed: int = 0) -> None:
    """`per_voice` clips of `kind` from every Piper voice, random speaker, phrase and pace."""
    rng = random.Random(seed)
    out_dir = paths.positives if kind == "positive" else paths.confusables
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {c.path for c in read_index(out_dir)}
    worker = Path(__file__).resolve().parents[2] / "tools" / "piper_worker.py"
    with tempfile.TemporaryDirectory() as tmp:
        process = subprocess.Popen(
            ["uv", "run", "--no-project", "--with", PIPER_PACKAGE, "python", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdin and process.stdout
        try:
            for voice in config.PIPER_VOICES:
                model, speakers = piper_model(paths, voice)
                for n in tqdm(range(per_voice), desc=f"piper {kind} {voice.key}"):
                    speaker = rng.randrange(speakers)
                    text = rng.choice(texts(kind, voice.language))
                    rel = f"piper/{voice.key}/{kind}-{seed}-{n:06d}.wav"
                    params = {
                        "length_scale": rng.uniform(0.75, 1.35),
                        "noise_scale": rng.uniform(0.4, 0.9),
                        "noise_w_scale": rng.uniform(0.5, 1.0),
                    }
                    if rel in done:
                        continue
                    raw = Path(tmp) / "clip.wav"
                    job = {
                        "model": str(model),
                        "text": text,
                        "speaker": speaker if speakers > 1 else None,
                        "out": str(raw),
                        **params,
                    }
                    process.stdin.write(json.dumps(job) + "\n")
                    answer = json.loads(process.stdout.readline())
                    if not answer["ok"]:
                        print(f"piper: {answer['error']}", file=sys.stderr)
                        continue
                    clip = audio.trim(audio.read(raw))
                    if not plausible(clip, text):
                        continue
                    audio.write(out_dir / rel, clip)
                    _append(
                        out_dir,
                        Clip(rel, kind, "piper", voice.key, f"piper:{voice.key}:{speaker}", voice.language, text),
                    )
        finally:
            process.stdin.close()
            process.wait()


# ---------------------------------------------------------------- Pocket


def pocket_model_dir(paths: config.Paths, language: str) -> Path:
    files = [
        Path(
            hf_hub_download(
                POCKET_REPO, f"{language}/{name}", revision=POCKET_REVISION, cache_dir=paths.models
            )
        )
        for name in POCKET_FILES
    ]
    return files[0].parent


def pocket_references(paths: config.Paths) -> list[tuple[str, Path]]:
    """Kyutai's CC0 voice donations, the cleaned ("enhanced") recording of each voice."""
    names = [
        f
        for f in list_repo_files(config.POCKET_VOICES_REPO, revision=POCKET_VOICES_REVISION)
        if f.startswith(config.POCKET_VOICES_DIR + "/") and f.endswith("_enhanced.wav")
    ]
    return [
        (
            Path(name).name.removesuffix("_enhanced.wav"),
            Path(
                hf_hub_download(
                    config.POCKET_VOICES_REPO, name, revision=POCKET_VOICES_REVISION, cache_dir=paths.models
                )
            ),
        )
        for name in tqdm(names, desc="voice donations")
    ]


def _pocket_engine(model_dir: Path, threads: int):
    import sherpa_onnx

    pocket = sherpa_onnx.OfflineTtsPocketModelConfig(
        *(str(model_dir / name) for name in POCKET_FILES)
    )
    model = sherpa_onnx.OfflineTtsModelConfig()
    model.pocket = pocket
    model.num_threads = threads
    tts_config = sherpa_onnx.OfflineTtsConfig()
    tts_config.model = model
    return sherpa_onnx.OfflineTts(tts_config)


def pocket(
    paths: config.Paths,
    kind: str,
    per_language: int,
    languages: Iterator[str] | None = None,
    seed: int = 0,
    threads: int = 4,
) -> None:
    """`per_language` clips of `kind` per Pocket language, each with a random donated voice."""
    import sherpa_onnx

    rng = random.Random(seed)
    out_dir = paths.positives if kind == "positive" else paths.confusables
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {c.path for c in read_index(out_dir)}
    references = pocket_references(paths)
    loaded: dict[str, np.ndarray] = {}
    for language in languages or config.PHRASES:
        tts = _pocket_engine(pocket_model_dir(paths, language), threads)
        for n in tqdm(range(per_language), desc=f"pocket {kind} {language}"):
            voice, reference = rng.choice(references)
            text = rng.choice(texts(kind, language))
            temperature = rng.uniform(0.5, 0.9)
            rel = f"pocket/{language}/{kind}-{seed}-{n:06d}.wav"
            if rel in done:
                continue
            if voice not in loaded:
                loaded[voice] = audio.read(reference)[: 15 * config.SAMPLE_RATE]
            generation = sherpa_onnx.GenerationConfig()
            generation.reference_audio = loaded[voice].tolist()
            generation.reference_sample_rate = config.SAMPLE_RATE
            generation.num_steps = 1
            generation.extra = {"temperature": f"{temperature:.2f}", "max_reference_audio_len": "15"}
            generated = tts.generate(text, generation)
            clip = audio.trim(audio.to_rate(np.asarray(generated.samples, np.float32), generated.sample_rate))
            if not plausible(clip, text):
                continue
            audio.write(out_dir / rel, clip)
            _append(out_dir, Clip(rel, kind, "pocket", language, f"pocket:{voice}", language, text))

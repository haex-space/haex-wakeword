"""Negative audio and noise: downloads, and the embeddings of long speech recordings.

Speech that is not the wake word comes from Multilingual LibriSpeech (CC-BY-4.0) and FLEURS
(CC-BY-4.0); MLS splits are speaker-disjoint, so training uses MLS train data and the false-accept
measurement uses MLS test and FLEURS dev, whose speakers the model never heard. Room impulse
responses and noises come from openSLR 28 (Apache-2.0).

Long recordings are stored as embedding frames (float16, [n, 96]) per source; the training cuts
its windows from them, which keeps 100 hours at under a gigabyte.
"""

import io
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from . import audio, config, features

# English is not part of facebook/multilingual_librispeech; parler-tts/mls_eng is the same MLS
# English data (CC-BY-4.0) as parquet.
REVISIONS = {
    config.MLS_REPO: "2e83e61823b4c47dcbcb1980bb88601274127609",
    "parler-tts/mls_eng": "faf6604dcf0bb9ec0d9c280a140ad74da9c931b0",
    config.FLEURS_REPO: "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
}


@dataclass(frozen=True)
class Source:
    """A long-speech source: one parquet file of MLS or one FLEURS archive."""

    name: str
    split: str  # "train" | "test"
    kind: str  # "mls" | "fleurs"
    language: str
    file: str
    repo: str = config.MLS_REPO


SOURCES = [
    # Training: six shards of German and English MLS train (about 30 h each), one of the others.
    # (The 9 h subsets of MLS are part of its train data, so they are not listed again.)
    Source("mls-de-train0", "train", "mls", "de", "german/train-00000-of-00063.parquet"),
    Source("mls-en-train0", "train", "mls", "en", "data/train-00000-of-01416.parquet", "parler-tts/mls_eng"),
    # More natural speech: the first runs (2026-11-08) fired on ordinary words with 98 h.
    *(
        Source(f"mls-de-train{i}", "train", "mls", "de", f"german/train-{i:05d}-of-00063.parquet")
        for i in range(1, 6)
    ),
    *(
        Source(f"mls-en-train{i}", "train", "mls", "en", f"data/train-{i:05d}-of-01416.parquet", "parler-tts/mls_eng")
        for i in range(1, 6)
    ),
    Source("mls-fr-train0", "train", "mls", "fr", "french/train-00000-of-00034.parquet"),
    Source("mls-es-train0", "train", "mls", "es", "spanish/train-00000-of-00030.parquet"),
    Source("mls-it-train0", "train", "mls", "it", "italian/train-00000-of-00008.parquet"),
    Source("mls-pt-train0", "train", "mls", "pt", "portuguese/train-00000-of-00006.parquet"),
    # Measurement: speakers the model never heard.
    Source("mls-de-test", "test", "mls", "de", "german/test-00000-of-00001.parquet"),
    Source("mls-en-test", "test", "mls", "en", "data/test-00000-of-00001.parquet", "parler-tts/mls_eng"),
    *(
        Source(f"fleurs-{lang}-dev", "test", "fleurs", lang, f"data/{code}/audio/dev.tar.gz", config.FLEURS_REPO)
        for lang, code in config.FLEURS_CONFIGS.items()
    ),
]


def download(paths: config.Paths, source: Source) -> Path:
    return Path(
        hf_hub_download(
            source.repo,
            source.file,
            repo_type="dataset",
            revision=REVISIONS[source.repo],
            cache_dir=paths.models / "datasets",
        )
    )


def recordings(path: Path, source: Source):
    """Yields the source's recordings as 16 kHz float32 arrays."""
    if source.kind == "mls":
        table = pq.ParquetFile(path)
        for batch in table.iter_batches(columns=["audio"], batch_size=64):
            for item in batch.column("audio").to_pylist():
                data, rate = sf.read(io.BytesIO(item["bytes"]), dtype="float32", always_2d=True)
                yield audio.to_rate(data.mean(axis=1), rate)
    else:
        with tarfile.open(path) as archive:
            for member in archive:
                if member.isfile() and member.name.endswith(".wav"):
                    file = archive.extractfile(member)
                    assert file is not None
                    data, rate = sf.read(io.BytesIO(file.read()), dtype="float32", always_2d=True)
                    yield audio.to_rate(data.mean(axis=1), rate)


def embed_source(paths: config.Paths, source: Source, embedder: features.Embedder) -> Path:
    """Embeds every recording of `source` and stores the frames with recording boundaries:
    `frames` [n, 96] float16 and `starts` (index of each recording's first frame)."""
    out = paths.features / "speech" / f"{source.name}.npz"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    frames, starts, total = [], [], 0
    seconds = 0.0
    for clip in tqdm(recordings(download(paths, source), source), desc=source.name, unit="rec"):
        seconds += len(clip) / config.SAMPLE_RATE
        emb = features_of(embedder, clip)
        if len(emb) == 0:
            continue
        starts.append(total)
        frames.append(emb.astype(np.float16))
        total += len(emb)
    np.savez(out, frames=np.concatenate(frames), starts=np.asarray(starts), seconds=seconds)
    return out


def features_of(embedder: features.Embedder, clip: np.ndarray) -> np.ndarray:
    """Run-time embeddings of `clip`, embedded in long recordings in chunks of 10 minutes that
    overlap by the receptive field, so no frame is lost or computed with zeros."""
    chunk = 600 * config.SAMPLE_RATE
    context = features.STREAM_SAMPLES - config.HOP
    if len(clip) <= chunk + context:
        return embedder.embed(clip)
    parts, start = [], 0
    while start + context < len(clip):
        piece = clip[start : start + chunk + context]
        parts.append(embedder.embed(piece))
        start += chunk
    return np.concatenate(parts)


# ---------------------------------------------------------------- noise


def rirs_noises(paths: config.Paths) -> Path:
    """openSLR 28, unpacked: simulated and real room impulse responses, point-source noises."""
    target = paths.noise
    if (target / "RIRS_NOISES").exists():
        return target / "RIRS_NOISES"
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "rirs_noises.zip"
    if not archive.exists():
        urlretrieve(config.RIRS_NOISES_URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(target)
    archive.unlink()
    return target / "RIRS_NOISES"


def noise_files(root: Path) -> list[Path]:
    return sorted((root / "pointsource_noises").glob("*.wav")) + sorted(
        (root / "real_rirs_isotropic_noises").glob("*noise*.wav")
    )


def rir_files(root: Path) -> list[Path]:
    return sorted((root / "simulated_rirs").rglob("*.wav")) + [
        p for p in sorted((root / "real_rirs_isotropic_noises").glob("*.wav")) if "noise" not in p.name
    ]

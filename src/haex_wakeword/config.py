"""Everything the pipeline pins: the feature model, the window, the phrases and the data sources.

The feature model and the window must match what haex runs (holzi spec 066, research R4): haex
feeds the same model the same audio and gives the classifier the same number of embeddings.
"""

from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16_000

# Google speech_embedding/1 (Apache-2.0) as an ONNX graph with the log-mel front end inside.
EMBEDDING_REPO = "androiddrew/google-speech-embedding-onnx"
EMBEDDING_REVISION = "e38c9f3880a56ba46e5094af7b69e78d3d2c4b7b"
EMBEDDING_FILE = "model.onnx"
EMBEDDING_SHA256 = "bfef7f0ace055206df8aeab960bb0fbe6e44898a8549ef79ae6a1ed6591169d3"
EMBEDDING_SIZE = 96
# One embedding per 1,280 samples (80 ms). The graph puts 11,120 zeros in front of its input, so an
# output frame j sees input samples [j*1280 - 11120, (j+1)*1280). Frame j depends on real audio
# only from j = 9 on (see features.py).
HOP = 1280
LEADING_ZEROS = 11_120
FIRST_REAL_FRAME = 9

# The classifier sees this many embeddings: 1.28 s, as in openWakeWord.
WINDOW_FRAMES = 16

# How haex decides: fire when `windows` consecutive scores reach `threshold`, then stay quiet for
# 2 s. "medium" is the default and meets holzi spec 066 SC-004 (at most 1 false accept in 8 h) on
# the held-out speech; chosen on run7 (2026-11-09).
SENSITIVITY = {
    "low": {"windows": 3, "threshold": 0.9},
    "medium": {"windows": 2, "threshold": 0.98},
    "high": {"windows": 2, "threshold": 0.5},
}
REFRACTORY_FRAMES = 25

# Said "Hex" like German "Hexe" without the e, [hɛks] (operator 2026-11-08). The written form
# the synthesizers get, not the brand spelling "Haex".
PHRASES = {
    "de": ["Hex", "Hex!", "Hey Hex", "Hey, Hex!", "Hey Hex?", "He, Hex"],
    "en": ["Hex", "Hex!", "Hey Hex", "Hey, Hex!", "Hey Hex?"],
    # Speakers of the other languages say the same name.
    "fr": ["Hex", "Hey Hex", "Hé, Hex"],
    "es": ["Hex", "Hey Hex", "Ey, Hex"],
    "it": ["Hex", "Hey Hex", "Ehi, Hex"],
    "pt": ["Hex", "Hey Hex", "Ei, Hex"],
}

# Words and phrases that sound close and must not trigger. Never anything that starts with or
# contains [hɛks] ("Hexe", "Hexagon", "Hecks"): the detector fires when the word ends, and at
# that moment such a word sounds exactly like the wake word, so it would only teach it noise.
CONFUSABLES = {
    "de": [
        "Hecht", "Hecke", "Echse", "Text", "Keks", "Sechs", "Klecks", "Fax",
        "Hessen", "Hektik", "Häcksler", "extra", "Experte", "hey", "Hey Max", "Hey Rex",
        "Hey Alex", "Hey Rick", "Hey Leute", "He, Max", "Hallo", "jetzt", "Haxe",
        "Hey Siri", "Alexa", "Okay", "Hier", "Helm", "Herd",
    ],
    "en": [
        "hacks", "heck", "decks", "checks", "next", "text", "six", "sex", "hats", "Rex",
        "Hey Max", "Hey Rex", "Hey Alex", "Hey Lex", "Hey there", "Hey you", "hey", "Hector",
        "headset", "Alexa", "Hey Siri", "Okay Google", "hello", "help", "Hicks",
    ],
}


# What the first runs (2026-11-08) fired on in held-out speech: "he said"/"he went" in English,
# the same few German passages. Synthesized as confusables in a second round.
CONFUSABLES_ROUND2 = {
    "de": [
        "Er sagte", "Heidi", "hielt", "Woyzeck", "Hawkens", "kicherte", "Hecke", "hechelt",
        "Hessen", "Keks", "Text", "sechs", "Klecks", "Fex", "Rex", "Max", "Lex", "Specht",
        "Heck", "Hecht", "Häcksel", "He, Max", "He, Rex", "Hey Jens", "Hey Lex", "Hey Rex",
        "Hey Klaus", "Hey Fax", "Hey Text", "Hey Keks", "Hey Sechs", "Hey Echs",
    ],
    "en": [
        "he said", "he says", "he sets", "he asked", "he went", "he left", "he sat", "he's next",
        "he checks", "he did", "she said", "they said", "he declared", "he repeated", "hey Jack",
        "hey Rex", "hey Max", "hey Lex", "hey Tex", "hey next", "hey decks", "hey checks",
        "hey text", "hey six", "Keswick", "excitement", "external", "excellent", "expect",
    ],
}

LONE_PHRASES = {"de": ["Hex", "Hex!", "Hex?", "Hex."], "en": ["Hex", "Hex!", "Hex?", "Hex."]}


@dataclass(frozen=True)
class PiperVoice:
    """A Piper voice whose training data is CC0, CC-BY, public domain or Apache-2.0. The number of
    speakers comes from the voice's own `.onnx.json` (the catalog's counts can be wrong)."""

    key: str
    path: str  # under rhasspy/piper-voices
    language: str
    licence: str


PIPER_REPO = "rhasspy/piper-voices"
# The MLS voices (de_DE-mls, fr_FR-mls) are left out on purpose: in a trial (2026-11-08) they
# babbled for seconds instead of saying two words. Voices with NC, SA or unclear licences are left
# out too.
PIPER_VOICES = [
    PiperVoice("de_DE-thorsten_emotional-medium", "de/de_DE/thorsten_emotional/medium", "de", "CC0"),
    PiperVoice("de_DE-thorsten-high", "de/de_DE/thorsten/high", "de", "CC0"),
    PiperVoice("en_US-libritts_r-medium", "en/en_US/libritts_r/medium", "en", "CC-BY-4.0 (LibriTTS-R)"),
    PiperVoice("en_GB-vctk-medium", "en/en_GB/vctk/medium", "en", "CC-BY-4.0 (VCTK)"),
    PiperVoice("en_GB-alba-medium", "en/en_GB/alba/medium", "en", "CC-BY-4.0"),
    PiperVoice("en_GB-cori-high", "en/en_GB/cori/high", "en", "public domain"),
    PiperVoice("en_US-bryce-medium", "en/en_US/bryce/medium", "en", "public domain"),
    PiperVoice("en_US-joe-medium", "en/en_US/joe/medium", "en", "CC0"),
    PiperVoice("en_US-john-medium", "en/en_US/john/medium", "en", "public domain"),
    PiperVoice("en_US-kristin-medium", "en/en_US/kristin/medium", "en", "public domain"),
    PiperVoice("en_US-ljspeech-high", "en/en_US/ljspeech/high", "en", "public domain"),
    PiperVoice("en_US-mike-medium", "en/en_US/mike/medium", "en", "CC0"),
    PiperVoice("en_US-norman-medium", "en/en_US/norman/medium", "en", "public domain"),
    PiperVoice("en_US-reza_ibrahim-medium", "en/en_US/reza_ibrahim/medium", "en", "CC0"),
    PiperVoice("en_US-sam-medium", "en/en_US/sam/medium", "en", "Apache-2.0"),
]

# Kyutai's voice donations: CC0, given for voice use (kyutai/tts-voices, voice-donations/).
POCKET_VOICES_REPO = "kyutai/tts-voices"
POCKET_VOICES_DIR = "voice-donations"

# Negatives: speech that is not the wake word, and noise. No login needed for any of them.
MLS_REPO = "facebook/multilingual_librispeech"  # CC-BY-4.0
MLS_CONFIGS = {"de": "german", "en": "english", "fr": "french", "es": "spanish", "it": "italian", "pt": "portuguese"}
FLEURS_REPO = "google/fleurs"  # CC-BY-4.0
FLEURS_CONFIGS = {"de": "de_de", "en": "en_us", "fr": "fr_fr", "es": "es_419", "it": "it_it", "pt": "pt_br"}
RIRS_NOISES_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"  # Apache-2.0


@dataclass(frozen=True)
class Paths:
    root: Path
    clips_name: str = "clips"

    @property
    def clips(self) -> Path:
        """The augmented, embedded clips of one build (`hw --clips <name> clips`)."""
        return self.features / self.clips_name

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def positives(self) -> Path:
        return self.raw / "positives"

    @property
    def confusables(self) -> Path:
        return self.raw / "confusables"

    @property
    def speech(self) -> Path:
        return self.raw / "speech"

    @property
    def noise(self) -> Path:
        return self.raw / "rirs_noises"

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

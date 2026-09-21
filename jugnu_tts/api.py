import re
import warnings
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

REPO = "altslate/jugnu-pocket-tts"
MODELS = {"lite": "lite-6l", "base": "base-12l", "teacher": "teacher-24l"}
# Kyutai's Mimi codec is not redistributed by us: it is fetched from their gated repo with the user's own access.
KYUTAI_REPO = "kyutai/pocket-tts"
KYUTAI_FILE = "languages/english_2026-04_24l/model.safetensors"
KYUTAI_REV = "492522650173a0653b7575cdc25ae09810e5d741"
CACHE = Path.home() / ".cache" / "jugnu_tts"


@dataclass
class Voice:
    """A cloned voice: the model state after listening to a reference clip. Reusable and saveable."""

    state: dict
    source: str = ""

    def save(self, path: str | Path) -> Path:
        from pocket_tts import export_model_state

        export_model_state(self.state, path)
        return Path(path)


def _assemble(folder: str) -> Path:
    """Merge our FlowLM weights with Kyutai's codec into one local bundle; returns the config to load."""
    out = CACHE / folder
    config = out / "config.local.yaml"
    if config.exists() and (out / "model.safetensors").exists():
        return config
    try:
        src = Path(snapshot_download(REPO, allow_patterns=[f"{folder}/*"])) / folder
        codec = hf_hub_download(KYUTAI_REPO, KYUTAI_FILE, revision=KYUTAI_REV)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the model. Both repos are gated: accept the terms at "
            f"https://huggingface.co/{REPO} and https://huggingface.co/{KYUTAI_REPO}, "
            "then run `huggingface-cli login`."
        ) from exc
    out.mkdir(parents=True, exist_ok=True)
    bundle = {}
    with safe_open(str(src / "flow_lm.safetensors"), "pt") as f:
        for k in f.keys():
            bundle["flow_lm." + k] = f.get_tensor(k)
    with safe_open(codec, "pt") as f:
        for k in f.keys():
            if k.startswith("mimi."):
                bundle[k] = f.get_tensor(k)
    save_file(bundle, str(out / "model.safetensors"))
    cfg = yaml.safe_load(open(src / "config.yaml"))
    cfg["weights_path"] = str(out / "model.safetensors")
    cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = str(src / "tokenizer.model")
    yaml.safe_dump(cfg, open(config, "w"), sort_keys=False)
    return config


class Jugnu:
    """Text-to-speech in Hindi, English and Hinglish, in a voice cloned from a short reference clip.

    model: "lite" (6 layers, fastest, runs on CPU), "base" (12 layers) or "teacher" (24 layers; needs
    guidance the standard runtime does not provide, so it sounds worse here than its students — use it for research).
    """

    def __init__(self, model: str = "lite", temperature: float | None = None):
        if model not in MODELS:
            raise ValueError(f"model must be one of {sorted(MODELS)}, got {model!r}")
        if model == "teacher":
            warnings.warn("The teacher needs classifier-free guidance 2.0, which this runtime does not apply; "
                          "prefer 'base' or 'lite'.", stacklevel=2)
        from pocket_tts import TTSModel

        self.name = model
        self._tts = TTSModel.load_model(config=str(_assemble(MODELS[model])), temp=temperature)

    @property
    def sample_rate(self) -> int:
        return self._tts.sample_rate

    def clone(self, reference: str | Path) -> Voice:
        """Clone a voice from an audio file (5–15 s of clean speech works best) or a saved Voice file.

        The output copies the reference's acoustics: a noisy or phone-quality clip gives noisy speech.
        Only clone voices you have explicit, lawful consent to use.
        """
        reference = Path(reference)
        if not reference.exists():
            raise FileNotFoundError(reference)
        return Voice(self._tts.get_state_for_audio_prompt(str(reference)), source=str(reference))

    load_voice = clone  # a Voice saved with Voice.save() loads the same way

    def speak(self, text: str, voice: Voice, out: str | Path | None = None) -> np.ndarray:
        """Synthesise `text` in `voice`. Returns mono float32 audio at `sample_rate`; also writes a wav if `out` is set."""
        audio = self._tts.generate_audio(voice.state, _check(text)).numpy()
        if out is not None:
            _write_wav(out, audio, self.sample_rate)
        return audio

    def stream(self, text: str, voice: Voice) -> Iterator[np.ndarray]:
        """Yield audio chunks as they are generated, for low-latency playback."""
        for chunk in self._tts.generate_audio_stream(voice.state, _check(text)):
            yield chunk.numpy()


def _write_wav(path: str | Path, audio: np.ndarray, sample_rate: int):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def _check(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        raise ValueError("text is empty")
    if re.search(r"\d", text):
        warnings.warn("Digits are not normalised: write numbers as words (e.g. 'तीन बजे', 'three thirty').", stacklevel=3)
    return text

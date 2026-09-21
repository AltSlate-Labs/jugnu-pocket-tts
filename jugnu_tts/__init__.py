"""Jugnu Pocket TTS — a small Python API for Hindi / English / Hinglish speech in a cloned voice.

    from jugnu_tts import Jugnu

    tts = Jugnu("lite")                          # downloads and assembles the model once, then caches it
    voice = tts.clone("my_consented_voice.wav")  # 5–15 s of clean speech
    tts.speak("आपका order confirm हो गया है", voice, out="hello.wav")

Only clone a voice with the speaker's explicit, lawful consent. See the model card for the use terms:
https://huggingface.co/altslate/jugnu-pocket-tts
"""

from jugnu_tts.api import Jugnu, Voice

__all__ = ["Jugnu", "Voice"]
__version__ = "0.1.0"

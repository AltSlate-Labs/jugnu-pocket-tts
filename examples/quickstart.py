"""Clone a voice once, reuse it, stream a long sentence.

    pip install git+https://github.com/AltSlate-Labs/jugnu-pocket-tts
    huggingface-cli login      # after accepting the terms of altslate/jugnu-pocket-tts and kyutai/pocket-tts
    python examples/quickstart.py my_consented_voice.wav
"""
import sys

import numpy as np

from jugnu_tts import Jugnu

tts = Jugnu("lite")
voice = tts.clone(sys.argv[1])
voice.save("my_voice.safetensors")          # next time: tts.load_voice("my_voice.safetensors")

tts.speak("नमस्ते, मेरा नाम जुगनू है और मैं आपकी मदद करने के लिए यहाँ हूँ", voice, out="hindi.wav")
tts.speak("आपका order confirm हो गया है और delivery कल शाम तक हो जाएगी", voice, out="hinglish.wav")
tts.speak("Hello, thank you for calling. How can I help you today?", voice, out="english.wav")

chunks = list(tts.stream("मैंने अपना laptop restart किया, फिर software update install किया", voice))
print(f"streamed {len(chunks)} chunks, {sum(map(len, chunks)) / tts.sample_rate:.1f} s of audio")

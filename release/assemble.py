"""Assemble a runnable Pocket TTS bundle from our FlowLM weights + Kyutai's Mimi codec.

We do not redistribute Kyutai's codec. This script downloads it from the gated
`kyutai/pocket-tts` repo with YOUR Hugging Face account (accept their terms first:
https://huggingface.co/kyutai/pocket-tts), merges it with `flow_lm.safetensors`
from this repo, and writes `model.safetensors` + `config.local.yaml` next to it.

    pip install pocket-tts huggingface_hub safetensors pyyaml
    huggingface-cli login
    python assemble.py            # run inside the downloaded repo folder
    pocket-tts generate --config config.local.yaml --voice my_voice.wav --text "नमस्ते"
"""
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
KYUTAI_REPO = "kyutai/pocket-tts"
KYUTAI_FILE = "languages/english_2026-04_24l/model.safetensors"
KYUTAI_REV = "492522650173a0653b7575cdc25ae09810e5d741"

bundle = {}
with safe_open(str(HERE / "flow_lm.safetensors"), "pt") as f:
    for k in f.keys():
        bundle["flow_lm." + k] = f.get_tensor(k)
codec = hf_hub_download(KYUTAI_REPO, KYUTAI_FILE, revision=KYUTAI_REV)
with safe_open(codec, "pt") as f:
    for k in f.keys():
        if k.startswith("mimi."):
            bundle[k] = f.get_tensor(k)
save_file(bundle, str(HERE / "model.safetensors"))

cfg = yaml.safe_load(open(HERE / "config.yaml"))
cfg["weights_path"] = str(HERE / "model.safetensors")
cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = str(HERE / "tokenizer.model")
yaml.safe_dump(cfg, open(HERE / "config.local.yaml", "w"), sort_keys=False)
print("wrote", HERE / "model.safetensors", "and", HERE / "config.local.yaml")

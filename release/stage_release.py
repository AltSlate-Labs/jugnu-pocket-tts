"""Stage the HF release of the 115k scratch teacher: our FlowLM weights only; Mimi is fetched from Kyutai's gated repo."""
import shutil, sys
from pathlib import Path
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

T = Path.home() / "hindi-tts"
out = T / "hf_release" / "teacher_24l"
out.mkdir(parents=True, exist_ok=True)
src = T / "final_v0" / "scratch" / "model.safetensors"   # EMA export written at step 115,192
f = safe_open(str(src), "pt")
flow = {k: f.get_tensor(k) for k in f.keys() if k.startswith("flow_lm.")}
save_file(flow, str(out / "flow_lm.safetensors"))
print("flow_lm tensors", len(flow), "params %.1fM" % (sum(v.numel() for v in flow.values()) / 1e6))
shutil.copy(T / "data/v0/tokenizer.model", out / "tokenizer.model")
cfg = yaml.safe_load(open(T / "configs/hinglish_24l.yaml"))
kyutai = cfg.pop("weights_path")
cfg.pop("weights_path_without_voice_cloning", None)
cfg["flow_lm"]["weights_path"] = sys.argv[1] + "/flow_lm.safetensors"
cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = sys.argv[1] + "/tokenizer.model"
cfg["mimi"]["weights_path"] = kyutai
yaml.safe_dump(cfg, open(out / "config.yaml", "w"), sort_keys=False)
print(open(out / "config.yaml").read()[:700])

"""Archive the latest student checkpoints and stage FlowLM-only release folders for all three models."""
import glob, shutil, sys
from pathlib import Path
import torch, yaml
from safetensors.torch import save_file

T = Path.home() / "hindi-tts"
REL = T / "hf_release"
base_cfg = yaml.safe_load(open(T / "configs/hinglish_24l.yaml"))
base_cfg.pop("weights_path", None); base_cfg.pop("weights_path_without_voice_cloning", None)
base_cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = "tokenizer.model"

def export(ckpt: Path, folder: str, layers: int):
    out = REL / folder; out.mkdir(parents=True, exist_ok=True)
    p = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = dict(p["model"]); state.update(p.get("ema") or {})
    flow = {k.removeprefix("flow_lm."): v.contiguous() for k, v in state.items() if k.startswith("flow_lm.")}
    save_file(flow, str(out / "flow_lm.safetensors"))
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg)); cfg["flow_lm"]["transformer"]["num_layers"] = layers
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"), sort_keys=False)
    shutil.copy(T / "data/v0/tokenizer.model", out / "tokenizer.model")
    shutil.copy(REL / "teacher_24l/assemble.py", out / "assemble.py")
    (out / ".gitignore").write_text("model.safetensors\nconfig.local.yaml\n")
    print(folder, "step", p["step"], "tensors", len(flow), "params %.1fM" % (sum(v.numel() for v in flow.values()) / 1e6))

for name, folder, layers in (("lite6", "lite-6l", 6), ("base12", "base-12l", 12)):
    ck = sorted(glob.glob(str(T / f"runs/{name}_v0/checkpoint_*.pt")))[-1]
    keep = T / "final_v0" / f"{name}_{Path(ck).stem.split('_')[1]}.pt"
    shutil.copy(ck, keep); export(keep, folder, layers); print("archived", keep)
if not (REL / "teacher-24l").exists():
    (REL / "teacher_24l").rename(REL / "teacher-24l")

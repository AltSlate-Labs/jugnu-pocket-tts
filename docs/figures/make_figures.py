"""README figures. Numbers are copied from notes/experiments.md (held-out benchmark, IndicConformer WER)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False})

def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)

# 1. WER over training steps -------------------------------------------------------------------------------------
steps_a = [25, 50, 75, 100, 115, 125, 150, 175, 200, 225, 250, 275, 300, 350, 375, 400]
scratch = {"Hindi": [9.7, 10.0, 10.3, 9.0, 9.1, 9.4, 10.1, 10.8, 9.8, 10.6, 10.9, 12.0, 12.2, 14.6, 15.4, 14.3],
           "Hinglish": [14.0, 13.8, 13.0, 12.9, 13.3, 13.5, 12.4, 13.3, 13.7, 14.5, 13.7, 14.4, 15.3, 19.6, 20.1, 18.9]}
steps_b = [25, 50, 75, 100, 115]
ftlang = {"Hindi": [9.1, 8.9, 9.3, 8.8, 9.1], "Hinglish": [12.7, 13.4, 12.9, 12.1, 12.2]}
real = {"Hindi": 14.5, "Hinglish": 15.7}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, lang in zip(axes, ("Hindi", "Hinglish")):
    style(ax)
    ax.axhline(real[lang], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(3, real[lang] + 0.35, f"same judge on the real recordings: {real[lang]}%", color=MUTED, fontsize=8.5)
    ax.plot(steps_a, scratch[lang], color=BLUE, linewidth=2, marker="o", markersize=4, label="Teacher from scratch")
    ax.plot(steps_b, ftlang[lang], color=ORANGE, linewidth=2, marker="o", markersize=4, label="Teacher from Kyutai's English model")
    ax.axvline(115, color=GRID, linewidth=1)
    ax.text(118, 6.4, "released\ncheckpoint\n(115k)", color=MUTED, fontsize=8.5, va="bottom")
    ax.set_title(lang, loc="left", color=INK, fontsize=11, fontweight="bold"); ax.set_xlabel("training steps (thousands)")
    ax.set_ylim(6, 22); ax.set_xlim(0, 410)
axes[0].set_ylabel("word error rate, % (lower is better)")
axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.93), fontsize=9, labelcolor=INK)
fig.suptitle("Longer training made the teacher worse on this noisy corpus", x=0.01, ha="left", color=INK, fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig("docs/figures/wer_vs_steps.png", dpi=150); plt.close(fig)

# 2. Model family ---------------------------------------------------------------------------------------------------
models = ["Teacher\n24 layers · 316M", "Base\n12 layers · 165M", "Lite\n6 layers · 89M"]
wer = {"Hindi": [9.1, 8.5, 9.0], "Hinglish": [13.3, 11.7, 12.2]}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, lang in zip(axes, ("Hindi", "Hinglish")):
    style(ax)
    bars = ax.bar(models, wer[lang], width=0.5, color=BLUE)
    for b, v in zip(bars, wer[lang]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v}%", ha="center", color=INK, fontsize=10)
    ax.axhline(real[lang], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(2.45, real[lang] + 0.3, f"real recordings: {real[lang]}%", color=MUTED, fontsize=8.5, ha="right")
    ax.set_title(lang, loc="left", color=INK, fontsize=11, fontweight="bold"); ax.set_ylim(0, 18)
axes[0].set_ylabel("word error rate, %")
fig.suptitle("The 6-layer student matches its 24-layer teacher", x=0.01, ha="left", color=INK, fontsize=12.5, fontweight="bold")
fig.text(0.01, 0.005, "Held-out speakers, 150 sentences per set. Released checkpoints: Base at 90k and Lite at 112.5k distillation steps.",
         color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.04, 1, 0.93)); fig.savefig("docs/figures/model_family.png", dpi=150); plt.close(fig)

# 3. Reference-voice quality ------------------------------------------------------------------------------------------
voices = ["Noisy field recording\n(male, phone mic)", "Noisy field recording\n(female, phone mic)", "Clean studio reader\n(LibriVox)"]
utmos = [1.89, 2.44, 3.65]
fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
bars = ax.barh(voices, utmos, height=0.5, color=BLUE)
for b, v in zip(bars, utmos): ax.text(v + 0.05, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", color=INK)
ax.set_xlim(1, 4.5); ax.set_xlabel("UTMOS of generated Hindi speech (1–5, higher is better)"); ax.spines["left"].set_visible(False)
ax.set_title("Same model, same text: the reference clip sets the sound quality", loc="left", color=INK, fontsize=11.5, fontweight="bold")
fig.tight_layout(); fig.savefig("docs/figures/reference_quality.png", dpi=150); plt.close(fig)
print("ok")

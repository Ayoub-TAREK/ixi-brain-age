"""README figures (docs/fig_preprocessing.png, docs/fig_dann.png); values from docs/journal.md. Run from the repository root."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2 = "#2a78d6", "#eb6834"  # categorical slots 1-2 of the reference palette
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def clean(ax):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# 1. out-of-site MAE per held-out hospital: own preprocessing vs Clinica t1-linear
sites = ["IOP", "HH", "Guys"]
v1, v2 = [10.8, 25.2, 17.1], [11.2, 8.6, 8.9]
fig, ax = plt.subplots(figsize=(8, 4.6))
w, gap = 0.36, 0.02
for i, (a, b) in enumerate(zip(v1, v2)):
    for x, val, col in [(i - w / 2 - gap, a, S1), (i + w / 2 + gap, b, S2)]:
        ax.bar(x, val, width=w, color=col, edgecolor=SURFACE, linewidth=2)
        ax.text(x, val + 0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=10.5, color=INK)
ax.axhspan(14, 16, color="#d9d8d3", alpha=0.45, lw=0, zorder=0)
ax.text(2.62, 15, "always predicting\nthe mean age", va="center", ha="right", fontsize=9, color=INK2)
ax.set_xticks(range(3), [f"{s}\nheld out" for s in sites])
ax.set_ylabel("Out-of-site MAE (years)")
ax.set_ylim(0, 34); ax.set_yticks(range(0, 30, 5))
ax.bar(0, 0, color=S1, label="v1: own preprocessing (2 mm, no registration)")
ax.bar(0, 0, color=S2, label="v2: Clinica t1-linear (bias correction + registration)")
ax.legend(frameon=False, loc="upper right", fontsize=9.5, labelcolor=INK)
ax.set_title("Error on the unseen hospital, before and after Clinica t1-linear", loc="left", fontsize=12.5, pad=12)
clean(ax)
fig.tight_layout()
fig.savefig("docs/fig_preprocessing.png", dpi=150)
plt.close(fig)

# 2. DANN: out-of-site MAE vs alignment strength, IOP (younger population) vs HH (similar population)
fig, ax = plt.subplots(figsize=(8, 4.6))
series = [("IOP (7 years younger than its training hospitals)", S1, [0, 0.1, 0.3], [8.87, 11.57, 13.03], 10.48),
          ("HH (2 years younger)", S2, [0, 0.3], [8.12, 8.01], 8.51)]
for name, col, lam, mae, iw in series:
    ax.plot(lam, mae, color=col, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=f"{name}: DANN")
    ax.plot(0.3, iw, linestyle="none", marker="o", ms=8, mfc=SURFACE, mec=col, mew=2,
            label=f"{name.split(' ')[0]}: IW-DANN (re-weighted by age)")
    ax.text(0.312, mae[-1], f"{mae[-1]:.1f}", va="center", fontsize=10, color=INK)
    ax.text(0.312, iw, f"{iw:.1f}", va="center", fontsize=10, color=INK2)
    ax.text(-0.012, mae[0], f"{mae[0]:.1f}", va="center", ha="right", fontsize=10, color=INK)
ax.set_xticks([0, 0.1, 0.3], ["0 (control)", "0.1", "0.3"])
ax.set_xlim(-0.045, 0.34)
ax.set_ylim(6, 14.5)
ax.set_xlabel("DANN alignment strength λ")
ax.set_ylabel("Out-of-site MAE (years)")
ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
ax.set_title("Aligning the hospitals hurts when their populations differ (seed 42)", loc="left", fontsize=12.5, pad=12)
clean(ax)
fig.tight_layout()
fig.savefig("docs/fig_dann.png", dpi=150)
plt.close(fig)
print("ok")

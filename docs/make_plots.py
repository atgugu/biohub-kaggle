"""Render docs/assets/public_vs_private_{light,dark}.png from data/submissions.csv."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "submissions.csv")
df = df[df["name"] != "biohub-d2-s10-rawilp"]  # raw-ILP experiment (0.897 / 0.880) would crush the y-axis

THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", muted="#52514e", grid="#f0efec", pub="#2a78d6", priv="#eb6834"),
    "dark": dict(bg="#1a1a19", ink="#ffffff", muted="#c3c2b7", grid="#383835", pub="#3987e5", priv="#d95926"),
}
FINAL = "X3e-tv"
MILESTONES = {  # name -> label
    "biohub-d1-s2-valoff": "public-notebook\nbaseline",
    "biohub-d3-r4-geofusion-veto": "fork veto",
    "X1h": "x138 base + our\ncoordinate head",
    "X3d-t50": "division attach v3",
    "X3e-tv": "TripletNet-aware veto\n(final pick)",
}

for mode, c in THEMES.items():
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=160)
    fig.patch.set_facecolor(c["bg"]); ax.set_facecolor(c["bg"])
    ax.plot(df.n, df.public, color=c["pub"], lw=2, marker="o", ms=4.5, mec=c["bg"], mew=1.2, label="Public LB")
    ax.plot(df.n, df.private, color=c["priv"], lw=2, marker="o", ms=4.5, mec=c["bg"], mew=1.2, label="Private LB")
    for name, label in MILESTONES.items():
        r = df[df["name"] == name].iloc[0]
        ax.annotate(label, (r.n, r.private), xytext=(0, {"X1h": -44, "X3e-tv": -40}.get(name, -34)), textcoords="offset points", ha="center",
                    fontsize=8, color=c["muted"], arrowprops=dict(arrowstyle="-", color=c["muted"], lw=0.7), bbox=dict(fc=c["bg"], ec="none", pad=1.5))
    last = df[df["name"] == FINAL].iloc[0]
    ax.annotate(f"{last.public:.5f}", (last.n, last.public), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color=c["ink"], fontweight="bold")
    ax.annotate(f"{last.private:.5f}", (last.n, last.private), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color=c["ink"], fontweight="bold")
    ax.set_ylim(0.895, 0.975); ax.set_xlim(0.5, df.n.max() + 0.5)
    ax.set_xlabel("submission # (chronological, 21 – 29 Sep 2026)", color=c["muted"], fontsize=9)
    ax.set_ylabel("score", color=c["muted"], fontsize=9)
    ax.grid(axis="y", color=c["grid"], lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"]); ax.tick_params(colors=c["muted"], labelsize=8, length=0)
    ax.set_title("Every submission, public vs. private score", loc="left", color=c["ink"], fontsize=12, fontweight="bold")
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for t in leg.get_texts(): t.set_color(c["ink"])
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "assets" / f"public_vs_private_{mode}.png", facecolor=c["bg"]); plt.close(fig)

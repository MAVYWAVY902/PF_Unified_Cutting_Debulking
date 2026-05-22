"""
Matplotlib preview of fig_architecture_comparison.tex
For visual reference — actual paper figure uses the TikZ .tex source.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from matplotlib.lines import Line2D
import numpy as np

# Colors (matching the TikZ palette)
REAL = "#C83C3C"
SIM = "#326EB4"
LOSS = "#7D50A0"
GRAD = "#328250"
ADAM = "#5A5A5A"
HIST = "#D2AA32"
OUTPUT = "#282828"
BGLEFT = "#F8F2F0"
BGRIGHT = "#F0F5FA"


def box(ax, x, y, w, h, text, color, fill_alpha=0.10, fontsize=8.5, font_weight="normal"):
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.6, edgecolor=color, facecolor=color, alpha=1.0,
    )
    rect.set_facecolor(_blend(color, "#FFFFFF", fill_alpha))
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=_darken(color, 0.4), weight=font_weight)


def ellipse(ax, x, y, w, h, text, color, fill_alpha=0.12, fontsize=8.5):
    e = Ellipse((x, y), w, h, linewidth=1.6, edgecolor=color,
                facecolor=_blend(color, "#FFFFFF", fill_alpha))
    ax.add_patch(e)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=_darken(color, 0.4))


def arrow(ax, x1, y1, x2, y2, color="#444444", style="solid", lw=1.4, label=None,
          label_offset=(0, 0), label_size=7):
    arrow_style = "-|>"
    linestyle = "-" if style == "solid" else "--" if style == "dashed" else ":"
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=arrow_style, mutation_scale=12,
        color=color, linewidth=lw, linestyle=linestyle,
    )
    ax.add_patch(arr)
    if label is not None:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=label_size, color=color, style="italic")


def _blend(c1, c2, alpha):
    """Blend hex colors. alpha=0 → c1, alpha=1 → c2."""
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    # Lighten by mixing with white
    r = int(r1 * (1 - alpha) + r2 * alpha + (255 * (1 - alpha)) * 0.85)
    g = int(g1 * (1 - alpha) + g2 * alpha + (255 * (1 - alpha)) * 0.85)
    b = int(b1 * (1 - alpha) + b2 * alpha + (255 * (1 - alpha)) * 0.85)
    r, g, b = min(255, r), min(255, g), min(255, b)
    return f"#{r:02X}{g:02X}{b:02X}"


def _darken(c, alpha):
    r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    r = int(r * (1 - alpha))
    g = int(g * (1 - alpha))
    b = int(b * (1 - alpha))
    return f"#{r:02X}{g:02X}{b:02X}"


fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 24)
ax.set_ylim(0, 14)
ax.set_aspect("equal")
ax.axis("off")

# =============================================================================
# (a) BEFORE — left panel
# =============================================================================

# Panel background
ax.add_patch(FancyBboxPatch((0.3, 0.5), 10.8, 13.0,
                            boxstyle="round,pad=0.05,rounding_size=0.2",
                            facecolor=BGLEFT, edgecolor=REAL, alpha=0.5,
                            linewidth=1.2))

# Title
ax.text(5.7, 13.0, "(a) Before: Offline Batch System Identification",
        ha="center", va="center", fontsize=11, weight="bold")

# Real + Sim
box(ax, 2.8, 11.5, 2.6, 0.9, "Real phantom\nexperiment", REAL)
ax.text(2.8, 10.85, "fixed trajectory", ha="center", va="center",
        fontsize=7, color=_darken(REAL, 0.4), style="italic")

box(ax, 8.6, 11.5, 2.6, 0.9, "Simulator $\\mathcal{S}_\\theta$\n$P(F)$ cohesive zone", SIM)
ax.text(8.6, 10.85, "with current $\\theta$", ha="center", va="center",
        fontsize=7, color=_darken(SIM, 0.4), style="italic")

# Cavities
ax.text(2.8, 9.7, "$\\alpha^{\\mathrm{real}}(T)$", ha="center", va="center",
        fontsize=10, color=_darken(REAL, 0.4))
ax.text(2.8, 9.3, "final cavity", ha="center", va="center",
        fontsize=7, color=_darken(REAL, 0.4), style="italic")

ax.text(8.6, 9.7, "$\\alpha^{\\mathrm{sim}}(T;\\theta)$", ha="center", va="center",
        fontsize=10, color=_darken(SIM, 0.4))
ax.text(8.6, 9.3, "final cavity", ha="center", va="center",
        fontsize=7, color=_darken(SIM, 0.4), style="italic")

# Loss
ellipse(ax, 5.7, 7.7, 5.4, 1.0,
        "$\\mathcal{L}_{\\mathrm{endpoint}} = \\|\\alpha^{\\mathrm{sim}} - \\alpha^{\\mathrm{real}}\\|^2$",
        LOSS, fontsize=9)

# Gradient
box(ax, 5.7, 6.2, 4.8, 0.9, "Warp tape backward\n$\\partial \\mathcal{L} / \\partial \\theta$", GRAD)

# Adam
box(ax, 5.7, 4.8, 4.8, 0.8, "Adam step  (50 iterations)", ADAM)

# Output
box(ax, 5.7, 3.4, 4.8, 0.8, "$\\theta^*$  (frozen)", OUTPUT)

# Arrows
arrow(ax, 2.8, 10.95, 2.8, 10.05, REAL)
arrow(ax, 8.6, 10.95, 8.6, 10.05, SIM)
arrow(ax, 2.8, 9.0, 4.8, 8.0, "#444444")
arrow(ax, 8.6, 9.0, 6.6, 8.0, "#444444")
arrow(ax, 5.7, 7.15, 5.7, 6.65, GRAD, lw=1.8)
arrow(ax, 5.7, 5.75, 5.7, 5.20, "#444444")
arrow(ax, 5.7, 4.40, 5.7, 3.80, "#444444")

# Adam → sim feedback loop (dashed)
arrow(ax, 7.9, 4.8, 9.7, 4.8, "#666666", style="dashed", lw=1.2,
      label="$\\theta\\!\\leftarrow\\!\\theta\\!+\\!\\Delta\\theta$", label_offset=(0, 0.3))
ax.plot([9.7, 9.7], [4.8, 11.5], "--", color="#666666", lw=1.0)
arrow(ax, 9.7, 11.5, 9.9, 11.5, "#666666", style="dashed", lw=1.2)

# Annotations
ax.text(5.7, 2.4, "Differentiability used: 50 backward calls",
        ha="center", va="center", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888888", lw=0.5),
        color="#444")
ax.text(5.7, 1.7, "Per calibration: ONCE   •   Output: 1 fixed $\\theta$",
        ha="center", va="center", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888888", lw=0.5),
        color="#444")

# =============================================================================
# (b) AFTER — right panel
# =============================================================================

ax.add_patch(FancyBboxPatch((11.8, 0.5), 11.9, 13.0,
                            boxstyle="round,pad=0.05,rounding_size=0.2",
                            facecolor=BGRIGHT, edgecolor=SIM, alpha=0.5,
                            linewidth=1.2))

# Title
ax.text(17.75, 13.05, "(b) After: Online Real-to-Sim Residual Calibration",
        ha="center", va="center", fontsize=11, weight="bold")
ax.text(17.75, 12.55,
        "per-frame streaming loop   $t = 0, 1, 2, \\ldots, T$",
        ha="center", va="center", fontsize=8, style="italic", color="#666")

# Shared control u_t
ax.text(17.75, 11.85, "$\\mathbf{u}_t$  (shared)", ha="center", va="center",
        fontsize=10, weight="bold", color="#444")

# Real + Sim per-frame
box(ax, 14.7, 10.85, 2.6, 0.85, "Real phantom\nframe $t$", REAL)
ax.text(14.7, 10.20, "obs: $(\\alpha^{\\mathrm{real}}_t, F^{\\mathrm{real}}_t)$",
        ha="center", va="center", fontsize=7,
        color=_darken(REAL, 0.4), style="italic")

box(ax, 20.8, 10.85, 2.6, 0.85, "Simulator $\\mathcal{S}_{\\theta_t}$\nframe $t$", SIM)
ax.text(20.8, 10.20, "pred: $(\\alpha^{\\mathrm{sim}}_t, \\delta^v_t, F^{\\mathrm{sim}}_t)$",
        ha="center", va="center", fontsize=7,
        color=_darken(SIM, 0.4), style="italic")

# u_t arrows
arrow(ax, 17.0, 11.7, 14.7, 11.30, "#666")
arrow(ax, 18.5, 11.7, 20.8, 11.30, "#666")

# Inner loop block
box(ax, 17.0, 8.4, 6.2, 1.5,
    "Inner Loop  (Mod 1: irreversible residual)\n"
    "$\\Delta\\delta_v = \\arg\\min_{\\Delta \\geq 0}\\;\\|\\sigma(\\delta_v\\!+\\!\\Delta\\!-\\!\\tau) - \\alpha^{\\mathrm{real}}_t\\|^2$\n"
    "30-step analytic Adam (softplus enforces $\\geq 0$)",
    GRAD, fontsize=7.5)

# Outer gradient block
box(ax, 17.0, 6.4, 6.2, 1.8,
    "Outer Gradient  (Warp tape ×5)\n"
    "$\\nabla_\\theta \\mathcal{L}_{\\mathrm{direct}}(t)$  +  "
    "$\\lambda_{\\mathrm{traj}} \\!\\sum_h \\nabla \\mathcal{L}(h)$  +  "
    "$\\lambda_p \\nabla \\mathcal{L}_{\\mathrm{prior}}$\n"
    "Mod 2 (multi-modal)  +  Mod 3 (trajectory consistency)",
    GRAD, fill_alpha=0.20, fontsize=7.5)

# AdamND
box(ax, 17.0, 4.5, 6.2, 0.7,
    "AdamND step  (Mod 4: 3D global $\\theta$, per-param lr)",
    ADAM, fontsize=8)

# Output
box(ax, 17.0, 3.4, 6.2, 0.7,
    "$\\theta_{t+1}$  (continuously adapting)",
    OUTPUT, fontsize=8)

# History buffer
box(ax, 22.1, 6.4, 1.7, 1.8,
    "History\nbuffer $H_t$\n$\\{(u_h, \\alpha^{\\mathrm{real}}_h)\\}$",
    HIST, fontsize=7.5)

# Data flow
arrow(ax, 14.7, 10.40, 15.0, 9.20, REAL)
arrow(ax, 20.8, 10.40, 19.0, 9.20, SIM)
arrow(ax, 17.0, 7.60, 17.0, 7.30, GRAD, lw=1.6)
arrow(ax, 17.0, 5.45, 17.0, 4.85, GRAD, lw=1.6)
arrow(ax, 17.0, 4.10, 17.0, 3.75, "#444")

# Feedback theta_t to sim
arrow(ax, 20.1, 3.4, 22.5, 3.4, "#666", style="dashed", lw=1.0,
      label="$t \\!\\leftarrow\\! t\\!+\\!1$", label_offset=(0, 0.3))
ax.plot([22.5, 22.5], [3.4, 10.85], "--", color="#666", lw=0.9)
arrow(ax, 22.5, 10.85, 22.05, 10.85, "#666", style="dashed", lw=1.0)

# L_traj sample from history
arrow(ax, 21.3, 6.4, 20.1, 6.4, HIST, style="dashed", lw=1.1,
      label="sample 4", label_offset=(0, 0.3), label_size=6.5)

# Store to history
ax.plot([15.6, 15.6, 22.1], [10.85, 7.6, 7.6], "--", color=HIST, lw=1.0, alpha=0.7)
arrow(ax, 22.1, 7.6, 22.1, 7.30, HIST, style="dashed", lw=1.1)
ax.text(18.5, 7.85, "store", ha="center", fontsize=6.5, color=_darken(HIST, 0.3),
        style="italic")

# Annotations
ax.text(17.0, 2.4, "Differentiability used: 35 backward calls per frame",
        ha="center", va="center", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888888", lw=0.5),
        color="#444")
ax.text(17.0, 1.7,
        "Streaming: every frame   •   Output: $\\theta(t)$ trajectory + state corrections",
        ha="center", va="center", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888888", lw=0.5),
        color="#444")

# =============================================================================
# Save
# =============================================================================
plt.tight_layout()
out_path = "paper/figs/fig_architecture_preview.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"saved: {out_path}")

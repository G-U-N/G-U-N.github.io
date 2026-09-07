"""Rebuild the original figures in rl-opd-vsd.html (NumPy + Matplotlib).

Run from any directory: python3 /path/to/plot_rl_opd_vsd.py
All curves are analytic Gaussian densities/convolutions; no training data.
"""
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rl-opd-vsd-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures"
INK, MUTED = "#1e1b4b", "#64748b"
VIOLET, TEAL, GRID = "#6366f1", "#0f766e", "#e5e9f1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False,
    "axes.titleweight": "bold", "axes.titlecolor": INK,
    "mathtext.fontset": "stix", "pdf.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def gaussian(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def base_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.tick_params(length=0, pad=8)
    ax.set_xlabel("Output coordinate  x", labelpad=10, fontsize=11)


def export(fig, stem, mobile=False):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=200)
    if not mobile:
        fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)
    print(stem)


def two_gradients(mobile=False):
    fig = plt.figure(figsize=(6, 9.5) if mobile else (12, 6))
    fig.text(.10 if mobile else .065, .98 if mobile else .925,
             "One mismatch.\nTwo kinds of feedback." if mobile else "One mismatch. Two kinds of feedback.", fontsize=20 if mobile else 21,
             fontfamily="DejaVu Serif", weight="bold", va="top" if mobile else "baseline")
    fig.text(.10 if mobile else .065, .875 if mobile else .865,
             r"Student $q=\mathcal{N}(-1,1)$     ·     Target $p=\mathcal{N}(1,1)$",
             color=MUTED, fontsize=13)
    gs = (fig.add_gridspec(2, 1, left=.10, right=.96, top=.76, bottom=.10, hspace=.65)
          if mobile else fig.add_gridspec(1, 2, left=.065, right=.97, top=.72, bottom=.24, wspace=.25))
    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(-3.7, 3.7, 500)
    base_axes(ax)
    ax.axhline(0, color=GRID, lw=1)
    ax.axvline(0, color=GRID, lw=1)
    ax.plot(x, 2*x, color=VIOLET, lw=2.8)
    ax.fill_between(x, 2*x, 0, where=x <= 0, color=VIOLET, alpha=.07)
    ax.fill_between(x, 2*x, 0, where=x >= 0, color=TEAL, alpha=.09)
    samples = np.array([-2.6, -1.8, -1.0, -.2, .6])
    ax.scatter(samples, 2*samples, s=42, color=VIOLET, edgecolor="white", linewidth=.9, zorder=3)
    ax.set(xlim=(-3.7, 3.7), ylim=(-8, 8), xticks=[-3, -1, 1, 3], yticks=[-6, 0, 6])
    ax.set_title("A   Weight sampled log probabilities", loc="left", fontsize=12, pad=20)
    ax.text(.05, .86, r"$r_\theta(x)=\log p(x)-\log q(x)=2x$", transform=ax.transAxes, fontsize=15)
    ax.text(.98, .03, "Higher x → higher reward", transform=ax.transAxes,
            ha="right", color=TEAL, fontsize=11)

    ax = fig.add_subplot(gs[1, 0] if mobile else gs[0, 1])
    base_axes(ax)
    q, p = gaussian(x, -1, 1), gaussian(x, 1, 1)
    ax.fill_between(x, q, color=VIOLET, alpha=.08)
    ax.fill_between(x, p, color=TEAL, alpha=.08)
    ax.plot(x, p, color=TEAL, lw=2.6, label="Target p")
    ax.plot(x, q, color=VIOLET, lw=2.6, linestyle=(0, (5, 3)), label="Student q")
    for sample in samples:
        ax.scatter(sample, -.055, color=VIOLET, s=27, zorder=3)
        ax.annotate("", xy=(sample+.62, -.055), xytext=(sample, -.055),
                    arrowprops={"arrowstyle": "-|>", "color": TEAL, "lw": 1.8})
    ax.set(xlim=(-3.7, 3.7), ylim=(-.09, .54), xticks=[-3, -1, 1, 3], yticks=[0, .2, .4])
    ax.set_title("B   Move differentiable outputs", loc="left", fontsize=12, pad=20)
    ax.text(.5, .9, r"$\partial_x r_\theta=s_p-s_q=2$", transform=ax.transAxes,
            fontsize=16, ha="center")
    if not mobile:
        fig.text(.065, .09, "SCALAR FEEDBACK", color=VIOLET, fontsize=10, weight="bold")
        fig.text(.065, .045, "Reward × parameter log-probability gradient", color=MUTED, fontsize=10.5)
        fig.text(.57, .09, "DIRECTIONAL FEEDBACK", color=TEAL, fontsize=10, weight="bold")
        fig.text(.57, .045, "Score difference × generator Jacobian", color=MUTED, fontsize=10.5)
    fig.legend(handles=[Line2D([0], [0], color=TEAL, lw=2.5, label="Target p"),
                        Line2D([0], [0], color=VIOLET, lw=2.5, ls="--", label="Student q")],
               loc="upper right", bbox_to_anchor=(.96, .85) if mobile else (.965, .89),
               frameon=False, fontsize=11 if mobile else 10, ncol=2 if mobile else 1)
    export(fig, "rl-opd-vsd-two-gradients" + ("-mobile" if mobile else ""), mobile)


def noisy_marginals(mobile=False):
    fig = plt.figure(figsize=(6, 11.5) if mobile else (12, 6))
    fig.text(.10 if mobile else .065, .98 if mobile else .925,
             "The same outputs,\nviewed through more noise" if mobile else "The same outputs, viewed through more noise", fontsize=19 if mobile else 20,
             fontfamily="DejaVu Serif", weight="bold", va="top" if mobile else "baseline")
    fig.text(.10 if mobile else .065, .908 if mobile else .86,
             r"$x_t=x_0+\sigma_t\epsilon$" if mobile else r"$x_t=x_0+\sigma_t\epsilon$     ·     Each panel is a marginal, not a sampler step",
             color=MUTED, fontsize=12)
    gs = (fig.add_gridspec(3, 1, left=.10, right=.96, top=.79, bottom=.10, hspace=.67)
          if mobile else fig.add_gridspec(1, 3, left=.065, right=.97, top=.70, bottom=.24, wspace=.23))
    x = np.linspace(-5, 5, 1600)
    clean_std = .26
    levels = [.12, .65, 1.4]
    for i, noise in enumerate(levels):
        ax = fig.add_subplot(gs[i, 0] if mobile else gs[0, i])
        std = np.sqrt(clean_std**2 + noise**2)
        # Same noising kernel for both mixtures; means and proportions stay fixed.
        target = .5*gaussian(x, -1.45, std) + .5*gaussian(x, 1.45, std)
        student = .22*gaussian(x, -1.1, std) + .78*gaussian(x, 1.8, std)
        base_axes(ax)
        ax.plot(x, target, color=TEAL, lw=2.4)
        ax.fill_between(x, target, color=TEAL, alpha=.075)
        ax.plot(x, student, color=VIOLET, lw=2.4, linestyle=(0, (5, 3)))
        ax.fill_between(x, student, color=VIOLET, alpha=.075)
        ax.set(xlim=(-5, 5), ylim=(0, 1.15), yticks=[0, .5, 1], xticks=[-4, 0, 4])
        ax.set_title(["A   Low noise", "B   Medium noise", "C   High noise"][i],
                     loc="left", fontsize=12, pad=28)
        ax.text(0, 1.025, rf"$\sigma_t={noise:.2f}$", transform=ax.transAxes, color=MUTED, fontsize=12)
        if mobile and i < 2:
            ax.set_xlabel("")
    fig.legend(handles=[Line2D([0], [0], color=TEAL, lw=2.5, label="Target marginal"),
                        Line2D([0], [0], color=VIOLET, lw=2.5, ls="--", label="Student marginal")],
               loc="lower left", bbox_to_anchor=(.08, .85) if mobile else (.055, .08),
               frameon=False, ncol=2, fontsize=11)
    fig.text(.10 if mobile else .065, .025 if mobile else .045,
             "Exact Gaussian convolutions · shared axis scale\nNo optimization between panels" if mobile else "Exact Gaussian convolutions · shared axis scale · no optimization between panels",
             color=MUTED, fontsize=11 if mobile else 10.5)
    export(fig, "rl-opd-vsd-noisy-marginals" + ("-mobile" if mobile else ""), mobile)


if __name__ == "__main__":
    two_gradients()
    noisy_marginals()
    two_gradients(mobile=True)
    noisy_marginals(mobile=True)

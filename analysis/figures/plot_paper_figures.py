#!/usr/bin/env python3
"""
Publication Figure Generation Script for Algebraic Transformer Paper.

Generates three publication-quality figures in both vector PDF and raster PNG formats (300+ DPI):
1. Figure 1: Scaling Laws (NLL vs Compute on log scale, FLOPs and parameter equivalence)
2. Figure 2: Architecture Diagram (Side-by-side computation graphs, component replacements)
3. Figure 3: Needle Recall Heatmap (Context budget vs depth retrieval accuracy across tasks)

Styling: Academic publication quality, readable typography, colorblind-accessible palettes,
zero AI clichés in labels.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

# Global Academic Styling Constants
COLORS = {
    "standard": "#2b5c8f",      # Deep academic navy
    "standard_light": "#e8eef5",
    "algebraic": "#d95f02",     # High-contrast vermilion/terracotta
    "algebraic_light": "#fef0e7",
    "synergy": "#7570b3",       # Rich purple for super-additive interaction
    "highlight": "#1b9e77",     # Deep teal for peak combination
    "neutral_dark": "#1a202c",  # Dark charcoal ink
    "neutral_mid": "#4a5568",   # Slate grey
    "neutral_light": "#edf2f7", # Light background tint
    "grid": "#e2e8f0",          # Subtle grid lines
    "border": "#cbd5e0",        # Clean border strokes
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "medium",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
    "figure.titlesize": 13,
    "figure.titleweight": "bold",
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "savefig.facecolor": "#ffffff",
    "savefig.edgecolor": "#ffffff",
    "text.color": COLORS["neutral_dark"],
    "axes.labelcolor": COLORS["neutral_dark"],
    "xtick.color": COLORS["neutral_dark"],
    "ytick.color": COLORS["neutral_dark"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": COLORS["neutral_mid"],
    "axes.linewidth": 0.8,
    "grid.color": COLORS["grid"],
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.8,
})


def save_figure(fig, out_dir: Path, basename: str, dpi: int = 300):
    """Save figure in both vector PDF and high-res raster PNG formats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{basename}.pdf"
    png_path = out_dir / f"{basename}.png"
    
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    pdf_size = pdf_path.stat().st_size
    png_size = png_path.stat().st_size
    print(f"Generated {basename}.pdf ({pdf_size / 1024:.1f} KB)")
    print(f"Generated {basename}.png ({png_size / 1024:.1f} KB, {dpi} DPI)")
    assert pdf_size > 10240, f"{pdf_path} too small ({pdf_size} bytes)"
    assert png_size > 10240, f"{png_path} too small ({png_size} bytes)"


# ==============================================================================
# Figure 1: Scaling Laws
# ==============================================================================

def plot_figure_1_scaling_laws(out_dir: Path):
    """
    Figure 1: Scaling Laws comparing Standard vs Algebraic models across
    125M, 350M, and 1B scales on FineWeb-Edu.
    Left panel: Training Compute (FLOPs) vs Held-Out NLL (nats).
    Right panel: Active Parameters vs Held-Out Perplexity (PPL).
    """
    scales = ["125M", "350M", "1B"]
    flops = np.array([1.853e18, 1.396e19, 1.170e20])
    params = np.array([123.55, 332.45, 975.28])  # Millions

    # NLL values (125M uses the 50k tokenizer; 350M/1B use the 32k tokenizer)
    std_nll = np.array([3.2001, 2.5845, 2.2961])
    std_nll_err = np.array([0.0037, 0.0018, 0.0]) # sample SD (1B single seed)

    alg_nll = np.array([3.1582, 2.5450, 2.2711])
    alg_nll_err = np.array([0.0044, 0.0019, 0.0]) # sample SD

    # Perplexity values
    std_ppl = np.array([24.53, 13.26, 9.94])
    std_ppl_err = np.array([0.09, 0.02, 0.0])

    alg_ppl = np.array([23.53, 12.74, 9.69])
    alg_ppl_err = np.array([0.10, 0.02, 0.0])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1.0]})
    
    # ------------------ Left Subplot: NLL vs Compute ------------------
    # Power-law linear fit in log10 space over the two 32k-tokenizer scales
    # (125M uses a different tokenizer, so it is plotted but excluded from fits).
    fit_idx = np.array([1, 2])
    log_flops = np.log10(flops[fit_idx])
    fit_flops = np.logspace(np.log10(1.0e18), np.log10(2.2e20), 100)
    log_fit_flops = np.log10(fit_flops)

    # Standard fit
    std_poly = np.polyfit(log_flops, std_nll[fit_idx], 1)
    std_fit_line = np.polyval(std_poly, log_fit_flops)

    # Algebraic fit
    alg_poly = np.polyfit(log_flops, alg_nll[fit_idx], 1)
    alg_fit_line = np.polyval(alg_poly, log_fit_flops)
    
    # Plot power-law projection lines
    ax1.plot(fit_flops, std_fit_line, color=COLORS["standard"], linestyle="--", alpha=0.5, lw=1.5,
             label=fr"Standard fit (Slope $= {std_poly[0]:.3f}/\mathrm{{dec}}$)")
    ax1.plot(fit_flops, alg_fit_line, color=COLORS["algebraic"], linestyle="--", alpha=0.5, lw=1.5,
             label=fr"Algebraic fit (Slope $= {alg_poly[0]:.3f}/\mathrm{{dec}}$)")
    
    # Plot empirical points (32k scales filled; 125M hollow: different tokenizer)
    ax1.errorbar(flops[fit_idx], std_nll[fit_idx], yerr=std_nll_err[fit_idx], fmt="o", color=COLORS["standard"],
                 ecolor=COLORS["standard"], elinewidth=1.6, capsize=4, capthick=1.2,
                 markersize=8, markeredgewidth=1.2, markeredgecolor="#ffffff",
                 label="Standard Transformer", zorder=5)

    ax1.errorbar(flops[fit_idx], alg_nll[fit_idx], yerr=alg_nll_err[fit_idx], fmt="D", color=COLORS["algebraic"],
                 ecolor=COLORS["algebraic"], elinewidth=1.6, capsize=4, capthick=1.2,
                 markersize=8, markeredgewidth=1.2, markeredgecolor="#ffffff",
                 label="Algebraic Transformer", zorder=5)

    ax1.errorbar(flops[0:1], std_nll[0:1], yerr=std_nll_err[0:1], fmt="o", color=COLORS["standard"],
                 ecolor=COLORS["standard"], elinewidth=1.6, capsize=4, capthick=1.2,
                 markersize=8, markeredgewidth=1.6, markerfacecolor="none",
                 label="125M (50k tokenizer)", zorder=5)

    ax1.errorbar(flops[0:1], alg_nll[0:1], yerr=alg_nll_err[0:1], fmt="D", color=COLORS["algebraic"],
                 ecolor=COLORS["algebraic"], elinewidth=1.6, capsize=4, capthick=1.2,
                 markersize=8, markeredgewidth=1.6, markerfacecolor="none",
                 zorder=5)

    # Annotate scale points and delta nats
    delta_labels = ["-0.0418 nats", "-0.0395 nats", "-0.0250 nats"]

    for i, (f, sn, an, sc, dl) in enumerate(zip(flops, std_nll, alg_nll, scales, delta_labels)):
        # Model scale label
        tag = f"{sc} (50k tok)" if sc == "125M" else sc
        ax1.annotate(f"{tag}\n({f/1e18:.1f}E18 FLOPs)" if f < 1e20 else f"{tag}\n({f/1e20:.1f}E20 FLOPs)",
                     (f, sn), textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=8.5, weight="semibold", color=COLORS["neutral_dark"])
        
        # Vertical delta arrow between Standard and Algebraic
        mid_y = (sn + an) / 2.0
        ax1.annotate("", xy=(f, an + 0.004), xytext=(f, sn - 0.004),
                     arrowprops=dict(arrowstyle="<->", color=COLORS["neutral_mid"], lw=1.2))
        ax1.text(f * 1.15, mid_y, dl, ha="left", va="center", fontsize=8.0,
                 weight="bold", color=COLORS["algebraic"],
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=COLORS["algebraic_light"],
                           edgecolor=COLORS["algebraic"], alpha=0.9, lw=0.6))
    
    # Compute Equivalence Horizontal Arrow (at 350M level)
    target_loss = 2.5450
    equiv_log_c = (target_loss - std_poly[1]) / std_poly[0]
    equiv_c = 10 ** equiv_log_c
    ax1.plot([flops[1], equiv_c], [target_loss, target_loss], color="#6b7280", linestyle=":", lw=1.4)
    ax1.scatter([equiv_c], [target_loss], marker="x", color=COLORS["standard"], s=50, zorder=6)
    ax1.annotate("~1.3x Compute Equivalence\n(equal loss at ~25% fewer FLOPs)",
                 xy=((flops[1] + equiv_c)/2, target_loss), xytext=(0, -22),
                 textcoords="offset points", ha="center", fontsize=8.0, weight="medium",
                 color=COLORS["neutral_dark"],
                 arrowprops=dict(arrowstyle="->", color="#6b7280", lw=0.8))

    ax1.set_xscale("log")
    ax1.set_xlim(8.0e17, 2.5e20)
    ax1.set_ylim(2.20, 3.34)
    ax1.set_xlabel(r"Training Compute ($C = 6ND$ in FLOPs)", labelpad=6)
    ax1.set_ylabel("Held-Out Validation NLL (nats/token)", labelpad=6)
    ax1.set_title("A. Scaling Trajectories on FineWeb-Edu", loc="left", pad=10)
    ax1.grid(True, which="both", axis="both")
    ax1.legend(loc="upper right", framealpha=0.9, facecolor="#ffffff", edgecolor=COLORS["border"])
    
    # ------------------ Right Subplot: PPL vs Parameters ------------------
    ax2.errorbar(params[fit_idx], std_ppl[fit_idx], yerr=std_ppl_err[fit_idx], fmt="o-", color=COLORS["standard"],
                 ecolor=COLORS["standard"], elinewidth=1.6, capsize=4, capthick=1.2,
                 markersize=8, markeredgewidth=1.2, markeredgecolor="#ffffff",
                 lw=1.8, label="Standard Transformer", zorder=5)

    ax2.errorbar(params[fit_idx], alg_ppl[fit_idx], yerr=alg_ppl_err[fit_idx], fmt="D-", color=COLORS["algebraic"],
                 ecolor=COLORS["algebraic"], elinewidth=1.6, capsize=4, capthick=1.2,
                 markersize=8, markeredgewidth=1.2, markeredgecolor="#ffffff",
                 lw=1.8, label="Algebraic Transformer", zorder=5)

    ax2.errorbar(params[0:1], std_ppl[0:1], yerr=std_ppl_err[0:1], fmt="o", color=COLORS["standard"],
                 ecolor=COLORS["standard"], elinewidth=1.6, capsize=4, capthick=1.2,
                 markersize=8, markeredgewidth=1.6, markerfacecolor="none",
                 label="125M (50k tokenizer)", zorder=5)

    ax2.errorbar(params[0:1], alg_ppl[0:1], yerr=alg_ppl_err[0:1], fmt="D", color=COLORS["algebraic"],
                 ecolor=COLORS["algebraic"], elinewidth=1.6, capsize=4, capthick=1.2,
                 markersize=8, markeredgewidth=1.6, markerfacecolor="none",
                 zorder=5)

    # Annotate PPL values and relative drops
    ppl_drops = ["-4.10%", "-3.87%", "-2.47%"]
    for i, (p, sp, ap, sc, drop) in enumerate(zip(params, std_ppl, alg_ppl, scales, ppl_drops)):
        x_off_std = -20 if i == 2 else -16
        y_off_std = 12 if i == 2 else 8
        ax2.annotate(f"{sp:.2f}", (p, sp), textcoords="offset points", xytext=(x_off_std, y_off_std),
                     ha="center", fontsize=8.0, color=COLORS["standard"], weight="semibold")
        ax2.annotate(f"{ap:.2f}", (p, ap), textcoords="offset points", xytext=(16, -14),
                     ha="center", fontsize=8.0, color=COLORS["algebraic"], weight="semibold")
        ax2.text(p * 1.15, (sp + ap)/2, drop, ha="left", va="center", fontsize=8.0,
                 weight="bold", color=COLORS["algebraic"])
    
    # Parameter Equivalence Callout (shifted to avoid colliding with data labels)
    ax2.annotate("~1.1-1.2x Parameter Equivalence\n(equivalent loss at ~9-14% fewer params)",
                 xy=(params[1], alg_ppl[1]), xytext=(0.40, 0.34),
                 textcoords="axes fraction", ha="center", fontsize=8.0, weight="medium",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["neutral_light"],
                           edgecolor=COLORS["border"], lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=COLORS["neutral_mid"], lw=0.9))

    ax2.set_xscale("log")
    ax2.set_xlim(80, 1400)
    ax2.set_ylim(8.8, 26.5)
    ax2.set_xticks([124, 332, 975])
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.set_xlabel("Active Parameters (Millions)", labelpad=6)
    ax2.set_ylabel("Held-Out Validation Perplexity (PPL)", labelpad=6)
    ax2.set_title("B. Parameter Efficiency & Perplexity", loc="left", pad=10)
    ax2.grid(True, which="both", axis="both")
    ax2.legend(loc="upper right", framealpha=0.9, facecolor="#ffffff", edgecolor=COLORS["border"])
    
    fig.text(0.5, -0.04,
             "Nearly two orders of compute magnitude. 125M: 5 seeds (50k tokenizer, hollow); 350M: 3 seeds; 1B: Seed 42. Trend lines fit to 32k scales.",
             ha="center", fontsize=8.5, color=COLORS["neutral_mid"], style="italic")
    
    fig.tight_layout()
    save_figure(fig, out_dir, "scaling_laws")


# ==============================================================================
# Figure 2: Architecture Diagram
# ==============================================================================

def plot_figure_3_architecture_diagram(out_dir: Path):
    """
    Figure 2: Side-by-side computation graph comparing Standard Transformer
    vs. Algebraic Transformer, illustrating the 4 exact component replacements.
    """
    fig, ax = plt.subplots(figsize=(12.8, 8.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    
    # Title Banner
    ax.text(50, 97.5, "Architecture Comparison: Transcendental Elimination",
            ha="center", va="center", fontsize=14, weight="bold", color=COLORS["neutral_dark"])
    ax.text(50, 94.5, "Standard Transcendental Architecture (Left) vs. Pure Algebraic Architecture (Right)",
            ha="center", va="center", fontsize=9.5, color=COLORS["neutral_mid"])
    
    # Column Headers
    # Left Header: Standard
    ax.add_patch(FancyBboxPatch((3, 88), 38.5, 4.5, boxstyle="round,pad=0.2",
                                facecolor=COLORS["standard_light"], edgecolor=COLORS["standard"], lw=1.5))
    ax.text(22.25, 90.2, "Standard Transformer (Baseline)", ha="center", va="center",
            fontsize=11, weight="bold", color=COLORS["standard"])
    ax.text(22.25, 87.0, r"Transcendental functions ($\exp, \sin, \cos, \log$)", ha="center", va="center",
            fontsize=8.0, color="#b91c1c", style="italic")
    
    # Right Header: Algebraic
    ax.add_patch(FancyBboxPatch((58.5, 88), 38.5, 4.5, boxstyle="round,pad=0.2",
                                facecolor=COLORS["algebraic_light"], edgecolor=COLORS["algebraic"], lw=1.5))
    ax.text(77.75, 90.2, "Pure Algebraic Transformer (Ours)", ha="center", va="center",
            fontsize=11, weight="bold", color=COLORS["algebraic"])
    ax.text(77.75, 87.0, "Rational arithmetic & reciprocal square root (lax.rsqrt)", ha="center", va="center",
            fontsize=8.0, color="#15803d", weight="semibold")
    
    # Center Divider Line
    ax.plot([50, 50], [5, 85], color=COLORS["grid"], linestyle=":", lw=1.5)
    
    # 4 Key Replacement Modules
    modules = [
        {
            "name": "1. Position",
            "y": 70,
            "std_title": "RoPE Rotary Position Embeddings",
            "std_eq": r"$R_{\Theta, m}^d = \mathrm{diag}(R_1, \dots, R_{d/2})$" + "\n" + r"$R_i = [\cos(m\theta_i), -\sin(m\theta_i); \sin(m\theta_i), \cos(m\theta_i)]$",
            "std_tag": r"Bottleneck: Transcendental $\sin, \cos$",
            "alg_title": "Geometric Cayley Rotations on SO(d)",
            "alg_eq": r"$R(A) = (I - A)(I + A)^{-1} \in SO(d)$" + "\n" + r"$A = -A^T$ (skew-symmetric rational matrix)" + "\n" + "Pure real division, exact orthogonality",
            "alg_tag": r"Rational matrix polynomial on $SO(d)$",
        },
        {
            "name": "2. Attention",
            "y": 52,
            "std_title": "Softmax Scaled Dot-Product Attention",
            "std_eq": r"$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}(QK^T / \sqrt{d_k}) V$" + "\n" + r"$\mathrm{softmax}(s)_i = \exp(s_i) / \sum_j \exp(s_j)$ (unbounded logits)",
            "std_tag": r"Bottleneck: Transcendental $\exp$, overflow risk",
            "alg_title": "Hyperbolic Asinh Attention Link + Learned Sink",
            "alg_eq": r"$a_{ij} = \rho(s_{ij}/8)^{16} / (\sum_k \rho(s_{ik}/8)^{16} + \rho(b/8)^{16}), \quad \mathbf{o}_i = \sum_{j \leq i} a_{ij}\mathbf{v}_j$" + "\n" + r"$\rho(x) = x + \sqrt{1 + x^2} = \exp(\mathrm{asinh}(x))$, learned sink $b$" + "\n" + r"Bounded polynomial attention, dimension-independent stability",
            "alg_tag": "Hardware-fused reciprocal square root",
        },
        {
            "name": "3. Feed-Forward",
            "y": 34,
            "std_title": "SwiGLU Gated Feed-Forward Network",
            "std_eq": r"$\mathrm{SwiGLU}(x) = (x W_g \cdot \mathrm{swish}(x W_u)) W_d$" + "\n" + r"$\mathrm{swish}(z) = z \cdot \sigma(z) = z / (1 + \exp(-z))$",
            "std_tag": r"Bottleneck: Transcendental Sigmoid $\sigma(z)$",
            "alg_title": "ALU-GLU Rational Disk Projection",
            "alg_eq": r"$\mathrm{ALU}$-$\mathrm{GLU}(h) = (\mathrm{ALU}(h W_g) \odot (h W_u)) W_d$" + "\n" + r"$\mathrm{ALU}(z) = \frac{z}{2}\left(1 + \frac{\beta z}{\sqrt{1 + \beta^2 z^2}}\right)$ with $\beta = 1/2$" + "\n" + "Evaluates purely via reciprocal square root rsqrt",
            "alg_tag": "Pure algebraic rational gating",
        },
        {
            "name": "4. Output & Loss",
            "y": 16,
            "std_title": "Cross-Entropy Softmax Loss",
            "std_eq": r"$\mathcal{L}_{\mathrm{CE}} = -\log (\exp(z_y) / \sum_j \exp(z_j))$" + "\n" + "Log-sum-exp transcendental reduction\nUnbounded logits require temperature tuning",
            "std_tag": r"Bottleneck: Transcendental $\exp, \log$",
            "alg_title": "Alg-CE Likelihood with Lorentzian Regularization",
            "alg_eq": r"$P(y) = \rho(z_y/2)^8 / \sum_j \rho(z_j/2)^8, \quad \mathcal{L} = -\log P(y)$" + "\n" + r"Intrinsic Lorentzian gradient damping: $\partial \mathcal{L} / \partial z_j \propto 1 / \sqrt{1 + (z_j/2)^2}$" + "\n" + "Eliminates logit divergence without explicit clipping",
            "alg_tag": "Lorentzian regularization, stable gradients",
        },
    ]
    
    # Draw Module Cards and Connecting Flow Arrows
    for mod in modules:
        y = mod["y"]
        h = 12.5
        
        # Section Label in Center
        ax.text(50, y + h/2, mod["name"], ha="center", va="center",
                fontsize=8.2, weight="bold", color=COLORS["neutral_dark"],
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor=COLORS["border"], lw=0.8))
        
        # Left Box (Standard)
        ax.add_patch(FancyBboxPatch((3, y - 0.5), 38.5, h, boxstyle="round,pad=0.3",
                                    facecolor="#f8fafc", edgecolor=COLORS["standard"], lw=1.2))
        ax.text(5, y + h - 2.2, mod["std_title"], ha="left", va="center",
                fontsize=9.0, weight="bold", color=COLORS["standard"])
        ax.text(5, y + h/2 - 0.5, mod["std_eq"], ha="left", va="center",
                fontsize=7.5, color=COLORS["neutral_dark"])
        ax.text(5, y + 1.2, mod["std_tag"], ha="left", va="center",
                fontsize=7.0, weight="semibold", color="#dc2626")
        
        # Right Box (Algebraic)
        ax.add_patch(FancyBboxPatch((58.5, y - 0.5), 38.5, h, boxstyle="round,pad=0.3",
                                    facecolor="#fffaf5", edgecolor=COLORS["algebraic"], lw=1.2))
        ax.text(60.5, y + h - 2.2, mod["alg_title"], ha="left", va="center",
                fontsize=9.0, weight="bold", color=COLORS["algebraic"])
        ax.text(60.5, y + h/2 - 0.5, mod["alg_eq"], ha="left", va="center",
                fontsize=7.5, color=COLORS["neutral_dark"])
        ax.text(60.5, y + 1.2, mod["alg_tag"], ha="left", va="center",
                fontsize=7.0, weight="semibold", color="#16a34a")
        
        # Horizontal Transformation Arrow between boxes
        ax.add_patch(FancyArrowPatch((42.0, y + h/2 - 1.8), (58.0, y + h/2 - 1.8),
                                     arrowstyle="-|>", mutation_scale=12, color=COLORS["neutral_mid"], lw=1.2))
        
    # Vertical Flow Arrows between modules
    for i in range(len(modules) - 1):
        y_curr = modules[i]["y"]
        y_next = modules[i+1]["y"] + 12.5
        # Standard vertical arrow
        ax.add_patch(FancyArrowPatch((22.25, y_curr - 0.8), (22.25, y_next + 0.5),
                                     arrowstyle="-|>", mutation_scale=10, color=COLORS["standard"], lw=1.0))
        # Algebraic vertical arrow
        ax.add_patch(FancyArrowPatch((77.75, y_curr - 0.8), (77.75, y_next + 0.5),
                                     arrowstyle="-|>", mutation_scale=10, color=COLORS["algebraic"], lw=1.0))
        
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    save_figure(fig, out_dir, "architecture_diagram")


# ==============================================================================
# Figure 3: Needle Recall Heatmap
# ==============================================================================

def plot_figure_4_needle_recall(out_dir: Path):
    """
    Figure 3: Exact Retrieval Accuracy across Context Budgets (256, 512, 1024, 2048)
    and Evidence Depths (0%, 25%, 50%, 75%, 100%) for Needle Passkey, Latest Value,
    Associative Recall, and Multi-Needle.
    Uses ground truth empirical data from Explorer 3 handoff report Section 1.E.
    """
    budgets = [256, 512, 1024, 2048]
    depths = ["0%", "25%", "50%", "75%", "100%"]
    
    # Ground truth matrix for Needle Passkey (350M Scale, present evidence)
    passkey_std = np.array([
        [10.42, 30.21, 60.42, 77.08, 77.08],
        [ 2.08, 20.83, 51.04, 60.42, 75.00],
        [ 0.00, 21.88, 29.17, 64.58, 77.08],
        [ 0.00, 33.33, 35.42, 35.42, 64.58]
    ])
    
    passkey_alg = np.array([
        [22.92, 67.71, 95.83, 96.88, 88.54],
        [ 2.08, 13.54, 31.25, 87.50, 85.42],
        [ 2.08, 14.58, 63.54, 84.38, 61.46],
        [ 0.00, 11.46, 19.79, 25.00, 33.33]
    ])

    # Dynamically verify or load from results/main_350m/recall_cells.csv if available
    csv_candidates = [
        Path("/home/tasma/algebraic-transformer/results/main_350m/recall_cells.csv"),
        Path("results/main_350m/recall_cells.csv"),
    ]
    for csv_path in csv_candidates:
        if csv_path.exists():
            import csv
            depth_floats = [0.0, 0.25, 0.5, 0.75, 1.0]
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    if row.get("task") == "needle_passkey" and row.get("condition") == "present":
                        b = int(row["context_budget"])
                        d = float(row["depth"])
                        if b in budgets and d in depth_floats:
                            r = budgets.index(b)
                            c = depth_floats.index(d)
                            val = round(float(row["mean"]) * 100, 2)
                            if row["architecture"] == "standard":
                                passkey_std[r, c] = val
                            elif row["architecture"] == "algebraic":
                                passkey_alg[r, c] = val
            break
    
    passkey_delta = passkey_alg - passkey_std
    
    fig = plt.figure(figsize=(13.6, 5.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.35], wspace=0.42)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    
    cmap_base = "Blues"
    cmap_alg = "Oranges"
    cmap_delta = "RdBu_r"
    
    # ------------------ Panel 1: Standard Passkey Heatmap ------------------
    im1 = ax1.imshow(passkey_std, vmin=0, vmax=100, cmap=cmap_base, aspect="auto")
    ax1.set_title("Standard Transformer\n(350M Passkey Accuracy %)", fontsize=9.5, color=COLORS["standard"], pad=8)
    ax1.set_xticks(range(len(depths)))
    ax1.set_xticklabels(depths, fontsize=8.0)
    ax1.set_yticks(range(len(budgets)))
    ax1.set_yticklabels(budgets, fontsize=8.5)
    ax1.set_xlabel("Evidence Depth in Context", labelpad=5)
    ax1.set_ylabel("Context Budget (tokens)", labelpad=5)
    
    for r in range(len(budgets)):
        for c in range(len(depths)):
            val = passkey_std[r, c]
            txt_color = "white" if val > 55 else COLORS["neutral_dark"]
            ax1.text(c, r, f"{val:.1f}%", ha="center", va="center", fontsize=7.8,
                     weight="semibold", color=txt_color)
            
    # ------------------ Panel 2: Algebraic Passkey Heatmap ------------------
    im2 = ax2.imshow(passkey_alg, vmin=0, vmax=100, cmap=cmap_alg, aspect="auto")
    ax2.set_title("Algebraic Transformer\n(350M Passkey Accuracy %)", fontsize=9.5, color=COLORS["algebraic"], pad=8)
    ax2.set_xticks(range(len(depths)))
    ax2.set_xticklabels(depths, fontsize=8.0)
    ax2.set_yticks(range(len(budgets)))
    ax2.set_yticklabels([])
    ax2.set_xlabel("Evidence Depth in Context", labelpad=5)
    
    for r in range(len(budgets)):
        for c in range(len(depths)):
            val = passkey_alg[r, c]
            txt_color = "white" if val > 55 else COLORS["neutral_dark"]
            ax2.text(c, r, f"{val:.1f}%", ha="center", va="center", fontsize=7.8,
                     weight="semibold", color=txt_color)
            
    # ------------------ Panel 3: Paired Difference Heatmap ------------------
    im3 = ax3.imshow(passkey_delta, vmin=-25, vmax=25, cmap=cmap_delta, aspect="auto")
    ax3.set_title(r"Paired Advantage ($\Delta$ pp)" + "\n(Algebraic minus Standard)", fontsize=9.5, color=COLORS["synergy"], pad=8)
    ax3.set_xticks(range(len(depths)))
    ax3.set_xticklabels(depths, fontsize=8.0)
    ax3.set_yticks(range(len(budgets)))
    ax3.set_yticklabels([])
    ax3.set_xlabel("Evidence Depth in Context", labelpad=5)
    
    for r in range(len(budgets)):
        for c in range(len(depths)):
            val = passkey_delta[r, c]
            txt_color = "white" if abs(val) > 12 else COLORS["neutral_dark"]
            prefix = "+" if val > 0 else ""
            ax3.text(c, r, f"{prefix}{val:.1f}", ha="center", va="center", fontsize=7.8,
                     weight="bold", color=txt_color)
            
    cbar3 = fig.colorbar(im3, ax=ax3, orientation="horizontal", pad=0.18, shrink=0.85)
    cbar3.set_label(r"$\Delta$ Accuracy (pp)", fontsize=8.0)
    cbar3.ax.tick_params(labelsize=7.5)
    
    # ------------------ Panel 4: Multi-Task Recall Summary across Scales ------------------
    tasks = ["Needle Passkey", "Latest Value", "Associative", "Multi-Needle"]
    std_350m = [41.30, 8.23, 2.45, 0.00]
    alg_350m = [45.36, 10.57, 3.02, 0.10]
    std_1b = [72.03, 17.03, 25.31, 0.47]
    alg_1b = [77.34, 17.81, 25.31, 0.47]
    
    y_tasks = np.arange(len(tasks))
    h = 0.18
    
    ax4.barh(y_tasks + 1.5*h, std_350m, height=h, color="#93c5fd", label="Standard 350M", zorder=3)
    ax4.barh(y_tasks + 0.5*h, alg_350m, height=h, color="#fdba74", label="Algebraic 350M", zorder=3)
    ax4.barh(y_tasks - 0.5*h, std_1b, height=h, color=COLORS["standard"], label="Standard 1B", zorder=3)
    ax4.barh(y_tasks - 1.5*h, alg_1b, height=h, color=COLORS["algebraic"], label="Algebraic 1B", zorder=3)
    
    for idx, (s1, a1) in enumerate(zip(std_1b, alg_1b)):
        d1 = a1 - s1
        sign = "+" if d1 >= 0 else ""
        ax4.text(a1 + 1.5, idx - 1.5*h, f"{sign}{d1:.1f} pp", va="center", fontsize=7.2,
                 weight="bold", color=COLORS["algebraic"])
        
    ax4.set_yticks(y_tasks)
    ax4.set_yticklabels(tasks, fontsize=8.5)
    ax4.invert_yaxis()
    ax4.set_xlabel("Present-Evidence Exact Retrieval (%)", labelpad=5)
    ax4.set_xlim(0, 95)
    ax4.set_title("Multi-Task Recall across Scales\n(350M & 1B Parameter Models)", fontsize=9.5, color=COLORS["neutral_dark"], pad=8)
    ax4.grid(True, axis="x")
    ax4.legend(loc="lower right", fontsize=7.5, framealpha=0.9, facecolor="#ffffff", edgecolor=COLORS["border"])
    
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.14, wspace=0.42)
    save_figure(fig, out_dir, "needle_recall")


# ==============================================================================
# Master Pipeline & Integrity Verification
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate publication figures for Algebraic Transformer paper.")
    default_out = Path("/home/tasma/algebraic-transformer/paper/figures")
    parser.add_argument("--out-dir", type=Path, default=default_out,
                        help=f"Output directory for figures (default: {default_out})")
    args = parser.parse_args()
    
    out_dir = args.out_dir
    print(f"Generating all 3 publication figures into: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Scaling Laws
    print("\n--- Generating Figure 1: Scaling Laws ---")
    plot_figure_1_scaling_laws(out_dir)

    # 2. Architecture Diagram
    print("\n--- Generating Figure 2: Architecture Diagram ---")
    plot_figure_3_architecture_diagram(out_dir)

    # 3. Needle Recall Heatmap
    print("\n--- Generating Figure 3: Needle Recall Heatmap ---")
    plot_figure_4_needle_recall(out_dir)

    # Verification check
    expected_files = [
        "scaling_laws.pdf", "scaling_laws.png",
        "architecture_diagram.pdf", "architecture_diagram.png",
        "needle_recall.pdf", "needle_recall.png"
    ]
    
    print("\n--- Verification Summary ---")
    all_passed = True
    for fname in expected_files:
        fpath = out_dir / fname
        if not fpath.exists():
            print(f"FAILED: Missing {fname}")
            all_passed = False
            continue
        size_kb = fpath.stat().st_size / 1024
        if size_kb < 10.0:
            print(f"FAILED: {fname} is too small ({size_kb:.1f} KB < 10 KB)")
            all_passed = False
        else:
            print(f"PASSED: {fname:<25} ({size_kb:>6.1f} KB)")
            
    if all_passed:
        print("\nAll 6 publication figure files successfully generated and verified!")
    else:
        print("\nVerification FAILED: One or more figures did not meet criteria.")
        sys.exit(1)


if __name__ == "__main__":
    main()

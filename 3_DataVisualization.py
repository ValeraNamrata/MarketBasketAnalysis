# =============================================================================
# Phase 3 — Data Visualisations
# Generates 6 charts to understand both datasets before modeling
# Run AFTER phase3_data_preparation.py (needs transaction matrices)
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

OUTPUT_DIR  = "data/phase3/"
FIGURES_DIR = "data/phase3/figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Load matrices ──────────────────────────────────────────────
dh = pd.read_csv(os.path.join(OUTPUT_DIR, "dunnhumby_transaction_matrix.csv"), index_col="basket_id")
ic = pd.read_csv(os.path.join(OUTPUT_DIR, "instacart_transaction_matrix.csv"), index_col="order_id")

CATEGORIES = list(dh.columns)

# ── Shared style ───────────────────────────────────────────────
DH_COLOR  = "#2E75B6"   # blue  — Dunnhumby
IC_COLOR  = "#1F7A4A"   # green — Instacart
GRAY      = "#888888"
plt.rcParams.update({
    "font.family"     : "sans-serif",
    "font.size"       : 11,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "figure.dpi"      : 130,
})

# =============================================================================
# CHART 1 — Category Support % Side-by-Side Bar Chart
# "Which categories appear most frequently in baskets at each retailer?"
# =============================================================================

dh_support = (dh.mean() * 100).sort_values(ascending=False)
ic_support = (ic.mean() * 100).reindex(dh_support.index)

x     = np.arange(len(CATEGORIES))
width = 0.38

fig, ax = plt.subplots(figsize=(13, 5.5))
ax.bar(x - width/2, dh_support.values, width, color=DH_COLOR, label="Dunnhumby", zorder=3)
ax.bar(x + width/2, ic_support.values, width, color=IC_COLOR,  label="Instacart",  zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(dh_support.index, rotation=35, ha="right", fontsize=9.5)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.set_ylabel("% of baskets containing category")
ax.set_title("Chart 1 — Category Support: Dunnhumby vs Instacart", fontsize=13, fontweight="bold", pad=12)
ax.legend(frameon=False)
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax.set_ylim(0, max(dh_support.max(), ic_support.max()) * 1.15)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "chart1_category_support.png"), bbox_inches="tight")
plt.close()
print("Saved: chart1_category_support.png")

# =============================================================================
# CHART 2 — Rank Divergence Dot Plot
# "Which categories shifted the most in rank between the two retailers?"
# =============================================================================

dh_rank = dh_support.rank(ascending=False).astype(int)
ic_rank = (ic.mean() * 100).rank(ascending=False).astype(int)
shift   = (ic_rank - dh_rank).reindex(dh_rank.index)
shift_sorted = shift.sort_values()

colors = [DH_COLOR if v > 0 else IC_COLOR for v in shift_sorted.values]

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(shift_sorted.index, shift_sorted.values, color=colors, zorder=3)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Rank shift  (positive = higher rank at Instacart)")
ax.set_title("Chart 2 — Category Rank Divergence (Dunnhumby → Instacart)",
             fontsize=13, fontweight="bold", pad=12)
ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)

# Annotate bars
for bar, val in zip(bars, shift_sorted.values):
    ax.text(val + (0.15 if val >= 0 else -0.15), bar.get_y() + bar.get_height()/2,
            f"{val:+d}", va="center", ha="left" if val >= 0 else "right", fontsize=9)

# Legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=IC_COLOR, label="Higher rank at Instacart"),
                   Patch(color=DH_COLOR, label="Higher rank at Dunnhumby")],
          frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "chart2_rank_divergence.png"), bbox_inches="tight")
plt.close()
print("Saved: chart2_rank_divergence.png")

# =============================================================================
# CHART 3 — Basket Size Distribution (categories per basket)
# "How many categories does a typical basket span?"
# =============================================================================

dh_basket_size = dh.sum(axis=1)
ic_basket_size = ic.sum(axis=1)

bins = range(1, 16)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

for ax, data, color, label, n in zip(
        axes,
        [dh_basket_size, ic_basket_size],
        [DH_COLOR, IC_COLOR],
        ["Dunnhumby", "Instacart"],
        [len(dh), len(ic)]):
    counts, edges = np.histogram(data, bins=bins)
    pct = counts / counts.sum() * 100
    ax.bar(edges[:-1], pct, color=color, width=0.75, zorder=3)
    ax.set_xlabel("Number of categories in basket")
    ax.set_ylabel("% of baskets")
    ax.set_title(f"{label}  (n={n:,})", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.axvline(data.mean(), color="red", linewidth=1.2, linestyle="--",
               label=f"Mean: {data.mean():.1f}")
    ax.legend(frameon=False, fontsize=9)

fig.suptitle("Chart 3 — Basket Size Distribution (categories per basket)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "chart3_basket_size_distribution.png"), bbox_inches="tight")
plt.close()
print("Saved: chart3_basket_size_distribution.png")

# =============================================================================
# CHART 4 — Co-occurrence Heatmap: Dunnhumby
# "Which category pairs most often appear together in the same basket?"
# =============================================================================

def cooccurrence_pct(matrix):
    """Co-occurrence as % of all baskets containing both categories."""
    m = matrix.values.astype(float)
    n = len(m)
    co = (m.T @ m) / n * 100
    return pd.DataFrame(co, index=matrix.columns, columns=matrix.columns)

dh_co = cooccurrence_pct(dh)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(dh_co.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(CATEGORIES)))
ax.set_yticks(range(len(CATEGORIES)))
ax.set_xticklabels(CATEGORIES, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(CATEGORIES, fontsize=8)
plt.colorbar(im, ax=ax, label="% of baskets containing both")

# Annotate cells
for i in range(len(CATEGORIES)):
    for j in range(len(CATEGORIES)):
        val = dh_co.values[i, j]
        if i != j and val > 1:
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=6.5, color="white" if val > 12 else "black")

ax.set_title("Chart 4 — Category Co-occurrence Heatmap: Dunnhumby",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "chart4_cooccurrence_dunnhumby.png"), bbox_inches="tight")
plt.close()
print("Saved: chart4_cooccurrence_dunnhumby.png")

# =============================================================================
# CHART 5 — Co-occurrence Heatmap: Instacart
# =============================================================================

ic_co = cooccurrence_pct(ic)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(ic_co.values, cmap="Greens", aspect="auto")
ax.set_xticks(range(len(CATEGORIES)))
ax.set_yticks(range(len(CATEGORIES)))
ax.set_xticklabels(CATEGORIES, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(CATEGORIES, fontsize=8)
plt.colorbar(im, ax=ax, label="% of orders containing both")

for i in range(len(CATEGORIES)):
    for j in range(len(CATEGORIES)):
        val = ic_co.values[i, j]
        if i != j and val > 1:
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=6.5, color="white" if val > 15 else "black")

ax.set_title("Chart 5 — Category Co-occurrence Heatmap: Instacart",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "chart5_cooccurrence_instacart.png"), bbox_inches="tight")
plt.close()
print("Saved: chart5_cooccurrence_instacart.png")

# =============================================================================
# CHART 6 — Top 10 Co-occurring Pairs: Both Retailers Side by Side
# "What are the strongest category pair relationships at each retailer?"
# =============================================================================

def top_pairs(co_matrix, n=10):
    """Extract top N off-diagonal category pairs by co-occurrence %."""
    pairs = []
    cats = co_matrix.columns.tolist()
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            pairs.append((cats[i], cats[j], co_matrix.iloc[i, j]))
    return (pd.DataFrame(pairs, columns=["Cat A", "Cat B", "Pct"])
              .sort_values("Pct", ascending=False)
              .head(n)
              .reset_index(drop=True))

dh_pairs = top_pairs(dh_co)
ic_pairs = top_pairs(ic_co)

dh_pairs["Pair"] = dh_pairs["Cat A"].str[:12] + " +\n" + dh_pairs["Cat B"].str[:12]
ic_pairs["Pair"] = ic_pairs["Cat A"].str[:12] + " +\n" + ic_pairs["Cat B"].str[:12]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, pairs, color, label in zip(
        axes,
        [dh_pairs, ic_pairs],
        [DH_COLOR, IC_COLOR],
        ["Dunnhumby", "Instacart"]):
    ax.barh(pairs["Pair"][::-1], pairs["Pct"][::-1], color=color, zorder=3)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_xlabel("% of baskets containing both")
    ax.set_title(f"{label} — Top 10 Co-occurring Pairs", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    for i, (_, row) in enumerate(pairs[::-1].iterrows()):
        ax.text(row["Pct"] + 0.2, i, f"{row['Pct']:.1f}%", va="center", fontsize=8.5)

fig.suptitle("Chart 6 — Top 10 Category Co-occurrence Pairs",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "chart6_top_pairs.png"), bbox_inches="tight")
plt.close()
print("Saved: chart6_top_pairs.png")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"""
All 6 charts saved to: {FIGURES_DIR}

  chart1_category_support.png       — Support % side-by-side bar chart
  chart2_rank_divergence.png        — Rank shift dot plot
  chart3_basket_size_distribution.png — Basket size histograms
  chart4_cooccurrence_dunnhumby.png — Co-occurrence heatmap (DH)
  chart5_cooccurrence_instacart.png — Co-occurrence heatmap (IC)
  chart6_top_pairs.png              — Top 10 co-occurring pairs (both)

Ready for Phase 4 — Modeling.
""")
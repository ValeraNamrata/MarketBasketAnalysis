# =============================================================================
# Phase 4 -- Stage 1: Parameter Sensitivity Comparison
#
# Tests Apriori and FP-Growth across multiple parameter combinations
# on 50,000-row samples from BOTH Dunnhumby and Instacart datasets.
#
# Variables tested:
#   Support    : 0.5%, 1%, 2%, 5%
#   Confidence : 20%, 40%, 60%
#   Lift       : 1.0, 1.2, 1.5
#
# Output:
#   - Full results saved to data/phase4/stage1_results.csv
#   - Summary table printed to console
# =============================================================================

import pandas as pd
import time
import os
import itertools
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────
DH_PATH      = "data/phase3/dunnhumby_transaction_matrix.csv"
IC_PATH      = "data/phase3/instacart_transaction_matrix.csv"
OUTPUT_DIR   = "data/phase4"
SAMPLE_SIZE  = 50_000
RANDOM_STATE = 42

SUPPORT_LEVELS    = [0.005, 0.01, 0.02, 0.05]   # 0.5%, 1%, 2%, 5%
CONFIDENCE_LEVELS = [0.20, 0.40, 0.60]           # 20%, 40%, 60%
LIFT_LEVELS       = [1.0, 1.2, 1.5]              # min lift filter
ALGORITHMS        = ["apriori", "fpgrowth"]
DATASETS          = ["dunnhumby", "instacart"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load and sample both datasets ─────────────────────────────
print("=" * 70)
print("PHASE 4 -- Stage 1: Parameter Sensitivity Comparison")
print("=" * 70)

print(f"\nLoading datasets (sample size: {SAMPLE_SIZE:,} rows each)...")

dh = pd.read_csv(DH_PATH, index_col="basket_id").astype(bool)
ic = pd.read_csv(IC_PATH, index_col="order_id").astype(bool)

dh_sample = dh.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
ic_sample = ic.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

datasets = {
    "dunnhumby": dh_sample,
    "instacart": ic_sample,
}

print(f"  Dunnhumby sample : {dh_sample.shape[0]:,} baskets x {dh_sample.shape[1]} categories | density {dh_sample.values.mean()*100:.1f}%")
print(f"  Instacart sample : {ic_sample.shape[0]:,} orders  x {ic_sample.shape[1]} categories | density {ic_sample.values.mean()*100:.1f}%")

# ── Run all combinations ──────────────────────────────────────
total_runs = len(ALGORITHMS) * len(DATASETS) * len(SUPPORT_LEVELS) * len(CONFIDENCE_LEVELS) * len(LIFT_LEVELS)
print(f"\nTotal parameter combinations to test: {total_runs}")
print("Running...\n")

results = []
run_num = 0

for dataset_name, algo_name in itertools.product(DATASETS, ALGORITHMS):
    data = datasets[dataset_name]

    for support in SUPPORT_LEVELS:
        # Run algorithm once per support level (not per confidence/lift)
        # Confidence and lift are post-filters on the same rule set
        try:
            t0 = time.time()

            if algo_name == "apriori":
                itemsets = apriori(
                    data,
                    min_support=support,
                    use_colnames=True,
                    max_len=3
                )
            else:
                itemsets = fpgrowth(
                    data,
                    min_support=support,
                    use_colnames=True,
                    max_len=3
                )

            algo_time = round(time.time() - t0, 2)
            n_itemsets = len(itemsets)

            if n_itemsets == 0:
                # No itemsets -- record zero rows for all conf/lift combos
                for conf in CONFIDENCE_LEVELS:
                    for lift in LIFT_LEVELS:
                        run_num += 1
                        results.append({
                            "run":          run_num,
                            "dataset":      dataset_name,
                            "algorithm":    algo_name,
                            "support_pct":  f"{support*100:.1f}%",
                            "confidence":   f"{conf*100:.0f}%",
                            "min_lift":     lift,
                            "n_itemsets":   0,
                            "n_rules_raw":  0,
                            "n_rules_lift": 0,
                            "exec_time_s":  algo_time,
                            "note":         "No itemsets at this support level",
                        })
                continue

            # Generate rules at minimum confidence threshold (20%)
            # Then filter by higher confidence and lift in post-processing
            all_rules = association_rules(
                itemsets,
                metric="confidence",
                min_threshold=min(CONFIDENCE_LEVELS)
            )

            for conf in CONFIDENCE_LEVELS:
                for lift in LIFT_LEVELS:
                    run_num += 1

                    # Apply confidence and lift filters
                    filtered = all_rules[
                        (all_rules["confidence"] >= conf) &
                        (all_rules["lift"] >= lift)
                    ]

                    note = ""
                    if len(filtered) == 0:
                        note = "No rules survive this conf+lift combination"
                    elif len(filtered) > 500:
                        note = "High rule count -- consider raising thresholds"
                    elif len(filtered) < 5:
                        note = "Very few rules -- consider lowering thresholds"

                    results.append({
                        "run":          run_num,
                        "dataset":      dataset_name,
                        "algorithm":    algo_name,
                        "support_pct":  f"{support*100:.1f}%",
                        "confidence":   f"{conf*100:.0f}%",
                        "min_lift":     lift,
                        "n_itemsets":   n_itemsets,
                        "n_rules_raw":  len(all_rules[all_rules["confidence"] >= conf]),
                        "n_rules_lift": len(filtered),
                        "exec_time_s":  algo_time,
                        "note":         note,
                    })

            print(f"  [{dataset_name:<12}] [{algo_name:<9}] support={support*100:.1f}% | "
                  f"itemsets={n_itemsets:>4} | time={algo_time:.2f}s")

        except Exception as e:
            for conf in CONFIDENCE_LEVELS:
                for lift in LIFT_LEVELS:
                    run_num += 1
                    results.append({
                        "run":          run_num,
                        "dataset":      dataset_name,
                        "algorithm":    algo_name,
                        "support_pct":  f"{support*100:.1f}%",
                        "confidence":   f"{conf*100:.0f}%",
                        "min_lift":     lift,
                        "n_itemsets":   "ERROR",
                        "n_rules_raw":  "ERROR",
                        "n_rules_lift": "ERROR",
                        "exec_time_s":  "ERROR",
                        "note":         str(e),
                    })

# ── Save full results ─────────────────────────────────────────
results_df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "stage1_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nFull results saved -> {csv_path}")

# ── Print summary table ───────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY TABLE -- Rules surviving confidence + lift filters")
print("=" * 70)
print("(Showing n_rules_lift -- rules that pass BOTH confidence and lift)\n")

# Pivot for readability: rows = dataset+support, cols = conf+lift
pivot_data = []
for dataset_name in DATASETS:
    for support in SUPPORT_LEVELS:
        row = {
            "Dataset":  dataset_name,
            "Support":  f"{support*100:.1f}%",
        }
        for algo in ALGORITHMS:
            for conf in CONFIDENCE_LEVELS:
                for lift in LIFT_LEVELS:
                    match = results_df[
                        (results_df["dataset"]    == dataset_name) &
                        (results_df["algorithm"]  == algo) &
                        (results_df["support_pct"]== f"{support*100:.1f}%") &
                        (results_df["confidence"] == f"{conf*100:.0f}%") &
                        (results_df["min_lift"]   == lift)
                    ]
                    col = f"{algo[:2].upper()} c={conf*100:.0f}% l={lift}"
                    row[col] = match["n_rules_lift"].values[0] if len(match) else "N/A"
        pivot_data.append(row)

pivot_df = pd.DataFrame(pivot_data)
print(pivot_df.to_string(index=False))

# ── Speed comparison ──────────────────────────────────────────
print("\n" + "=" * 70)
print("SPEED COMPARISON -- Avg execution time per support level (seconds)")
print("=" * 70)

speed = results_df.groupby(["dataset", "algorithm", "support_pct"])["exec_time_s"].first().reset_index()
speed = speed[speed["exec_time_s"] != "ERROR"]
speed["exec_time_s"] = speed["exec_time_s"].astype(float)
print(speed.pivot_table(
    index=["dataset", "support_pct"],
    columns="algorithm",
    values="exec_time_s"
).to_string())

# ── Recommended parameter ranges ─────────────────────────────
print("\n" + "=" * 70)
print("PARAMETER RANGES PRODUCING MEANINGFUL RULE COUNTS")
print("=" * 70)

# Get Dunnhumby and Instacart rule counts separately
dh_df = results_df[results_df["dataset"] == "dunnhumby"][
    ["algorithm", "support_pct", "confidence", "min_lift", "n_rules_lift"]
].rename(columns={"n_rules_lift": "n_dunnhumby_rules"})

ic_df = results_df[results_df["dataset"] == "instacart"][
    ["algorithm", "support_pct", "confidence", "min_lift", "n_rules_lift"]
].rename(columns={"n_rules_lift": "n_instacart_rules"})

# Merge both datasets on the shared parameter columns
combined = pd.merge(
    dh_df,
    ic_df,
    on=["algorithm", "support_pct", "confidence", "min_lift"]
)

# Filter to rows where BOTH datasets produce meaningful rule counts
meaningful = combined[
    combined["n_instacart_rules"].apply(
        lambda x: str(x).isdigit() and 50 <= int(x) <= 500
    ) &
    combined["n_dunnhumby_rules"].apply(
        lambda x: str(x).isdigit() and 50 <= int(x) <= 900
    )
]

if len(meaningful) > 0:
    print(meaningful[[
        "algorithm", "support_pct", "confidence", "min_lift",
        "n_dunnhumby_rules", "n_instacart_rules"
    ]].to_string(index=False))
else:
    print("  No combinations produced between 100 and 500 rules for BOTH datasets.")
    print("  Showing all combinations with Instacart in range:")
    fallback = combined[
        combined["n_instacart_rules"].apply(
            lambda x: str(x).isdigit() and 100 <= int(x) <= 500
        )
    ]
    print(fallback[[
        "algorithm", "support_pct", "confidence", "min_lift",
        "n_dunnhumby_rules", "n_instacart_rules"
    ]].to_string(index=False))

print("\n" + "=" * 70)
print("Stage 1 complete. Review stage1_results.csv for full details.")
print("Use these results to select final parameters for Stage 2.")
print("=" * 70)


# Filter to FP-Growth only to avoid duplicate points
plot_df = combined[combined["algorithm"] == "fpgrowth"].copy()
plot_df["n_dunnhumby_rules"] = pd.to_numeric(plot_df["n_dunnhumby_rules"], errors="coerce")
plot_df["n_instacart_rules"] = pd.to_numeric(plot_df["n_instacart_rules"], errors="coerce")
plot_df = plot_df.dropna(subset=["n_dunnhumby_rules", "n_instacart_rules"])

# Label for each point
plot_df["label"] = (
    "supp=" + plot_df["support_pct"] +
    " conf=" + plot_df["confidence"] +
    " lift=" + plot_df["min_lift"].astype(str)
)

# Mark chosen combination
plot_df["chosen"] = (
    (plot_df["support_pct"] == "1.0%") &
    (plot_df["confidence"]  == "40%") &
    (plot_df["min_lift"]    == 1.2)
)

fig, ax = plt.subplots(figsize=(10, 7))

# All other combinations -- grey
ax.scatter(
    plot_df[~plot_df["chosen"]]["n_dunnhumby_rules"],
    plot_df[~plot_df["chosen"]]["n_instacart_rules"],
    color="steelblue", alpha=0.5, s=60, label="Other combinations"
)

# Chosen combination -- red star
chosen_row = plot_df[plot_df["chosen"]]
ax.scatter(
    chosen_row["n_dunnhumby_rules"],
    chosen_row["n_instacart_rules"],
    color="red", s=300, marker="*", zorder=5, label="Chosen parameters"
)

# Annotate the chosen point
ax.annotate(
    "CHOSEN\nFP-Growth | 1% | 40% conf | lift 1.2\nDH: 838 rules  IC: 438 rules",
    xy=(chosen_row["n_dunnhumby_rules"].values[0],
        chosen_row["n_instacart_rules"].values[0]),
    xytext=(600, 350),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="red"),
    color="red",
    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="red")
)

# Reference lines for the meaningful range bounds
ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axhline(y=500, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axvline(x=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axvline(x=500, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(105, 505, "meaningful range boundary", fontsize=7, color="gray")

ax.set_xlabel("Dunnhumby Rule Count", fontsize=11)
ax.set_ylabel("Instacart Rule Count", fontsize=11)
ax.set_title("Parameter Combinations -- Dunnhumby vs Instacart Rule Counts\n(FP-Growth only, all support / confidence / lift combinations)", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("data/phase4/parameter_selection_chart.png", dpi=150)
print("  Chart saved -> data/phase4/parameter_selection_chart.png")
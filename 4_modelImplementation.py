# =============================================================================
# Phase 4 -- Stage 2: Full FP-Growth Run
#
# Runs FP-Growth on the FULL Dunnhumby and Instacart transaction matrices.
#
# Parameters (agreed from Stage 1 analysis):
#   Algorithm      : FP-Growth
#   Min Support    : 1%
#   Min Confidence : 40%
#   Min Lift       : 1.2 (primary comparison threshold)
#   Max itemset    : 3 (pairs and triplets)
#
# Supplementary:
#   Instacart also filtered at lift 1.5 to document high-support effect
#
# Outputs:
#   data/phase4/dunnhumby_rules.csv
#   data/phase4/instacart_rules.csv
#   data/phase4/instacart_rules_lift15.csv  (supplementary)
#   data/phase4/stage2_summary.csv
# =============================================================================

import pandas as pd
import time
import os
import warnings
warnings.filterwarnings("ignore")

from mlxtend.frequent_patterns import fpgrowth, association_rules

# ── Config ────────────────────────────────────────────────────
DH_PATH      = "data/phase3/dunnhumby_transaction_matrix.csv"
IC_PATH      = "data/phase3/instacart_transaction_matrix.csv"
OUTPUT_DIR   = "data/phase4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_SUPPORT    = 0.01   # 1%
MIN_CONFIDENCE = 0.40   # 40%
MIN_LIFT       = 1.2    # primary threshold
MIN_LIFT_SUPP  = 1.5    # supplementary threshold for Instacart
MAX_LEN        = 3      # pairs and triplets
TOP_N          = 20     # rules to preview in console

print("=" * 65)
print("PHASE 4 -- Stage 2: Full FP-Growth Run")
print("=" * 65)
print(f"\nParameters:")
print(f"  Algorithm      : FP-Growth")
print(f"  Min Support    : {MIN_SUPPORT*100:.1f}%")
print(f"  Min Confidence : {MIN_CONFIDENCE*100:.0f}%")
print(f"  Min Lift       : {MIN_LIFT} (primary)")
print(f"  Max itemset    : {MAX_LEN}")

# ── Helper: clean rule dataframe ──────────────────────────────
def clean_rules(rules_df):
    """Convert frozensets to readable strings and round metrics."""
    df = rules_df.copy()
    df["antecedents"] = df["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    df["consequents"] = df["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
    df["support"]     = df["support"].round(4)
    df["confidence"]  = df["confidence"].round(4)
    df["lift"]        = df["lift"].round(4)
    return df[["antecedents", "consequents", "support", "confidence", "lift"]]

# ── Helper: print top N rules ─────────────────────────────────
def print_top_rules(rules_df, dataset_name, n=TOP_N):
    top = rules_df.nlargest(n, "lift").reset_index(drop=True)
    print(f"\n  Top {n} rules by lift -- {dataset_name}:")
    print(f"  {'#':<4} {'Antecedent':<28} {'Consequent':<22} {'Supp':>6} {'Conf':>6} {'Lift':>6}")
    print(f"  {'-'*4} {'-'*28} {'-'*22} {'-'*6} {'-'*6} {'-'*6}")
    for i, row in top.iterrows():
        print(f"  {i+1:<4} {row['antecedents']:<28} {row['consequents']:<22} "
              f"{row['support']:>6.3f} {row['confidence']:>6.3f} {row['lift']:>6.3f}")

# =============================================================================
# DUNNHUMBY -- Full Run
# =============================================================================
print("\n" + "=" * 65)
print("DATASET 1: Dunnhumby (full -- 240,855 baskets)")
print("=" * 65)

print("\nLoading matrix...")
dh = pd.read_csv(DH_PATH, index_col="basket_id").astype(bool)
print(f"  Shape   : {dh.shape[0]:,} baskets x {dh.shape[1]} categories")
print(f"  Density : {dh.values.mean()*100:.1f}%")

print("\nRunning FP-Growth...")
t0 = time.time()
dh_itemsets = fpgrowth(dh, min_support=MIN_SUPPORT, use_colnames=True, max_len=MAX_LEN)
dh_fp_time = round(time.time() - t0, 2)
print(f"  Frequent itemsets : {len(dh_itemsets):,}")
print(f"  FP-Growth time    : {dh_fp_time}s")

print("\nGenerating association rules...")
t0 = time.time()
dh_rules_all = association_rules(dh_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
dh_rules = dh_rules_all[dh_rules_all["lift"] >= MIN_LIFT].copy()
dh_rules_time = round(time.time() - t0, 2)

dh_rules_clean = clean_rules(dh_rules)

print(f"  Rules at conf >= {MIN_CONFIDENCE*100:.0f}%             : {len(dh_rules_all):,}")
print(f"  Rules after lift >= {MIN_LIFT} filter   : {len(dh_rules):,}")
print(f"  Rule generation time               : {dh_rules_time}s")
print(f"  Lift range                         : {dh_rules['lift'].min():.3f} -- {dh_rules['lift'].max():.3f}")
print(f"  Avg lift                           : {dh_rules['lift'].mean():.3f}")

print_top_rules(dh_rules_clean, "Dunnhumby")

dh_out = os.path.join(OUTPUT_DIR, "dunnhumby_rules.csv")
dh_rules_clean.sort_values("lift", ascending=False).to_csv(dh_out, index=False)
print(f"\n  Saved -> {dh_out}")

# =============================================================================
# INSTACART -- Full Run
# =============================================================================
print("\n" + "=" * 65)
print("DATASET 2: Instacart (full -- 3,213,813 orders)")
print("=" * 65)
print("\nNote: this may take several minutes on the full dataset.")

print("\nLoading matrix...")
ic = pd.read_csv(IC_PATH, index_col="order_id").astype(bool)
print(f"  Shape   : {ic.shape[0]:,} orders x {ic.shape[1]} categories")
print(f"  Density : {ic.values.mean()*100:.1f}%")

print("\nRunning FP-Growth...")
t0 = time.time()
ic_itemsets = fpgrowth(ic, min_support=MIN_SUPPORT, use_colnames=True, max_len=MAX_LEN)
ic_fp_time = round(time.time() - t0, 2)
print(f"  Frequent itemsets : {len(ic_itemsets):,}")
print(f"  FP-Growth time    : {ic_fp_time}s")

print("\nGenerating association rules...")
t0 = time.time()
ic_rules_all = association_rules(ic_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
ic_rules = ic_rules_all[ic_rules_all["lift"] >= MIN_LIFT].copy()
ic_rules_time = round(time.time() - t0, 2)

ic_rules_clean = clean_rules(ic_rules)

print(f"  Rules at conf >= {MIN_CONFIDENCE*100:.0f}%             : {len(ic_rules_all):,}")
print(f"  Rules after lift >= {MIN_LIFT} filter   : {len(ic_rules):,}")
print(f"  Rule generation time               : {ic_rules_time}s")
print(f"  Lift range                         : {ic_rules['lift'].min():.3f} -- {ic_rules['lift'].max():.3f}")
print(f"  Avg lift                           : {ic_rules['lift'].mean():.3f}")

print_top_rules(ic_rules_clean, "Instacart")

ic_out = os.path.join(OUTPUT_DIR, "instacart_rules.csv")
ic_rules_clean.sort_values("lift", ascending=False).to_csv(ic_out, index=False)
print(f"\n  Saved -> {ic_out}")

# =============================================================================
# INSTACART -- Supplementary at lift 1.5
# =============================================================================
print("\n" + "-" * 65)
print("SUPPLEMENTARY: Instacart at lift >= 1.5")
print("-" * 65)

ic_rules_15 = ic_rules_all[ic_rules_all["lift"] >= MIN_LIFT_SUPP].copy()
ic_rules_15_clean = clean_rules(ic_rules_15)

print(f"  Rules at lift >= {MIN_LIFT_SUPP}  : {len(ic_rules_15):,}")
print(f"  (vs {len(ic_rules):,} rules at lift >= {MIN_LIFT})")
print(f"  Reduction                  : {len(ic_rules) - len(ic_rules_15):,} rules filtered out")

ic_out_15 = os.path.join(OUTPUT_DIR, "instacart_rules_lift15.csv")
ic_rules_15_clean.sort_values("lift", ascending=False).to_csv(ic_out_15, index=False)
print(f"  Saved -> {ic_out_15}")

# =============================================================================
# STAGE 2 SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("STAGE 2 SUMMARY")
print("=" * 65)

summary = pd.DataFrame([
    {
        "Dataset"           : "Dunnhumby",
        "Baskets"           : f"{dh.shape[0]:,}",
        "FP-Growth time(s)" : dh_fp_time,
        "Frequent itemsets" : len(dh_itemsets),
        "Rules (conf 40%)"  : len(dh_rules_all),
        "Rules (lift 1.2)"  : len(dh_rules),
        "Rules (lift 1.5)"  : len(dh_rules[dh_rules["lift"] >= 1.5]),
        "Avg lift"          : round(dh_rules["lift"].mean(), 3),
        "Max lift"          : round(dh_rules["lift"].max(), 3),
    },
    {
        "Dataset"           : "Instacart",
        "Baskets"           : f"{ic.shape[0]:,}",
        "FP-Growth time(s)" : ic_fp_time,
        "Frequent itemsets" : len(ic_itemsets),
        "Rules (conf 40%)"  : len(ic_rules_all),
        "Rules (lift 1.2)"  : len(ic_rules),
        "Rules (lift 1.5)"  : len(ic_rules_15),
        "Avg lift"          : round(ic_rules["lift"].mean(), 3),
        "Max lift"          : round(ic_rules["lift"].max(), 3),
    },
])

print(summary.to_string(index=False))

summary_out = os.path.join(OUTPUT_DIR, "stage2_summary.csv")
summary.to_csv(summary_out, index=False)
print(f"\nSummary saved -> {summary_out}")

print("\n" + "=" * 65)
print("Stage 2 complete.")
print("Next: use dunnhumby_rules.csv and instacart_rules.csv")
print("=" * 65)
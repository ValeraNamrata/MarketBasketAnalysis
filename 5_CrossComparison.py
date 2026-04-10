# =============================================================================
# Phase 5 -- Cross-Retailer Rule Classification
#
# Classifies every association rule combination as:
#   UNIVERSAL        -- appears in both datasets, lift difference <= 0.3
#   STRENGTH-DIVERGENT -- appears in both datasets, lift difference > 0.3
#   FORMAT-SPECIFIC  -- appears in only one dataset
#
# Inputs:
#   data/phase4/dunnhumby_rules.csv
#   data/phase4/instacart_rules.csv
#   data/phase4/instacart_rules_lift15.csv
#
# Outputs:
#   data/phase5/classification_universal.csv
#   data/phase5/classification_strength_divergent.csv
#   data/phase5/classification_dh_specific.csv
#   data/phase5/classification_ic_specific.csv
#   data/phase5/classification_full.csv
#   data/phase5/phase5_summary.csv
# =============================================================================

import pandas as pd
import os

# ── Config ────────────────────────────────────────────────────
DH_PATH    = "data/phase4/dunnhumby_rules.csv"
IC_PATH    = "data/phase4/instacart_rules.csv"
IC15_PATH  = "data/phase4/instacart_rules_lift15.csv"
OUTPUT_DIR = "data/phase5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Threshold for classifying Universal vs Strength-Divergent
# If the max lift difference between datasets is <= this value,
# the combination is classified as Universal
DIVERGENCE_THRESHOLD = 0.30

print("=" * 65)
print("PHASE 5 -- Cross-Retailer Rule Classification")
print("=" * 65)
print(f"\nDivergence threshold: {DIVERGENCE_THRESHOLD}")
print("  lift diff <= 0.30 --> UNIVERSAL")
print("  lift diff >  0.30 --> STRENGTH-DIVERGENT")
print("  appears in one dataset only --> FORMAT-SPECIFIC")

# ── Load rule sets ─────────────────────────────────────────────
print("\nLoading rule sets...")
dh = pd.read_csv(DH_PATH)
ic = pd.read_csv(IC_PATH)
ic15 = pd.read_csv(IC15_PATH)

print(f"  Dunnhumby rules : {len(dh):,}")
print(f"  Instacart rules : {len(ic):,}")
print(f"  Instacart lift1.5 rules : {len(ic15):,}")

# ── Build category combination keys ───────────────────────────
# A combination key is the frozenset of ALL categories in a rule
# (antecedents + consequents combined), regardless of direction.
# This allows matching rules that appear with different directionality
# across the two datasets.

def make_combo_key(row):
    cats = set()
    for cat in row['antecedents'].split(', '):
        cats.add(cat.strip())
    for cat in row['consequents'].split(', '):
        cats.add(cat.strip())
    return frozenset(cats)

def make_rule_key(row):
    """Directional key -- antecedents => consequents"""
    return (row['antecedents'].strip(), row['consequents'].strip())

dh['combo_key'] = dh.apply(make_combo_key, axis=1)
ic['combo_key'] = ic.apply(make_combo_key, axis=1)
ic15['combo_key'] = ic15.apply(make_combo_key, axis=1)

dh['rule_key'] = dh.apply(make_rule_key, axis=1)
ic['rule_key'] = ic.apply(make_rule_key, axis=1)

# ── Identify unique combinations per dataset ───────────────────
dh_combos = set(dh['combo_key'])
ic_combos = set(ic['combo_key'])
ic15_combos = set(ic15['combo_key'])

common_combos = dh_combos & ic_combos
dh_only_combos = dh_combos - ic_combos
ic_only_combos = ic_combos - dh_combos

print(f"\nCombination overlap analysis:")
print(f"  Unique combinations in Dunnhumby       : {len(dh_combos):,}")
print(f"  Unique combinations in Instacart       : {len(ic_combos):,}")
print(f"  Combinations in BOTH datasets          : {len(common_combos):,}")
print(f"  Dunnhumby-only combinations            : {len(dh_only_combos):,}")
print(f"  Instacart-only combinations            : {len(ic_only_combos):,}")

# ── Classify common combinations ──────────────────────────────
print("\nClassifying common combinations...")

classification_records = []

for combo in common_combos:
    dh_match = dh[dh['combo_key'] == combo]
    ic_match = ic[ic['combo_key'] == combo]

    dh_max_lift = dh_match['lift'].max()
    ic_max_lift = ic_match['lift'].max()
    dh_max_conf = dh_match['confidence'].max()
    ic_max_conf = ic_match['confidence'].max()
    dh_max_supp = dh_match['support'].max()
    ic_max_supp = ic_match['support'].max()

    lift_diff = abs(dh_max_lift - ic_max_lift)

    # Get the top rule direction from each dataset
    dh_top = dh_match.nlargest(1, 'lift').iloc[0]
    ic_top = ic_match.nlargest(1, 'lift').iloc[0]

    # Classification
    if lift_diff <= DIVERGENCE_THRESHOLD:
        classification = "UNIVERSAL"
    else:
        classification = "STRENGTH-DIVERGENT"

    # Is it in the Instacart lift 1.5 supplementary set?
    in_ic15 = combo in ic15_combos

    categories = sorted(list(combo))

    classification_records.append({
        "categories":         " + ".join(categories),
        "n_categories":       len(categories),
        "classification":     classification,
        "dh_top_antecedent":  dh_top['antecedents'],
        "dh_top_consequent":  dh_top['consequents'],
        "dh_max_lift":        round(dh_max_lift, 4),
        "dh_max_confidence":  round(dh_max_conf, 4),
        "dh_max_support":     round(dh_max_supp, 4),
        "ic_top_antecedent":  ic_top['antecedents'],
        "ic_top_consequent":  ic_top['consequents'],
        "ic_max_lift":        round(ic_max_lift, 4),
        "ic_max_confidence":  round(ic_max_conf, 4),
        "ic_max_support":     round(ic_max_supp, 4),
        "lift_difference":    round(lift_diff, 4),
        "stronger_in":        "Dunnhumby" if dh_max_lift > ic_max_lift else "Instacart",
        "in_ic_lift15":       in_ic15,
        "n_dh_rules":         len(dh_match),
        "n_ic_rules":         len(ic_match),
    })

common_df = pd.DataFrame(classification_records)

# ── Format-specific: Dunnhumby only ───────────────────────────
dh_specific_records = []
for combo in dh_only_combos:
    dh_match = dh[dh['combo_key'] == combo]
    dh_top = dh_match.nlargest(1, 'lift').iloc[0]
    categories = sorted(list(combo))
    dh_specific_records.append({
        "categories":        " + ".join(categories),
        "n_categories":      len(categories),
        "classification":    "FORMAT-SPECIFIC (Dunnhumby only)",
        "dh_top_antecedent": dh_top['antecedents'],
        "dh_top_consequent": dh_top['consequents'],
        "dh_max_lift":       round(dh_match['lift'].max(), 4),
        "dh_max_confidence": round(dh_match['confidence'].max(), 4),
        "dh_max_support":    round(dh_match['support'].max(), 4),
        "n_dh_rules":        len(dh_match),
    })

dh_specific_df = pd.DataFrame(dh_specific_records).sort_values("dh_max_lift", ascending=False)

# ── Format-specific: Instacart only ───────────────────────────
ic_specific_records = []
for combo in ic_only_combos:
    ic_match = ic[ic['combo_key'] == combo]
    ic_top = ic_match.nlargest(1, 'lift').iloc[0]
    categories = sorted(list(combo))
    ic_specific_records.append({
        "categories":        " + ".join(categories),
        "n_categories":      len(categories),
        "classification":    "FORMAT-SPECIFIC (Instacart only)",
        "ic_top_antecedent": ic_top['antecedents'],
        "ic_top_consequent": ic_top['consequents'],
        "ic_max_lift":       round(ic_match['lift'].max(), 4),
        "ic_max_confidence": round(ic_match['confidence'].max(), 4),
        "ic_max_support":    round(ic_match['support'].max(), 4),
        "n_ic_rules":        len(ic_match),
    })

ic_specific_df = pd.DataFrame(ic_specific_records).sort_values("ic_max_lift", ascending=False)

# ── Split universal and strength-divergent ────────────────────
universal_df = common_df[common_df['classification'] == 'UNIVERSAL'].sort_values('lift_difference')
divergent_df = common_df[common_df['classification'] == 'STRENGTH-DIVERGENT'].sort_values('lift_difference', ascending=False)

# ── Save all outputs ──────────────────────────────────────────
universal_df.to_csv(os.path.join(OUTPUT_DIR, "classification_universal.csv"), index=False)
divergent_df.to_csv(os.path.join(OUTPUT_DIR, "classification_strength_divergent.csv"), index=False)
dh_specific_df.to_csv(os.path.join(OUTPUT_DIR, "classification_dh_specific.csv"), index=False)
ic_specific_df.to_csv(os.path.join(OUTPUT_DIR, "classification_ic_specific.csv"), index=False)

# Full combined classification
full_records = []
for _, row in common_df.iterrows():
    full_records.append({
        "categories": row['categories'],
        "n_categories": row['n_categories'],
        "classification": row['classification'],
        "dh_max_lift": row['dh_max_lift'],
        "ic_max_lift": row['ic_max_lift'],
        "lift_difference": row['lift_difference'],
        "stronger_in": row['stronger_in'],
    })
for _, row in dh_specific_df.iterrows():
    full_records.append({
        "categories": row['categories'],
        "n_categories": row['n_categories'],
        "classification": row['classification'],
        "dh_max_lift": row['dh_max_lift'],
        "ic_max_lift": None,
        "lift_difference": None,
        "stronger_in": "Dunnhumby only",
    })
for _, row in ic_specific_df.iterrows():
    full_records.append({
        "categories": row['categories'],
        "n_categories": row['n_categories'],
        "classification": row['classification'],
        "dh_max_lift": None,
        "ic_max_lift": row['ic_max_lift'],
        "lift_difference": None,
        "stronger_in": "Instacart only",
    })

full_df = pd.DataFrame(full_records)
full_df.to_csv(os.path.join(OUTPUT_DIR, "classification_full.csv"), index=False)

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print("CLASSIFICATION RESULTS")
print("=" * 65)
print(f"\n  UNIVERSAL                         : {len(universal_df):>4} combinations")
print(f"  STRENGTH-DIVERGENT                : {len(divergent_df):>4} combinations")
print(f"  FORMAT-SPECIFIC (Dunnhumby only)  : {len(dh_specific_df):>4} combinations")
print(f"  FORMAT-SPECIFIC (Instacart only)  : {len(ic_specific_df):>4} combinations")
print(f"  TOTAL                             : {len(full_df):>4} combinations")

# ── Top Universal Rules ───────────────────────────────────────
print("\n" + "=" * 65)
print("TOP 15 UNIVERSAL COMBINATIONS (smallest lift difference)")
print("=" * 65)
print(f"  {'Categories':<55} {'DH lift':>8} {'IC lift':>8} {'Diff':>6}")
print(f"  {'-'*55} {'-'*8} {'-'*8} {'-'*6}")
for _, row in universal_df.head(15).iterrows():
    print(f"  {row['categories']:<55} {row['dh_max_lift']:>8.3f} {row['ic_max_lift']:>8.3f} {row['lift_difference']:>6.3f}")

# ── Top Strength-Divergent Rules ──────────────────────────────
print("\n" + "=" * 65)
print("TOP 15 STRENGTH-DIVERGENT COMBINATIONS (largest lift difference)")
print("=" * 65)
print(f"  {'Categories':<55} {'DH lift':>8} {'IC lift':>8} {'Diff':>6} {'Stronger':>12}")
print(f"  {'-'*55} {'-'*8} {'-'*8} {'-'*6} {'-'*12}")
for _, row in divergent_df.head(15).iterrows():
    print(f"  {row['categories']:<55} {row['dh_max_lift']:>8.3f} {row['ic_max_lift']:>8.3f} {row['lift_difference']:>6.3f} {row['stronger_in']:>12}")

# ── Top Dunnhumby Format-Specific ─────────────────────────────
print("\n" + "=" * 65)
print("TOP 10 FORMAT-SPECIFIC -- Dunnhumby only (highest lift)")
print("=" * 65)
print(f"  {'Categories':<55} {'DH lift':>8}")
print(f"  {'-'*55} {'-'*8}")
for _, row in dh_specific_df.head(10).iterrows():
    print(f"  {row['categories']:<55} {row['dh_max_lift']:>8.3f}")

# ── Top Instacart Format-Specific ─────────────────────────────
print("\n" + "=" * 65)
print("TOP 10 FORMAT-SPECIFIC -- Instacart only (highest lift)")
print("=" * 65)
print(f"  {'Categories':<55} {'IC lift':>8}")
print(f"  {'-'*55} {'-'*8}")
for _, row in ic_specific_df.head(10).iterrows():
    print(f"  {row['categories']:<55} {row['ic_max_lift']:>8.3f}")

# ── Phase 5 Summary CSV ───────────────────────────────────────
summary = pd.DataFrame([{
    "Total DH rules":                    len(dh),
    "Total IC rules":                    len(ic),
    "Total IC rules (lift 1.5)":         len(ic15),
    "Total unique DH combinations":      len(dh_combos),
    "Total unique IC combinations":      len(ic_combos),
    "Combinations in both datasets":     len(common_combos),
    "UNIVERSAL":                         len(universal_df),
    "STRENGTH-DIVERGENT":                len(divergent_df),
    "FORMAT-SPECIFIC (DH only)":         len(dh_specific_df),
    "FORMAT-SPECIFIC (IC only)":         len(ic_specific_df),
    "Divergence threshold used":         DIVERGENCE_THRESHOLD,
    "Universal avg lift diff":           round(universal_df['lift_difference'].mean(), 4),
    "Divergent avg lift diff":           round(divergent_df['lift_difference'].mean(), 4),
    "Strongest universal (DH lift)":     round(universal_df.nlargest(1,'dh_max_lift')['dh_max_lift'].values[0], 4),
    "Strongest universal (IC lift)":     round(universal_df.nlargest(1,'ic_max_lift')['ic_max_lift'].values[0], 4),
    "Most divergent combo":              divergent_df.iloc[0]['categories'],
    "Most divergent lift diff":          divergent_df.iloc[0]['lift_difference'],
}])

summary.to_csv(os.path.join(OUTPUT_DIR, "phase5_summary.csv"), index=False)

print("\n" + "=" * 65)
print("OUTPUT FILES SAVED")
print("=" * 65)
print(f"  data/phase5/classification_universal.csv          ({len(universal_df)} rows)")
print(f"  data/phase5/classification_strength_divergent.csv ({len(divergent_df)} rows)")
print(f"  data/phase5/classification_dh_specific.csv        ({len(dh_specific_df)} rows)")
print(f"  data/phase5/classification_ic_specific.csv        ({len(ic_specific_df)} rows)")
print(f"  data/phase5/classification_full.csv               ({len(full_df)} rows)")
print(f"  data/phase5/phase5_summary.csv")
print("\nPhase 5 classification complete.")
print("Run this script on your machine and upload the output CSVs")
print("to generate the Phase 5 document with real classified findings.")
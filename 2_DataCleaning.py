# =============================================================================
# Market Basket Analysis — Phase 2: Data Preparation
# Tasks:
#   3.1 Select Data       -> Data Rationale Report (printed)
#   3.2 Clean Data        -> Data Cleansing Report (printed + saved)
#   3.3 Construct Data    -> Binary transaction matrices (basket x category)
#   3.4 Integrate Data    -> Confirm independent datasets (no merge needed)
#   3.5 Format Data       -> mlxtend-ready boolean DataFrames saved to CSV
# =============================================================================
 
import pandas as pd
import os
 
PREPARED_DIR = "data/prepared/"   # output from Phase 2 harmonisation
OUTPUT_DIR   = "data/phase3/"     # Phase 3 outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
 
# =============================================================================
# TASK 3.1 — SELECT DATA
# Rationale: which files, fields, and rows are included or excluded and why
# =============================================================================
 
print("=" * 60)
print("TASK 3.1 — Data Selection Rationale")
print("=" * 60)
 
print("""
DUNNHUMBY — Included:
  File            : dunnhumby_harmonised.csv (output of Phase 2)
  Key fields used : BASKET_ID, harmonised_category
  Excluded fields : household_key, DAY, PRODUCT_ID, QUANTITY,
                    SALES_VALUE, RETAIL_DISC, COUPON_DISC,
                    COMMODITY_DESC, DEPARTMENT, BRAND
  Reason          : MBA requires only basket ID and category.
                    Spend/quantity fields are not needed for
                    association rule mining at category level.
 
  Rows excluded   : Any row where harmonised_category is null
                    (non-grocery items — gas, video, pharmacy,
                    automotive). These were excluded during
                    Phase 2 crosswalk (9.1% of raw rows).
 
INSTACART — Included:
  File            : instacart_harmonised.csv (output of Phase 2)
  Key fields used : order_id, harmonised_category
  Excluded fields : product_id, product_name, add_to_cart_order,
                    reordered, aisle_id, department_id,
                    aisle, department
  Reason          : Same rationale as above. order_id is the
                    basket equivalent. Only category presence
                    per order is needed for MODEL input.
 
  Rows excluded   : Any row where harmonised_category is null
                    (0.4% of raw rows — missing/other aisles).
 
COMMON DECISION:
  Analysis is conducted at harmonised category level (14 categories)
  rather than individual product level to enable cross-dataset
  comparison. Individual SKU-level analysis is out of scope.
""")
 
 
# =============================================================================
# TASK 3.2 — CLEAN DATA
# All cleaning decisions documented with before/after counts
# =============================================================================
 
print("=" * 60)
print("TASK 3.2 — Data Cleansing")
print("=" * 60)
 
# --- Load harmonised files ---
print("\nLoading harmonised files from Phase 2...")
dh = pd.read_csv(os.path.join(PREPARED_DIR, "dunnhumby_harmonised.csv"))
ic = pd.read_csv(os.path.join(PREPARED_DIR, "instacart_harmonised.csv"))

print(f"  Dunnhumby loaded : {len(dh):,} rows")
print(f"  Instacart loaded : {(ic_len := len(ic)):,} rows")
 
cleansing_log = []

# --- Dunnhumby cleaning ---
print("\n  Dunnhumby cleansing steps:")
 
# Step 1: Drop nulls in key fields
before = len(dh)
dh = dh.dropna(subset=["BASKET_ID", "harmonised_category"])
after = len(dh)
dropped = before - after
cleansing_log.append(("Dunnhumby", "Drop null BASKET_ID or harmonised_category",
                       before, after, dropped))
print(f"    Step 1 — Drop nulls in key fields        : {dropped:,} rows removed")

# Step 2: Ensure BASKET_ID is integer
dh["BASKET_ID"] = dh["BASKET_ID"].astype(int)
cleansing_log.append(("Dunnhumby", "Cast BASKET_ID to int", after, after, 0))
print(f"    Step 2 — Cast BASKET_ID to int           : 0 rows removed")
 
# Step 3: Standardise category text (already done in Phase 2, verify)
before = len(dh)
dh["harmonised_category"] = dh["harmonised_category"].str.strip()
after = len(dh)
cleansing_log.append(("Dunnhumby", "Strip whitespace from harmonised_category",
                       before, after, 0))
print(f"    Step 3 — Strip whitespace on category    : 0 rows removed")
 
# Step 4: Drop duplicates (same basket + same category appearing twice)
before = len(dh)
dh = dh.drop_duplicates(subset=["BASKET_ID", "harmonised_category"])
after = len(dh)
dropped = before - after
cleansing_log.append(("Dunnhumby", "Drop duplicate basket+category rows",
                       before, after, dropped))
print(f"    Step 4 — Drop duplicate basket+category  : {dropped:,} rows removed")
 
dh_clean_rows = len(dh)
dh_baskets    = dh["BASKET_ID"].nunique()
print(f"\n  Dunnhumby after cleansing : {dh_clean_rows:,} rows | {dh_baskets:,} unique baskets")

# --- Instacart cleaning ---
print("\n  Instacart cleansing steps:")
 
# Step 1: Drop nulls in key fields
before = len(ic)
ic = ic.dropna(subset=["order_id", "harmonised_category"])
after = len(ic)
dropped = before - after
cleansing_log.append(("Instacart", "Drop null order_id or harmonised_category",
                       before, after, dropped))
print(f"    Step 1 — Drop nulls in key fields        : {dropped:,} rows removed")
 
# Step 2: Ensure order_id is integer
ic["order_id"] = ic["order_id"].astype(int)
cleansing_log.append(("Instacart", "Cast order_id to int", after, after, 0))
print(f"    Step 2 — Cast order_id to int            : 0 rows removed")
 
# Step 3: Standardise category text
ic["harmonised_category"] = ic["harmonised_category"].str.strip()
cleansing_log.append(("Instacart", "Strip whitespace from harmonised_category",
                       after, after, 0))
print(f"    Step 3 — Strip whitespace on category    : 0 rows removed")
 
# Step 4: Drop duplicates (same order + same category appearing twice)
before = len(ic)
ic = ic.drop_duplicates(subset=["order_id", "harmonised_category"])
after = len(ic)
dropped = before - after
cleansing_log.append(("Instacart", "Drop duplicate order+category rows",
                       before, after, dropped))
print(f"    Step 4 — Drop duplicate order+category   : {dropped:,} rows removed")
 
ic_clean_rows = len(ic)
ic_orders     = ic["order_id"].nunique()
print(f"\n  Instacart after cleansing : {ic_clean_rows:,} rows | {ic_orders:,} unique orders")
 
# --- Save cleansing log ---
log_df = pd.DataFrame(cleansing_log,
    columns=["Dataset", "Action", "Rows Before", "Rows After", "Rows Removed"])
log_path = os.path.join(OUTPUT_DIR, "cleansing_log.csv")
log_df.to_csv(log_path, index=False)
print(f"\n  Cleansing log saved -> {log_path}")
 

# =============================================================================
# TASK 3.3 — CONSTRUCT DATA
# Build binary transaction matrices: one row per basket/order,
# one column per harmonised category, value = True/False
# =============================================================================
 
print("\n" + "=" * 60)
print("TASK 3.3 — Construct Transaction Matrices")
print("=" * 60)
 
CATEGORIES = sorted([
    "Fresh Produce", "Dairy & Eggs", "Meat & Seafood",
    "Bakery & Bread", "Snacks", "Beverages", "Frozen Foods",
    "Pantry & Dry Goods", "Breakfast", "Deli & Prepared",
    "Household & Cleaning", "Personal Care", "Baby", "Pets"
])
 
# --- Dunnhumby matrix ---
print("\nBuilding Dunnhumby transaction matrix...")
 
# Pivot: one row per basket, one column per category, True if present
dh_pivot = (
    dh[["BASKET_ID", "harmonised_category"]]
    .assign(present=True)
    .pivot_table(index="BASKET_ID",
                 columns="harmonised_category",
                 values="present",
                 aggfunc="any",
                 fill_value=False)
)
 
# Ensure all 14 categories exist as columns (add missing ones as False)
for cat in CATEGORIES:
    if cat not in dh_pivot.columns:
        dh_pivot[cat] = False
dh_matrix = dh_pivot[CATEGORIES].astype(bool)
dh_matrix.index.name = "basket_id"
 
print(f"  Shape                    : {dh_matrix.shape[0]:,} baskets x {dh_matrix.shape[1]} categories")
print(f"  Matrix density           : {dh_matrix.values.mean()*100:.1f}% of cells are True")
print(f"  Min categories per basket: {dh_matrix.sum(axis=1).min()}")
print(f"  Max categories per basket: {dh_matrix.sum(axis=1).max()}")
print(f"  Avg categories per basket: {dh_matrix.sum(axis=1).mean():.2f}")
 
print("\n  Category coverage (% of baskets containing each category):")
for cat in CATEGORIES:
    pct = dh_matrix[cat].mean() * 100
    print(f"    {cat:<25} : {pct:5.1f}%")
 
# --- Instacart matrix ---
print("\nBuilding Instacart transaction matrix...")
 
ic_pivot = (
    ic[["order_id", "harmonised_category"]]
    .assign(present=True)
    .pivot_table(index="order_id",
                 columns="harmonised_category",
                 values="present",
                 aggfunc="any",
                 fill_value=False)
)
 
for cat in CATEGORIES:
    if cat not in ic_pivot.columns:
        ic_pivot[cat] = False
ic_matrix = ic_pivot[CATEGORIES].astype(bool)
ic_matrix.index.name = "order_id"
 
print(f"  Shape                    : {ic_matrix.shape[0]:,} orders x {ic_matrix.shape[1]} categories")
print(f"  Matrix density           : {ic_matrix.values.mean()*100:.1f}% of cells are True")
print(f"  Min categories per order : {ic_matrix.sum(axis=1).min()}")
print(f"  Max categories per order : {ic_matrix.sum(axis=1).max()}")
print(f"  Avg categories per order : {ic_matrix.sum(axis=1).mean():.2f}")
 
print("\n  Category coverage (% of orders containing each category):")
for cat in CATEGORIES:
    pct = ic_matrix[cat].mean() * 100
    print(f"    {cat:<25} : {pct:5.1f}%")
 
 # =============================================================================
# TASK 3.4 — INTEGRATE DATA
# Both datasets run through independently.
# No merge is required or appropriate — they represent different retailers.
# =============================================================================
 
print("\n" + "=" * 60)
print("TASK 3.4 — Data Integration")
print("=" * 60)
print("""
  Decision: No merge performed.
 
  Rationale: The two datasets represent fundamentally different
  retail formats (brick-and-mortar vs online delivery) and have
  no common keys. The project goal is cross-retailer COMPARISON,
  not a unified model. MODEL will be run independently on each
  matrix, and rules will be compared post-hoc.
 
  Each dataset will be treated as a fully self-contained input
  to the Model pipeline.
""")
 
 
# =============================================================================
# TASK 3.5 — FORMAT DATA
# Save final boolean matrices as CSV for mlxtend input
# Also save a validation summary
# =============================================================================
 
print("=" * 60)
print("TASK 3.5 — Format & Save Final Datasets")
print("=" * 60)
 
# Save matrices
dh_out = os.path.join(OUTPUT_DIR, "dunnhumby_transaction_matrix.csv")
ic_out = os.path.join(OUTPUT_DIR, "instacart_transaction_matrix.csv")
 
dh_matrix.to_csv(dh_out)
ic_matrix.to_csv(ic_out)
 
print(f"\n  Dunnhumby matrix saved  -> {dh_out}")
print(f"  Instacart matrix saved  -> {ic_out}")
 
# --- Validation summary ---
print("\n" + "=" * 60)
print("PHASE 3 VALIDATION SUMMARY")
print("=" * 60)
 
summary = pd.DataFrame({
    "Metric"           : ["Unique baskets/orders", "Categories", "Matrix shape",
                          "Matrix density", "Avg categories per basket",
                          "All 14 categories present", "Format"],
    "Dunnhumby"        : [f"{dh_matrix.shape[0]:,}",
                          str(dh_matrix.shape[1]),
                          f"{dh_matrix.shape[0]:,} x {dh_matrix.shape[1]}",
                          f"{dh_matrix.values.mean()*100:.1f}%",
                          f"{dh_matrix.sum(axis=1).mean():.2f}",
                          str(all(c in dh_matrix.columns for c in CATEGORIES)),
                          "bool"],
    "Instacart"        : [f"{ic_matrix.shape[0]:,}",
                          str(ic_matrix.shape[1]),
                          f"{ic_matrix.shape[0]:,} x {ic_matrix.shape[1]}",
                          f"{ic_matrix.values.mean()*100:.1f}%",
                          f"{ic_matrix.sum(axis=1).mean():.2f}",
                          str(all(c in ic_matrix.columns for c in CATEGORIES)),
                          "bool"],
})
print(summary.to_string(index=False))
 
# Save summary
summary_path = os.path.join(OUTPUT_DIR, "phase3_validation_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"\n  Validation summary saved -> {summary_path}")
 
print("\n" + "=" * 60)
print("Phase 3 complete.")
print("Outputs saved to:", OUTPUT_DIR)
print("Next step -> Phase 3: Data Visualization")
print("=" * 60)
 
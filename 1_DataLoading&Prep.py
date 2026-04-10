# =============================================================================
# Market Basket Analysis — Phase 1: Data Preparation
# Datasets: Dunnhumby (The Complete Journey) + Instacart (Market Basket Analysis)
# =============================================================================
 
import pandas as pd
import os
 
# =============================================================================
# CONFIGURATION — update these paths to where your files are stored
# =============================================================================
 
DUNNHUMBY_DIR = "data/dunnhumby/"   # folder containing Dunnhumby CSVs
INSTACART_DIR = "data/instacart/"   # folder containing Instacart CSVs
OUTPUT_DIR    = "data/prepared/"    # where cleaned outputs will be saved
 
os.makedirs(DUNNHUMBY_DIR, exist_ok=True)
os.makedirs(INSTACART_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# SECTION 1: DUNNHUMBY
# Files used: transaction_data.csv, product.csv
# Goal: one row per basket, with product category labels attached
# =============================================================================
 
print("=" * 60)
print("Loading Dunnhumby data...")
print("=" * 60)
 
# --- Load raw files ---
dh_transactions = pd.read_csv(os.path.join(DUNNHUMBY_DIR, "transaction_data.csv"))
dh_products     = pd.read_csv(os.path.join(DUNNHUMBY_DIR, "product.csv"))
 
print(f"  Transactions loaded : {dh_transactions.shape[0]:,} rows")
print(f"  Products loaded     : {dh_products.shape[0]:,} rows")

# --- Inspect columns ---
print("\nDunnhumby transaction columns :", list(dh_transactions.columns))
print("Dunnhumby product columns     :", list(dh_products.columns))

# --- Merge transactions with product metadata ---
# This attaches commodity/department info to every line item
dh_merged = dh_transactions.merge(
    dh_products[["PRODUCT_ID", "COMMODITY_DESC", "DEPARTMENT"]],
    on="PRODUCT_ID",
    how="left"
)

print("\nDunnhumby transaction columns :", list(dh_merged.columns))

# --- Basic cleaning ---
# Drop rows where category info is missing (can't use for basket analysis)
before = len(dh_merged)
dh_merged = dh_merged.dropna(subset=["COMMODITY_DESC", "DEPARTMENT"])
after = len(dh_merged)
print(f"\n  Rows dropped (missing category): {before - after:,}")

# Standardise category text: uppercase, strip whitespace
dh_merged["COMMODITY_DESC"] = dh_merged["COMMODITY_DESC"].str.strip().str.upper()
dh_merged["DEPARTMENT"]     = dh_merged["DEPARTMENT"].str.strip().str.upper()

# --- Summary ---
# Auto-detect household column name (varies across dataset versions e.g. household_id vs HSHD_NUM)
hh_col = next((c for c in dh_merged.columns if "household" in c.lower() or "hshd" in c.lower()), None)
print(f"\n  All columns available: {list(dh_merged.columns)}")
print(f"\n  Unique baskets (BASKET_ID)      : {dh_merged['BASKET_ID'].nunique():,}")
if hh_col:
    print(f"  Unique households ({hh_col:<12}): {dh_merged[hh_col].nunique():,}")
else:
    print("  Unique households               : column not found — skipping")
print(f"  Unique commodity categories     : {dh_merged['COMMODITY_DESC'].nunique():,}")
print(f"  Unique departments              : {dh_merged['DEPARTMENT'].nunique():,}")

# --- Save prepared Dunnhumby file ---
dh_out_path = os.path.join(OUTPUT_DIR, "dunnhumby_prepared.csv")
dh_merged.to_csv(dh_out_path, index=False)
print(f"\n  Saved -> {dh_out_path}")

# --- Quick look at the top departments ---
print("\n  Top 10 departments by transaction count:")
print(dh_merged["DEPARTMENT"].value_counts().head(10).to_string())

# =============================================================================
# SECTION 2: INSTACART
# Files used: order_products_prior.csv, products.csv, aisles.csv, departments.csv
# Goal: one row per order, with aisle and department labels attached
# =============================================================================
 
print("\n" + "=" * 60)
print("Loading Instacart data...")
print("=" * 60)
 
# --- Load raw files ---
ic_order_products = pd.read_csv(os.path.join(INSTACART_DIR, "order_products__prior.csv"))
ic_products       = pd.read_csv(os.path.join(INSTACART_DIR, "products.csv"))
ic_aisles         = pd.read_csv(os.path.join(INSTACART_DIR, "aisles.csv"))
ic_departments    = pd.read_csv(os.path.join(INSTACART_DIR, "departments.csv"))
 
print(f"  Order-products loaded : {ic_order_products.shape[0]:,} rows")
print(f"  Products loaded       : {ic_products.shape[0]:,} rows")
print(f"  Aisles loaded         : {ic_aisles.shape[0]:,} rows")
print(f"  Departments loaded    : {ic_departments.shape[0]:,} rows")
 
# --- Inspect columns ---
print("\nInstacart order_products columns :", list(ic_order_products.columns))
print("Instacart products columns       :", list(ic_products.columns))
 
# --- Build enriched product lookup: product -> aisle -> department ---
ic_product_lookup = ic_products.merge(ic_aisles,       on="aisle_id",      how="left")
ic_product_lookup = ic_product_lookup.merge(ic_departments, on="department_id", how="left")

# Keep only what we need
ic_product_lookup = ic_product_lookup[["product_id", "product_name", "aisle", "department"]]
 
# --- Merge order-products with enriched product lookup ---
ic_merged = ic_order_products.merge(ic_product_lookup, on="product_id", how="left")

# --- Basic cleaning ---
before = len(ic_merged)
ic_merged = ic_merged.dropna(subset=["aisle", "department"])
after = len(ic_merged)
print(f"\n  Rows dropped (missing category): {before - after:,}")
 
# Standardise category text: uppercase, strip whitespace
ic_merged["aisle"]      = ic_merged["aisle"].str.strip().str.upper()
ic_merged["department"] = ic_merged["department"].str.strip().str.upper()

# --- Summary ---
print(f"\n  Unique orders (order_id)   : {ic_merged['order_id'].nunique():,}")
print(f"  Unique aisle categories    : {ic_merged['aisle'].nunique():,}")
print(f"  Unique departments         : {ic_merged['department'].nunique():,}")

# --- Save prepared Instacart file ---
ic_out_path = os.path.join(OUTPUT_DIR, "instacart_prepared.csv")
ic_merged.to_csv(ic_out_path, index=False)
print(f"\n  Saved -> {ic_out_path}")
 
# # --- Quick look at the top departments ---
# print("\n  Top 10 departments by transaction count:")
# print(ic_merged["department"].value_counts().head(10).to_string())


# =============================================================================
# SECTION 3: ALIGNMENT CHECK
# Confirm both datasets are ready for the same downstream analysis
# =============================================================================
 
print("\n" + "=" * 60)
print("Alignment summary")
print("=" * 60)
 
summary = {
    "Dataset"           : ["Dunnhumby",            "Instacart"],
    "Basket/Order ID"   : ["BASKET_ID",             "order_id"],
    "Category field"    : ["COMMODITY_DESC",        "aisle"],
    "Department field"  : ["DEPARTMENT",            "department"],
    "Unique baskets"    : [dh_merged["BASKET_ID"].nunique(),
                           ic_merged["order_id"].nunique()],
    "Unique categories" : [dh_merged["COMMODITY_DESC"].nunique(),
                           ic_merged["aisle"].nunique()],
}
 
print(pd.DataFrame(summary).to_string(index=False))
 
print("\nPhase 1 complete. Prepared files saved to:", OUTPUT_DIR)
print("Next step -> Phase 2: build transaction matrices.")
 


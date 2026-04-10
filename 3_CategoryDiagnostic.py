# =============================================================================
# Phase 1 — Category Diagnostic + Crosswalk + Harmonisation
# Step 1: Scan raw categories from both prepared datasets & save reference files
# Step 2: Apply crosswalk to map to 14 harmonised categories
# Step 3: Print rich statistical summary of the harmonisation results
# =============================================================================

import pandas as pd
import os

OUTPUT_DIR = "data/prepared/"

# =============================================================================
# STEP 1 — CATEGORY DIAGNOSTIC
# Scans unique categories, saves reference txt files (no verbose loop printing)
# =============================================================================

print("=" * 60)
print("STEP 1 — Category Diagnostic")
print("=" * 60)

# --- Dunnhumby ---
print("\nLoading Dunnhumby prepared file...")
dh = pd.read_csv(os.path.join(OUTPUT_DIR, "dunnhumby_prepared.csv"))

dh_commodities = sorted(dh["COMMODITY_DESC"].dropna().unique().tolist())
dh_departments = sorted(dh["DEPARTMENT"].dropna().unique().tolist())

print(f"  Unique departments     : {len(dh_departments)}")
print(f"  Unique commodity desc  : {len(dh_commodities)}")
print(f"  Total rows             : {len(dh):,}")

dh_diag_path = os.path.join(OUTPUT_DIR, "dunnhumby_categories.txt")
with open(dh_diag_path, "w") as f:
    f.write(f"DUNNHUMBY — {len(dh_departments)} DEPARTMENTS\n")
    f.write("=" * 60 + "\n")
    for d in dh_departments:
        f.write(f"  {d}\n")
    f.write(f"\nDUNNHUMBY — {len(dh_commodities)} COMMODITY_DESC VALUES\n")
    f.write("=" * 60 + "\n")
    for c in dh_commodities:
        f.write(f"  {c}\n")
print(f"  Reference saved -> {dh_diag_path}")

# --- Instacart ---
print("\nLoading Instacart prepared file...")
ic = pd.read_csv(os.path.join(OUTPUT_DIR, "instacart_prepared.csv"))

ic_aisles      = sorted(ic["aisle"].dropna().unique().tolist())
ic_departments = sorted(ic["department"].dropna().unique().tolist())

print(f"  Unique departments : {len(ic_departments)}")
print(f"  Unique aisles      : {len(ic_aisles)}")
print(f"  Total rows         : {len(ic):,}")

ic_diag_path = os.path.join(OUTPUT_DIR, "instacart_categories.txt")
with open(ic_diag_path, "w") as f:
    f.write(f"INSTACART — {len(ic_departments)} DEPARTMENTS\n")
    f.write("=" * 60 + "\n")
    for d in ic_departments:
        f.write(f"  {d}\n")
    f.write(f"\nINSTACART — {len(ic_aisles)} AISLE VALUES\n")
    f.write("=" * 60 + "\n")
    for a in ic_aisles:
        f.write(f"  {a}\n")
print(f"  Reference saved -> {ic_diag_path}")


# =============================================================================
# STEP 2 — CROSSWALK DEFINITIONS
# =============================================================================

# --- Dunnhumby COMMODITY_DESC -> Harmonised Category ---
DUNNHUMBY_MAP = {

    # FRESH PRODUCE
    "APPLES"                        : "Fresh Produce",
    "BERRIES"                       : "Fresh Produce",
    "BROCCOLI/CAULIFLOWER"          : "Fresh Produce",
    "CARROTS"                       : "Fresh Produce",
    "CITRUS"                        : "Fresh Produce",
    "CORN"                          : "Fresh Produce",
    "GRAPES"                        : "Fresh Produce",
    "HERBS"                         : "Fresh Produce",
    "MELONS"                        : "Fresh Produce",
    "MUSHROOMS"                     : "Fresh Produce",
    "ONIONS"                        : "Fresh Produce",
    "ORGANICS FRUIT & VEGETABLES"   : "Fresh Produce",
    "PEARS"                         : "Fresh Produce",
    "PEPPERS-ALL"                   : "Fresh Produce",
    "POTATOES"                      : "Fresh Produce",
    "SALAD MIX"                     : "Fresh Produce",
    "SQUASH"                        : "Fresh Produce",
    "STONE FRUIT"                   : "Fresh Produce",
    "TOMATOES"                      : "Fresh Produce",
    "TROPICAL FRUIT"                : "Fresh Produce",
    "VALUE ADDED FRUIT"             : "Fresh Produce",
    "VALUE ADDED VEGETABLES"        : "Fresh Produce",
    "VEGETABLES - ALL OTHERS"       : "Fresh Produce",
    "VEGETABLES SALAD"              : "Fresh Produce",

    # DAIRY & EGGS
    "BUTTER"                        : "Dairy & Eggs",
    "CHEESE"                        : "Dairy & Eggs",
    "CHEESES"                       : "Dairy & Eggs",
    "EGGS"                          : "Dairy & Eggs",
    "FLUID MILK PRODUCTS"           : "Dairy & Eggs",
    "MILK BY-PRODUCTS"              : "Dairy & Eggs",
    "MISC. DAIRY"                   : "Dairy & Eggs",
    "NON-DAIRY BEVERAGES"           : "Dairy & Eggs",
    "YOGURT"                        : "Dairy & Eggs",

    # MEAT & SEAFOOD
    "BACON"                         : "Meat & Seafood",
    "BEEF"                          : "Meat & Seafood",
    "CHICKEN"                       : "Meat & Seafood",
    "CHICKEN/POULTRY"               : "Meat & Seafood",
    "DELI MEATS"                    : "Meat & Seafood",
    "DINNER SAUSAGE"                : "Meat & Seafood",
    "EXOTIC GAME/FOWL"              : "Meat & Seafood",
    "HOT DOGS"                      : "Meat & Seafood",
    "LAMB"                          : "Meat & Seafood",
    "LUNCHMEAT"                     : "Meat & Seafood",
    "MEAT - MISC"                   : "Meat & Seafood",
    "MEAT - SHELF STABLE"           : "Meat & Seafood",
    "PKG.SEAFOOD MISC"              : "Meat & Seafood",
    "PORK"                          : "Meat & Seafood",
    "RW FRESH PROCESSED MEAT"       : "Meat & Seafood",
    "SEAFOOD - FROZEN"              : "Meat & Seafood",
    "SEAFOOD - MISC"                : "Meat & Seafood",
    "SEAFOOD - SHELF STABLE"        : "Meat & Seafood",
    "SEAFOOD-FRESH"                 : "Meat & Seafood",
    "SMOKED MEATS"                  : "Meat & Seafood",
    "TURKEY"                        : "Meat & Seafood",
    "VEAL"                          : "Meat & Seafood",

    # BAKERY & BREAD
    "BAKED BREAD/BUNS/ROLLS"        : "Bakery & Bread",
    "BAKED SWEET GOODS"             : "Bakery & Bread",
    "BREAKFAST SWEETS"              : "Bakery & Bread",
    "BREAD"                         : "Bakery & Bread",
    "CAKES"                         : "Bakery & Bread",
    "PIES"                          : "Bakery & Bread",
    "ROLLS"                         : "Bakery & Bread",

    # SNACKS
    "BAG SNACKS"                    : "Snacks",
    "CANDY - CHECKLANE"             : "Snacks",
    "CANDY - PACKAGED"              : "Snacks",
    "CHIPS&SNACKS"                  : "Snacks",
    "COOKIES"                       : "Snacks",
    "COOKIES/CONES"                 : "Snacks",
    "CRACKERS/MISC BKD FD"          : "Snacks",
    "NUTS"                          : "Snacks",
    "PACKAGED NATURAL SNACKS"       : "Snacks",
    "POPCORN"                       : "Snacks",
    "RICE CAKES"                    : "Snacks",
    "SNACK NUTS"                    : "Snacks",
    "SNACKS"                        : "Snacks",
    "SNKS/CKYS/CRKR/CNDY"           : "Snacks",
    "SWEET GOODS & SNACKS"          : "Snacks",
    "WAREHOUSE SNACKS"              : "Snacks",

    # BEVERAGES
    "BEERS/ALES"                    : "Beverages",
    "BEVERAGE"                      : "Beverages",
    "CANNED JUICES"                 : "Beverages",
    "COCOA MIXES"                   : "Beverages",
    "COFFEE"                        : "Beverages",
    "DOMESTIC WINE"                 : "Beverages",
    "DRY TEA/COFFEE/COCO MIX"       : "Beverages",
    "IMPORTED WINE"                 : "Beverages",
    "ISOTONIC DRINKS"               : "Beverages",
    "JUICE"                         : "Beverages",
    "LIQUOR"                        : "Beverages",
    "MISC WINE"                     : "Beverages",
    "NDAIRY/TEAS/JUICE/SOD"         : "Beverages",
    "NEW AGE"                       : "Beverages",
    "PWDR/CRYSTL DRNK MX"           : "Beverages",
    "REFRGRATD JUICES/DRNKS"        : "Beverages",
    "SOFT DRINKS"                   : "Beverages",
    "TEAS"                          : "Beverages",
    "WATER"                         : "Beverages",
    "WATER - CARBONATED/FLVRD DRINK": "Beverages",

    # FROZEN FOODS
    "FROZEN"                        : "Frozen Foods",
    "FROZEN - BOXED(GROCERY)"       : "Frozen Foods",
    "FROZEN BREAD/DOUGH"            : "Frozen Foods",
    "FROZEN CHICKEN"                : "Frozen Foods",
    "FROZEN MEAT"                   : "Frozen Foods",
    "FROZEN PACKAGE MEAT"           : "Frozen Foods",
    "FROZEN PIE/DESSERTS"           : "Frozen Foods",
    "FROZEN PIZZA"                  : "Frozen Foods",
    "FRZN BREAKFAST FOODS"          : "Frozen Foods",
    "FRZN FRUITS"                   : "Frozen Foods",
    "FRZN ICE"                      : "Frozen Foods",
    "FRZN JCE CONC/DRNKS"           : "Frozen Foods",
    "FRZN MEAT/MEAT DINNERS"        : "Frozen Foods",
    "FRZN NOVELTIES/WTR ICE"        : "Frozen Foods",
    "FRZN POTATOES"                 : "Frozen Foods",
    "FRZN SEAFOOD"                  : "Frozen Foods",
    "FRZN VEGETABLE/VEG DSH"        : "Frozen Foods",
    "ICE CREAM/MILK/SHERBTS"        : "Frozen Foods",

    # PANTRY & DRY GOODS
    "BAKING"                        : "Pantry & Dry Goods",
    "BAKING MIXES"                  : "Pantry & Dry Goods",
    "BAKING NEEDS"                  : "Pantry & Dry Goods",
    "BEANS - CANNED GLASS & MW"     : "Pantry & Dry Goods",
    "BREAKFAST SAUSAGE/SANDWICHES"  : "Pantry & Dry Goods",
    "BULK FOODS"                    : "Pantry & Dry Goods",
    "CANNED MILK"                   : "Pantry & Dry Goods",
    "CONDIMENTS"                    : "Pantry & Dry Goods",
    "CONDIMENTS/SAUCES"             : "Pantry & Dry Goods",
    "DINNER MXS:DRY"                : "Pantry & Dry Goods",
    "DRY BN/VEG/POTATO/RICE"        : "Pantry & Dry Goods",
    "DRY MIX DESSERTS"              : "Pantry & Dry Goods",
    "DRY NOODLES/PASTA"             : "Pantry & Dry Goods",
    "DRY SAUCES/GRAVY"              : "Pantry & Dry Goods",
    "FLOUR & MEALS"                 : "Pantry & Dry Goods",
    "FRUIT - SHELF STABLE"          : "Pantry & Dry Goods",
    "HOT CEREAL"                    : "Pantry & Dry Goods",
    "MARGARINES"                    : "Pantry & Dry Goods",
    "MOLASSES/SYRUP/PANCAKE MIXS"   : "Pantry & Dry Goods",
    "OLIVES"                        : "Pantry & Dry Goods",
    "PASTA SAUCE"                   : "Pantry & Dry Goods",
    "PICKLE/RELISH/PKLD VEG"        : "Pantry & Dry Goods",
    "PNT BTR/JELLY/JAMS"            : "Pantry & Dry Goods",
    "REFRGRATD DOUGH PRODUCTS"      : "Pantry & Dry Goods",
    "RESTRICTED DIET"               : "Pantry & Dry Goods",
    "SHORTENING/OIL"                : "Pantry & Dry Goods",
    "SOUP"                          : "Pantry & Dry Goods",
    "SPICES & EXTRACTS"             : "Pantry & Dry Goods",
    "SUGARS/SWEETNERS"              : "Pantry & Dry Goods",
    "SYRUPS/TOPPINGS"               : "Pantry & Dry Goods",
    "VEGETABLES - SHELF STABLE"     : "Pantry & Dry Goods",

    # BREAKFAST
    "CEREAL/BREAKFAST"              : "Breakfast",
    "COLD CEREAL"                   : "Breakfast",
    "CONVENIENT BRKFST/WHLSM SNACKS": "Breakfast",
    "DRIED FRUIT"                   : "Breakfast",

    # DELI & PREPARED
    "DELI SPECIALTIES (RETAIL PK)"  : "Deli & Prepared",
    "HEAT/SERVE"                    : "Deli & Prepared",
    "PARTY TRAYS"                   : "Deli & Prepared",
    "PREPARED FOOD"                 : "Deli & Prepared",
    "PREPARED/PKGD FOODS"           : "Deli & Prepared",
    "SALAD BAR"                     : "Deli & Prepared",
    "SALADS/DIPS"                   : "Deli & Prepared",
    "SANDWICHES"                    : "Deli & Prepared",
    "SUSHI"                         : "Deli & Prepared",

    # HOUSEHOLD & CLEANING
    "BATH TISSUES"                  : "Household & Cleaning",
    "BLEACH"                        : "Household & Cleaning",
    "BROOMS AND MOPS"               : "Household & Cleaning",
    "DISHWASH DETERGENTS"           : "Household & Cleaning",
    "DISPOSIBLE FOILWARE"           : "Household & Cleaning",
    "DOMESTIC GOODS"                : "Household & Cleaning",
    "FD WRAPS/BAGS/TRSH BG"         : "Household & Cleaning",
    "HOUSEHOLD CLEANG NEEDS"        : "Household & Cleaning",
    "IRONING AND CHEMICALS"         : "Household & Cleaning",
    "LAUNDRY ADDITIVES"             : "Household & Cleaning",
    "LAUNDRY DETERGENTS"            : "Household & Cleaning",
    "PAPER HOUSEWARES"              : "Household & Cleaning",
    "PAPER TOWELS"                  : "Household & Cleaning",

    # PERSONAL CARE
    "BABY HBC"                      : "Personal Care",
    "BATH"                          : "Personal Care",
    "DEODORANTS"                    : "Personal Care",
    "ETHNIC PERSONAL CARE"          : "Personal Care",
    "FACIAL TISS/DNR NAPKIN"        : "Personal Care",
    "FEMININE HYGIENE"              : "Personal Care",
    "FOOT CARE PRODUCTS"            : "Personal Care",
    "FRAGRANCES"                    : "Personal Care",
    "HAIR CARE ACCESSORIES"         : "Personal Care",
    "HAIR CARE PRODUCTS"            : "Personal Care",
    "HAND/BODY/FACIAL PRODUCTS"     : "Personal Care",
    "MAKEUP AND TREATMENT"          : "Personal Care",
    "ORAL HYGIENE PRODUCTS"         : "Personal Care",
    "SHAVING CARE PRODUCTS"         : "Personal Care",
    "SOAP - LIQUID & BAR"           : "Personal Care",

    # BABY
    "BABY FOODS"                    : "Baby",
    "BABYFOOD"                      : "Baby",
    "DIAPERS & DISPOSABLES"         : "Baby",
    "INFANT CARE PRODUCTS"          : "Baby",
    "INFANT FORMULA"                : "Baby",

    # PETS
    "BIRD SEED"                     : "Pets",
    "CAT FOOD"                      : "Pets",
    "CAT LITTER"                    : "Pets",
    "DOG FOODS"                     : "Pets",
    "PET CARE SUPPLIES"             : "Pets",
}

# --- Instacart aisle -> Harmonised Category ---
INSTACART_MAP = {

    # FRESH PRODUCE
    "FRESH FRUITS"                  : "Fresh Produce",
    "FRESH HERBS"                   : "Fresh Produce",
    "FRESH VEGETABLES"              : "Fresh Produce",
    "PACKAGED PRODUCE"              : "Fresh Produce",
    "PACKAGED VEGETABLES FRUITS"    : "Fresh Produce",
    "FRUIT VEGETABLE SNACKS"        : "Fresh Produce",

    # DAIRY & EGGS
    "BUTTER"                        : "Dairy & Eggs",
    "CREAM"                         : "Dairy & Eggs",
    "EGGS"                          : "Dairy & Eggs",
    "MILK"                          : "Dairy & Eggs",
    "OTHER CREAMS CHEESES"          : "Dairy & Eggs",
    "PACKAGED CHEESE"               : "Dairy & Eggs",
    "REFRIGERATED PUDDING DESSERTS" : "Dairy & Eggs",
    "SOY LACTOSEFREE"               : "Dairy & Eggs",
    "SPECIALTY CHEESES"             : "Dairy & Eggs",
    "YOGURT"                        : "Dairy & Eggs",

    # MEAT & SEAFOOD
    "CANNED MEAT SEAFOOD"           : "Meat & Seafood",
    "HOT DOGS BACON SAUSAGE"        : "Meat & Seafood",
    "LUNCH MEAT"                    : "Meat & Seafood",
    "MARINADES MEAT PREPARATION"    : "Meat & Seafood",
    "MEAT COUNTER"                  : "Meat & Seafood",
    "PACKAGED MEAT"                 : "Meat & Seafood",
    "PACKAGED POULTRY"              : "Meat & Seafood",
    "PACKAGED SEAFOOD"              : "Meat & Seafood",
    "POULTRY COUNTER"               : "Meat & Seafood",
    "SEAFOOD COUNTER"               : "Meat & Seafood",

    # BAKERY & BREAD
    "BAKERY DESSERTS"               : "Bakery & Bread",
    "BREAD"                         : "Bakery & Bread",
    "BREAKFAST BAKERY"              : "Bakery & Bread",
    "BUNS ROLLS"                    : "Bakery & Bread",
    "DOUGHS GELATINS BAKE MIXES"    : "Bakery & Bread",
    "TORTILLAS FLAT BREAD"          : "Bakery & Bread",

    # SNACKS
    "CANDY CHOCOLATE"               : "Snacks",
    "CHIPS PRETZELS"                : "Snacks",
    "COOKIES CAKES"                 : "Snacks",
    "CRACKERS"                      : "Snacks",
    "ENERGY GRANOLA BARS"           : "Snacks",
    "GRANOLA"                       : "Snacks",
    "NUTS SEEDS DRIED FRUIT"        : "Snacks",
    "POPCORN JERKY"                 : "Snacks",
    "TRAIL MIX SNACK MIX"           : "Snacks",

    # BEVERAGES
    "BEERS COOLERS"                 : "Beverages",
    "COFFEE"                        : "Beverages",
    "COCOA DRINK MIXES"             : "Beverages",
    "ENERGY SPORTS DRINKS"          : "Beverages",
    "JUICE NECTARS"                 : "Beverages",
    "RED WINES"                     : "Beverages",
    "SOFT DRINKS"                   : "Beverages",
    "SPIRITS"                       : "Beverages",
    "SPECIALTY WINES CHAMPAGNES"    : "Beverages",
    "TEA"                           : "Beverages",
    "WATER SELTZER SPARKLING WATER" : "Beverages",
    "WHITE WINES"                   : "Beverages",

    # FROZEN FOODS
    "FROZEN APPETIZERS SIDES"       : "Frozen Foods",
    "FROZEN BREADS DOUGHS"          : "Frozen Foods",
    "FROZEN BREAKFAST"              : "Frozen Foods",
    "FROZEN DESSERT"                : "Frozen Foods",
    "FROZEN JUICE"                  : "Frozen Foods",
    "FROZEN MEALS"                  : "Frozen Foods",
    "FROZEN MEAT SEAFOOD"           : "Frozen Foods",
    "FROZEN PIZZA"                  : "Frozen Foods",
    "FROZEN PRODUCE"                : "Frozen Foods",
    "FROZEN VEGAN VEGETARIAN"       : "Frozen Foods",
    "ICE CREAM ICE"                 : "Frozen Foods",
    "ICE CREAM TOPPINGS"            : "Frozen Foods",

    # PANTRY & DRY GOODS
    "ASIAN FOODS"                   : "Pantry & Dry Goods",
    "BAKING INGREDIENTS"            : "Pantry & Dry Goods",
    "BAKING SUPPLIES DECOR"         : "Pantry & Dry Goods",
    "BULK DRIED FRUITS VEGETABLES"  : "Pantry & Dry Goods",
    "BULK GRAINS RICE DRIED GOODS"  : "Pantry & Dry Goods",
    "CANNED FRUIT APPLESAUCE"       : "Pantry & Dry Goods",
    "CANNED JARRED VEGETABLES"      : "Pantry & Dry Goods",
    "CANNED MEALS BEANS"            : "Pantry & Dry Goods",
    "CONDIMENTS"                    : "Pantry & Dry Goods",
    "DRY PASTA"                     : "Pantry & Dry Goods",
    "GRAINS RICE DRIED GOODS"       : "Pantry & Dry Goods",
    "HONEYS SYRUPS NECTARS"         : "Pantry & Dry Goods",
    "INDIAN FOODS"                  : "Pantry & Dry Goods",
    "INSTANT FOODS"                 : "Pantry & Dry Goods",
    "KOSHER FOODS"                  : "Pantry & Dry Goods",
    "LATINO FOODS"                  : "Pantry & Dry Goods",
    "OILS VINEGARS"                 : "Pantry & Dry Goods",
    "PASTA SAUCE"                   : "Pantry & Dry Goods",
    "PICKLED GOODS OLIVES"          : "Pantry & Dry Goods",
    "PRESERVED DIPS SPREADS"        : "Pantry & Dry Goods",
    "SALAD DRESSING TOPPINGS"       : "Pantry & Dry Goods",
    "SOUP BROTH BOUILLON"           : "Pantry & Dry Goods",
    "SPICES SEASONINGS"             : "Pantry & Dry Goods",
    "SPREADS"                       : "Pantry & Dry Goods",

    # BREAKFAST
    "BREAKFAST BARS PASTRIES"       : "Breakfast",
    "CEREAL"                        : "Breakfast",
    "HOT CEREAL PANCAKE MIXES"      : "Breakfast",

    # DELI & PREPARED
    "FRESH DIPS TAPENADES"          : "Deli & Prepared",
    "FRESH PASTA"                   : "Deli & Prepared",
    "PREPARED MEALS"                : "Deli & Prepared",
    "PREPARED SOUPS SALADS"         : "Deli & Prepared",
    "REFRIGERATED"                  : "Deli & Prepared",
    "TOFU MEAT ALTERNATIVES"        : "Deli & Prepared",

    # HOUSEHOLD & CLEANING
    "AIR FRESHENERS CANDLES"        : "Household & Cleaning",
    "CLEANING PRODUCTS"             : "Household & Cleaning",
    "DISH DETERGENTS"               : "Household & Cleaning",
    "FOOD STORAGE"                  : "Household & Cleaning",
    "KITCHEN SUPPLIES"              : "Household & Cleaning",
    "LAUNDRY"                       : "Household & Cleaning",
    "MORE HOUSEHOLD"                : "Household & Cleaning",
    "PAPER GOODS"                   : "Household & Cleaning",
    "PLATES BOWLS CUPS FLATWARE"    : "Household & Cleaning",
    "TRASH BAGS LINERS"             : "Household & Cleaning",

    # PERSONAL CARE
    "BEAUTY"                        : "Personal Care",
    "BODY LOTIONS SOAP"             : "Personal Care",
    "COLD FLU ALLERGY"              : "Personal Care",
    "DEODORANTS"                    : "Personal Care",
    "DIGESTION"                     : "Personal Care",
    "EYE EAR CARE"                  : "Personal Care",
    "FACIAL CARE"                   : "Personal Care",
    "FEMININE CARE"                 : "Personal Care",
    "FIRST AID"                     : "Personal Care",
    "HAIR CARE"                     : "Personal Care",
    "MINT GUM"                      : "Personal Care",
    "MUSCLES JOINTS PAIN RELIEF"    : "Personal Care",
    "ORAL HYGIENE"                  : "Personal Care",
    "SHAVE NEEDS"                   : "Personal Care",
    "SKIN CARE"                     : "Personal Care",
    "SOAP"                          : "Personal Care",
    "VITAMINS SUPPLEMENTS"          : "Personal Care",

    # BABY
    "BABY ACCESSORIES"              : "Baby",
    "BABY BATH BODY CARE"           : "Baby",
    "BABY FOOD FORMULA"             : "Baby",
    "DIAPERS WIPES"                 : "Baby",

    # PETS
    "CAT FOOD CARE"                 : "Pets",
    "DOG FOOD CARE"                 : "Pets",
}

HARMONISED_CATEGORIES = [
    "Fresh Produce", "Dairy & Eggs", "Meat & Seafood",
    "Bakery & Bread", "Snacks", "Beverages", "Frozen Foods",
    "Pantry & Dry Goods", "Breakfast", "Deli & Prepared",
    "Household & Cleaning", "Personal Care", "Baby", "Pets"
]


# =============================================================================
# STEP 2 — APPLY CROSSWALK
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2 — Applying Crosswalk")
print("=" * 60)

# --- Dunnhumby ---
dh["harmonised_category"] = dh["COMMODITY_DESC"].map(DUNNHUMBY_MAP)
dh_total     = len(dh)
dh_matched   = dh["harmonised_category"].notna().sum()
dh_unmatched = dh["harmonised_category"].isna().sum()
dh_filtered  = dh[dh["harmonised_category"].notna()].copy()
dh_filtered.to_csv(os.path.join(OUTPUT_DIR, "dunnhumby_harmonised.csv"), index=False)
print(f"\n  Dunnhumby  matched  : {dh_matched:,} / {dh_total:,} rows ({dh_matched/dh_total*100:.1f}%)")
print(f"  Dunnhumby  excluded : {dh_unmatched:,} rows ({dh_unmatched/dh_total*100:.1f}%) — non-grocery noise")

# --- Instacart ---
ic["harmonised_category"] = ic["aisle"].map(INSTACART_MAP)
ic_total     = len(ic)
ic_matched   = ic["harmonised_category"].notna().sum()
ic_unmatched = ic["harmonised_category"].isna().sum()
ic_filtered  = ic[ic["harmonised_category"].notna()].copy()
ic_filtered.to_csv(os.path.join(OUTPUT_DIR, "instacart_harmonised.csv"), index=False)
print(f"\n  Instacart  matched  : {ic_matched:,} / {ic_total:,} rows ({ic_matched/ic_total*100:.1f}%)")
print(f"  Instacart  excluded : {ic_unmatched:,} rows ({ic_unmatched/ic_total*100:.1f}%) — non-grocery noise")


# =============================================================================
# STEP 3 — STATISTICAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3 — Harmonisation Statistics")
print("=" * 60)

# --- Build per-category stats for each dataset ---
dh_basket_col = "BASKET_ID"
ic_basket_col = "order_id"

dh_stats = (
    dh_filtered.groupby("harmonised_category")
    .agg(
        dh_rows        = ("harmonised_category", "count"),
        dh_baskets     = (dh_basket_col, "nunique"),
        dh_source_cats = ("COMMODITY_DESC", "nunique"),
    )
    .reset_index()
)
dh_stats["dh_pct_rows"] = (dh_stats["dh_rows"] / dh_matched * 100).round(1)

ic_stats = (
    ic_filtered.groupby("harmonised_category")
    .agg(
        ic_rows        = ("harmonised_category", "count"),
        ic_baskets     = (ic_basket_col, "nunique"),
        ic_source_cats = ("aisle", "nunique"),
    )
    .reset_index()
)
ic_stats["ic_pct_rows"] = (ic_stats["ic_rows"] / ic_matched * 100).round(1)

# Merge into one comparison table
stats = pd.merge(dh_stats, ic_stats, on="harmonised_category", how="outer").fillna(0)
stats = stats.sort_values("harmonised_category").reset_index(drop=True)

# Rank columns
stats["dh_rank"] = stats["dh_rows"].rank(ascending=False).astype(int)
stats["ic_rank"] = stats["ic_rows"].rank(ascending=False).astype(int)
stats["rank_shift"] = stats["dh_rank"] - stats["ic_rank"]

# --- Print 1: Match Coverage ---
print("\n── Coverage & Source Complexity ──────────────────────────")
print(f"  {'Category':<22} {'DH src cats':>11} {'IC src cats':>11}  {'DH rows':>10} {'IC rows':>12}")
print(f"  {'-'*22} {'-'*11} {'-'*11}  {'-'*10} {'-'*12}")
for _, r in stats.iterrows():
    print(f"  {r['harmonised_category']:<22} {int(r['dh_source_cats']):>11,} {int(r['ic_source_cats']):>11,}  "
          f"{int(r['dh_rows']):>10,} {int(r['ic_rows']):>12,}")

# --- Print 2: Share of basket distribution ---
print("\n── Category Share of Total Transactions (%) ──────────────")
print(f"  {'Category':<22} {'DH share%':>10} {'IC share%':>10}  {'DH rank':>8} {'IC rank':>8}  {'Rank shift':>10}")
print(f"  {'-'*22} {'-'*10} {'-'*10}  {'-'*8} {'-'*8}  {'-'*10}")
for _, r in stats.iterrows():
    shift_str = f"+{int(r['rank_shift'])}" if r['rank_shift'] > 0 else str(int(r['rank_shift']))
    print(f"  {r['harmonised_category']:<22} {r['dh_pct_rows']:>9.1f}% {r['ic_pct_rows']:>9.1f}%  "
          f"{int(r['dh_rank']):>8} {int(r['ic_rank']):>8}  {shift_str:>10}")

# --- Print 3: Basket reach ---
print("\n── Unique Basket/Order Reach per Category ─────────────────")
print(f"  {'Category':<22} {'DH baskets':>12} {'IC orders':>12}")
print(f"  {'-'*22} {'-'*12} {'-'*12}")
for _, r in stats.iterrows():
    print(f"  {r['harmonised_category']:<22} {int(r['dh_baskets']):>12,} {int(r['ic_baskets']):>12,}")

# --- Print 4: Biggest rank divergences (most interesting for MBA) ---
print("\n── Top Category Rank Divergences (most interesting for MBA comparison) ──")
divergent = stats.reindex(stats["rank_shift"].abs().sort_values(ascending=False).index)
for _, r in divergent.head(5).iterrows():
    direction = "higher in DH" if r["rank_shift"] > 0 else "higher in IC"
    print(f"  {r['harmonised_category']:<22}  DH rank #{int(r['dh_rank'])}  vs  IC rank #{int(r['ic_rank'])}  "
          f"→ {abs(int(r['rank_shift']))} places {direction}")

# --- Final counts ---
print("\n── Overall Summary ─────────────────────────────────────────")
print(f"  Harmonised categories          : {len(HARMONISED_CATEGORIES)}")
print(f"  Dunnhumby  — rows kept         : {dh_matched:,} / {dh_total:,}  ({dh_matched/dh_total*100:.1f}%)")
print(f"  Instacart  — rows kept         : {ic_matched:,} / {ic_total:,}  ({ic_matched/ic_total*100:.1f}%)")
print(f"  Dunnhumby  — unique baskets    : {dh_filtered[dh_basket_col].nunique():,}")
print(f"  Instacart  — unique orders     : {ic_filtered[ic_basket_col].nunique():,}")
print(f"  Source cats mapped (DH)        : {len(DUNNHUMBY_MAP)} commodity descriptions")
print(f"  Source cats mapped (IC)        : {len(INSTACART_MAP)} aisles")

print("\nPhase 1 complete. Harmonised files saved to:", OUTPUT_DIR)
print("Next step -> Phase 2: Build transaction matrices")
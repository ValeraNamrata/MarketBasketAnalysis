## Prerequisites

### Python version
Python 3.10 or later recommended. The project was developed and tested on Python 3.13.

### Virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### Required libraries
```bash
pip install pandas mlxtend matplotlib
```

| Library | Version tested | Purpose |
|---|---|---|
| pandas | 2.x | Data loading, manipulation, output |
| mlxtend | 0.23.x | FP-Growth and Apriori algorithms |
| matplotlib | 3.x | Parameter selection scatter plot (Script 4) |

---

## Raw Data Requirements

Before running any script, download and place the raw data files in the following locations. The scripts expect this exact folder structure.
```
data/
  raw/
    dunnhumby/
      transaction_data.csv       <- household-level transaction records
      product.csv                <- product metadata with COMMODITY_DESC
    instacart/
      order_products__prior.csv  <- order-product line items
      products.csv               <- product metadata
      aisles.csv                 <- aisle labels (134 aisles)
      departments.csv            <- department labels
```

**Dunnhumby source:** The Complete Journey dataset (available via Dunnhumby / Kaggle)  https://www.kaggle.com/code/haijie/dunnhumby-exploratory-data-analysis/input

**Instacart source:** Instacart Online Grocery Shopping Dataset 2017 (available via Kaggle) https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis

---

## Folder Structure After Full Pipeline Run

```
data/
  raw/                           <- original downloaded files (not modified)
  prepared/
    dunnhumby_prepared.csv       <- output of Script 1
    instacart_prepared.csv       <- output of Script 1
    dunnhumby_categories.txt     <- output of Script 2 (diagnostic reference)
    instacart_categories.txt     <- output of Script 2 (diagnostic reference)
    dunnhumby_harmonised.csv     <- output of Script 2 (crosswalk applied)
    instacart_harmonised.csv     <- output of Script 2 (crosswalk applied)
  phase3/
    dunnhumby_transaction_matrix.csv   <- output of Script 3
    instacart_transaction_matrix.csv   <- output of Script 3
  phase4/
    stage1_results.csv           <- output of Script 4 (all 144 combinations)
    parameter_selection_chart.png  <- output of Script 4 (scatter plot)
    dunnhumby_rules.csv          <- output of Script 4 (835 rules)
    instacart_rules.csv          <- output of Script 4 (423 rules)
    instacart_rules_lift15.csv   <- output of Script 4 (supplementary, 102 rules)
    stage2_summary.csv           <- output of Script 4 (model summary)
  phase5/
    classification_universal.csv         <- output of Script 5 (85 combinations)
    classification_strength_divergent.csv  <- output of Script 5 (144 combinations)
    classification_dh_specific.csv       <- output of Script 5 (129 combinations)
    classification_ic_specific.csv       <- output of Script 5 (14 combinations)
    classification_full.csv              <- output of Script 5 (372 combinations)
    phase5_summary.csv                   <- output of Script 5 (summary stats)
```

---

## Scripts -- Execution Order

Run the scripts in strict sequential order. Each script depends on the output of the previous one.

---

### Script 1 -- Data Loading and Preparation
**File:** `1_DataLoading&Prep.py`  
**Phase:** Data Loading  
**Runtime:** ~2 minutes (Instacart is 32M rows)

```bash
python "1_DataLoading&Prep.py"
```

**What it does:**
- Loads Dunnhumby transaction and product files and joins on PRODUCT_ID
- Loads Instacart order_products, products, aisles, and departments files and joins on product_id
- Validates that no rows are dropped due to missing category fields
- Saves `dunnhumby_prepared.csv` and `instacart_prepared.csv` to `data/prepared/`

**Key output columns:**
- Dunnhumby: `BASKET_ID`, `COMMODITY_DESC`, `DEPARTMENT`, `household_key`
- Instacart: `order_id`, `aisle`, `department`, `product_id`

**Expected console output:**
```
Dunnhumby -- Unique baskets: 276,484 | Unique households: 2,500
Instacart -- Unique orders: 3,214,874 | Unique aisles: 134
```

---

### Script 2 -- Category Diagnostic and Crosswalk
**File:** `3_CategoryDiagnostic.py`  
**Phase:** Data Understanding (Phase 2)  
**Runtime:** ~1 minute

```bash
python 3_CategoryDiagnostic.py
```

**What it does:**
- Step 1: Scans all unique COMMODITY_DESC values (Dunnhumby) and aisle values (Instacart) and saves them to reference `.txt` files
- Step 2: Applies the 14-category crosswalk using two Python dictionaries (`DUNNHUMBY_MAP` and `INSTACART_MAP`) to add a `harmonised_category` column to both datasets. Non-grocery categories (gas, pharmacy, video rental) are excluded.
- Step 3: Prints a full statistical summary of the harmonisation including match rates, category share of transactions, basket reach per category, and rank divergences between datasets

**Crosswalk match rates:**
- Dunnhumby: 90.9% of rows matched to a harmonised category
- Instacart: 99.6% of rows matched to a harmonised category

**Expected console output:**
```
Dunnhumby  matched  : 2,357,xxx / 2,595,732 rows (90.9%)
Instacart  matched  : 32,xxx,xxx / 32,434,489 rows (99.6%)
```

**Note:** If you need to update the crosswalk mappings (for example, if a new product category appears in the raw data), edit the `DUNNHUMBY_MAP` or `INSTACART_MAP` dictionaries directly in this script before re-running.

---

### Script 3 -- Transaction Matrix Construction
**File:** `2_TransactionMatrix.py`  
**Phase:** Data Preparation (Phase 3)  
**Runtime:** ~5 minutes (Instacart pivot is memory-intensive)

```bash
python 2_TransactionMatrix.py
```

**What it does:**
- Reads the harmonised CSV files produced by Script 2
- Pivots each dataset from line-item format (one row per category per basket) to binary transaction matrix format (one row per basket, one column per category, True/False)
- Validates that both matrices have the same 14 column names
- Saves both matrices to `data/phase3/`

**Output matrix dimensions:**
- Dunnhumby: 240,855 rows x 14 columns, density 28.7%
- Instacart: 3,213,813 rows x 14 columns, density 31.7%

**Memory note:** The Instacart matrix pivot requires approximately 4GB of available RAM. If you encounter memory errors, consider chunking the input or running on a machine with more RAM.

---

### Script 4a -- Parameter Sensitivity Testing (Stage 1)
**File:** `4_ModelComparison.py`  
**Phase:** Modeling (Phase 4) -- Stage 1  
**Runtime:** ~3 minutes (50,000-row samples from each dataset)

```bash
python 4_ModelComparison.py
```

**What it does:**
- Samples 50,000 rows from each transaction matrix
- Tests both Apriori and FP-Growth across 144 parameter combinations:
  - Support: 0.5%, 1%, 2%, 5%
  - Confidence: 20%, 40%, 60%
  - Lift: 1.0, 1.2, 1.5
- Saves full results to `data/phase4/stage1_results.csv`
- Prints a side-by-side comparison showing Dunnhumby and Instacart rule counts per combination
- Highlights the chosen parameter combination
- Generates and saves a scatter plot (`parameter_selection_chart.png`) showing all combinations with the chosen one marked in red

**Chosen parameters (confirmed by Stage 1 output):**

| Parameter | Value | Justification |
|---|---|---|
| Algorithm | FP-Growth | Identical output to Apriori; 6-9x faster at full Instacart scale |
| Min Support | 1% | Standard academic threshold; most balanced rule volumes across both datasets |
| Min Confidence | 40% | Filters weak rules without over-restricting Instacart |
| Min Lift | 1.2 | Applied identically to both datasets for comparability |
| Max itemset length | 3 | Pairs and triplets |

---

### Script 4b -- Full FP-Growth Model Run (Stage 2)
**File:** `4_modelImplementation.py`  
**Phase:** Modeling (Phase 4) -- Stage 2  
**Runtime:** ~15 seconds (Dunnhumby ~1s, Instacart ~10s)

```bash
python 4_modelImplementation.py
```

**What it does:**
- Runs FP-Growth on the full Dunnhumby transaction matrix (240,855 baskets)
- Runs FP-Growth on the full Instacart transaction matrix (3,213,813 orders)
- Applies the agreed parameters from Stage 1
- Generates and prints top 20 rules by lift for each dataset
- Runs a supplementary Instacart filter at lift 1.5 to document the high-support inflation effect
- Saves four output files to `data/phase4/`

**Model results summary:**

| | Dunnhumby | Instacart |
|---|---|---|
| FP-Growth time | 0.78s | 10.34s |
| Frequent itemsets | 403 | 345 |
| Rules (conf 40%) | 859 | 599 |
| Rules (lift 1.2) | **835** | **423** |
| Rules (lift 1.5) | 699 | 102 |
| Avg lift | 1.785 | 1.406 |
| Max lift | 3.280 | 1.936 |

---

### Script 5 -- Cross-Retailer Classification
**File:** `5_CrossComparison.py`  
**Phase:** Evaluation (Phase 5)  
**Runtime:** ~30 seconds

```bash
python 5_CrossComparison.py
```

**What it does:**
- Loads both rule sets from Phase 4
- Builds combination keys by stripping rule directionality (so A -> B and B -> A are treated as the same combination)
- Identifies which combinations appear in both datasets, only in Dunnhumby, or only in Instacart
- Classifies all 229 common combinations using a divergence threshold of 0.30:
  - Lift difference <= 0.30 → **UNIVERSAL**
  - Lift difference > 0.30 → **STRENGTH-DIVERGENT**
  - One dataset only → **FORMAT-SPECIFIC**
- Saves six output files to `data/phase5/`

**Classification results:**

| Class | Count |
|---|---|
| Universal | 85 |
| Strength-Divergent | 144 |
| Format-Specific (Dunnhumby only) | 129 |
| Format-Specific (Instacart only) | 14 |
| **Total** | **372** |

---

## Running the Full Pipeline -- Quick Reference

```bash
# Step 1 -- Load and prepare raw data
python "1_DataLoading&Prep.py"

# Step 2 -- Category diagnostic and crosswalk
python 3_CategoryDiagnostic.py

# Step 3 -- Build transaction matrices
python 2_TransactionMatrix.py

# Step 4a -- Parameter sensitivity test (Stage 1)
python 4_ModelComparison.py

# Step 4b -- Full FP-Growth run (Stage 2)
python 4_modelImplementation.py

# Step 5 -- Cross-retailer classification
python 5_CrossComparison.py
```

Total estimated runtime on a standard laptop: **~20 minutes** (dominated by the Instacart data loading and matrix pivot in Script 3).

---

## Rerunning on New Data

All scripts are parameterised. To run the pipeline on a different grocery retailer's transaction data:

1. Replace the raw input files in `data/raw/`
2. Update the column name mappings at the top of `1_DataLoading&Prep.py` to match your data schema
3. Update the crosswalk dictionary in `3_CategoryDiagnostic.py` to map your category labels to the 14 harmonised categories
4. Run all five scripts in order -- no other changes are required

The 14 harmonised categories, the FP-Growth parameters, and the 0.30 divergence threshold are all defined as named constants at the top of their respective scripts and can be adjusted without modifying any logic.

---

## Key Configuration Constants

| Script | Constant | Default | Description |
|---|---|---|---|
| `4_ModelComparison.py` | `SAMPLE_SIZE` | 50,000 | Row sample size for Stage 1 sensitivity test |
| `4_modelImplementation.py` | `MIN_SUPPORT` | 0.01 | Minimum support threshold |
| `4_modelImplementation.py` | `MIN_CONFIDENCE` | 0.40 | Minimum confidence threshold |
| `4_modelImplementation.py` | `MIN_LIFT` | 1.2 | Primary lift filter |
| `4_modelImplementation.py` | `MIN_LIFT_SUPP` | 1.5 | Supplementary Instacart lift filter |
| `4_modelImplementation.py` | `MAX_LEN` | 3 | Maximum itemset length |
| `5_CrossComparison.py` | `DIVERGENCE_THRESHOLD` | 0.30 | Lift difference threshold for Universal vs Strength-Divergent |

---

## Troubleshooting

**MemoryError during Script 3**  
The Instacart pivot requires ~4GB RAM. Close other applications or increase available memory before running.

**mlxtend import error**  
Ensure mlxtend is installed in the active virtual environment: `pip install mlxtend`

**FileNotFoundError on any script**  
Confirm that all previous scripts have completed successfully and that the `data/` folder structure matches the layout described above.

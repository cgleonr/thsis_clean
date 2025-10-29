# Quick Start Guide
## Getting Your Thesis Project Up and Running

This guide will help you get from your current messy repository to a working system in **~2-3 hours**.

---

## 🎯 Overview

You'll rebuild your project with:
1. ✅ Clean data pipeline
2. ✅ Working baseline (Sentence-BERT)
3. ✅ Novel hierarchical model (academic contribution)
4. ✅ Web demo interface
5. ✅ Evaluation framework

---

## 📋 Prerequisites

- Python 3.8+
- 8GB+ RAM
- Internet connection (for downloading models)

---

## 🚀 Step-by-Step Setup

### Step 1: Environment Setup (5 minutes)

```bash
# Navigate to clean project directory
cd thesis-clean

# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# This will take a few minutes to download all packages
```

### Step 2: Data Acquisition (15-30 minutes)

You need two main datasets:

#### A. WTO ADB Tariff Data

**Option 1: Download from WTO (Recommended)**
1. Go to: https://www.wto.org/english/res_e/statis_e/trade_datasets_e.htm
2. Navigate to "Tariff Analysis Online" or "Analytical Database"
3. Download tariff data for:
   - Countries: EU, Canada, Switzerland
   - Years: 2020-2024
   - Format: CSV
4. Save to: `data/raw/wto/adb_tariff_data.csv`

**Option 2: Use Your Existing Data**
If you already have WTO data from your old repo:
```bash
# Copy from old repo
cp /path/to/old/repo/data/raw/wto/*.csv data/raw/wto/
```

#### B. WCO HS Descriptions

**Option 1: Download from WCO**
1. Go to: https://www.wcoomd.org/en/topics/nomenclature/instrument-and-tools/hs-nomenclature-2022-edition.aspx
2. Download HS2022 nomenclature (Excel or CSV format)
3. Save to: `data/raw/wco/wco_hs_hs2022.csv`

**Option 2: Use Existing Data**
```bash
# Copy from old repo
cp /path/to/old/repo/data/raw/wco/*.csv data/raw/wco/
```

**Quick Test**: Check if files exist
```bash
ls -la data/raw/wto/
ls -la data/raw/wco/
```

### Step 3: Data Preprocessing (10 minutes)

```bash
# Run preprocessing
python src/data/preprocessing.py
```

Expected output:
- `data/processed/wto_tariffs_clean.csv`
- `data/processed/wco_hs_descriptions_clean.csv`
- `data/processed/unified_hs_tariff_dataset.csv`

**Troubleshooting**:
- If you get "File not found" errors, check Step 2
- If column names don't match, you may need to adjust the preprocessing script

### Step 4: Build Baseline Model (15 minutes)

```bash
# Build and test baseline
python src/models/baseline.py
```

This will:
1. Load HS descriptions
2. Create Sentence-BERT embeddings
3. Build search index
4. Run demo predictions

Expected output:
- `models/baseline/hs_embeddings.npy`
- `models/baseline/hs_metadata.csv`
- `models/baseline/config.pkl`

You should see test predictions for:
- "leather handbag with shoulder strap"
- "cotton t-shirt for men"
- "smartphone with touchscreen"
- etc.

### Step 5: Launch Web Interface (5 minutes)

```bash
# Start Streamlit app
streamlit run src/app/streamlit_app.py
```

This opens a browser at `http://localhost:8501`

**Test it**:
1. Enter a product description
2. Click "Classify Product"
3. See top-3 HS code predictions with tariff rates

---

## 🎓 Academic Component: Hierarchical Model

### Step 6: Generate Synthetic Training Data (30 minutes)

For the hierarchical model, you need training examples. Since you have the single-instance-per-class problem, generate synthetic data:

**Option A: Manual Creation** (Quick but limited)
Create `data/processed/synthetic_train.csv`:
```csv
description,hs6
leather women's handbag,420221
leather men's briefcase,420222
cotton men's t-shirt,610910
polyester women's t-shirt,610990
smartphone with 5G,851712
```

**Option B: Use GPT for Generation** (Better, requires API key)
```python
# Create src/data/synthetic_gen.py
# Use GPT-4 to generate variations of HS descriptions
# Example: HS 420221 → 20 different product descriptions
```

**Option C: Data Augmentation**
Use paraphrasing to create variations:
- Original: "articles of leather or of composition leather"
- Augmented: "leather goods and products"
- Augmented: "items made from leather or synthetic leather"

### Step 7: Train Hierarchical Model (60-90 minutes)

```bash
# Train hierarchical model
python src/models/hierarchical.py --train --epochs 20
```

This trains the custom neural network that:
- Exploits HS code hierarchy (Chapter → Heading → HS6)
- Uses DistilBERT as base encoder
- Multi-task learning across 3 levels

**Note**: This requires GPU for reasonable training time. If you don't have GPU:
- Use Google Colab (free GPU)
- Reduce epochs to 5-10
- Use smaller model (distilbert-base-uncased)

### Step 8: Evaluation (20 minutes)

```bash
# Evaluate both models
python src/evaluation/compare_models.py
```

This generates:
- Top-1, Top-3, Top-5 accuracy
- Mean Reciprocal Rank (MRR)
- Hierarchical accuracy (Chapter, Heading, HS6)
- Per-category performance
- Confusion matrices

---

## 📊 What You'll Have After This

1. **Working System**
   - Data pipeline: acquisition → preprocessing → modeling
   - Baseline model with ~70-80% Top-3 accuracy (estimated)
   - Web interface for demos

2. **Academic Contribution**
   - Novel hierarchical architecture
   - Comparative evaluation vs baseline
   - Methodology for single-instance problem (synthetic data)

3. **Thesis Content**
   - Code for all experiments
   - Evaluation results
   - Working prototype
   - Screenshots for thesis

---

## 🐛 Common Issues & Solutions

### Issue: "ModuleNotFoundError"
```bash
# Make sure you're in the venv
pip install -r requirements.txt
```

### Issue: "CUDA out of memory" (when training)
```python
# Reduce batch size in hierarchical.py
batch_size = 8  # instead of 16
```

### Issue: "File not found" errors
```bash
# Check data directories
ls data/raw/wto/
ls data/raw/wco/
ls data/processed/
```

### Issue: Baseline predictions seem random
- Check if embeddings were created correctly
- Verify HS descriptions are cleaned properly
- Look at similarity scores (should be > 0.3 for good matches)

---

## 🎯 Next Steps After Quick Start

Once you have the basics working:

1. **Improve Data Quality**
   - Get more comprehensive WTO data
   - Clean descriptions better
   - Add more countries

2. **Generate More Synthetic Data**
   - Use GPT-4 for high-quality augmentation
   - Create 50-100 examples per HS6 code
   - Validate synthetic data quality

3. **Tune Hierarchical Model**
   - Experiment with loss weights (α, β, γ)
   - Try different base models (BERT, RoBERTa)
   - Add dropout, regularization

4. **Comprehensive Evaluation**
   - Collect real product descriptions (manual)
   - Test on e-commerce data (scrape Amazon/eBay)
   - Error analysis by category

5. **Write Thesis Sections**
   - Methodology chapter (complete description)
   - Results chapter (tables, charts)
   - Discussion (why hierarchical works)

---

## 📞 Help & Support

If you get stuck:
1. Check the logs - most errors are self-explanatory
2. Read the docstrings in the code
3. Look at the example notebooks (when created)
4. Ask me for help!

---

## ✅ Checklist

Use this to track your progress:

- [ ] Environment set up
- [ ] WTO data downloaded
- [ ] WCO data downloaded
- [ ] Preprocessing completed
- [ ] Baseline model built
- [ ] Baseline model tested
- [ ] Web interface running
- [ ] Synthetic data created
- [ ] Hierarchical model implemented
- [ ] Hierarchical model trained
- [ ] Evaluation completed
- [ ] Results documented

**Good luck! You've got this! 🚀**

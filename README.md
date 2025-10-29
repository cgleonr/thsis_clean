# Machine Learning-Based HS Code Classification and Customs Duty Estimation

Master's Thesis Project by Carlos Leon  
Supervisor: Oliver Staubli  
Client: On AG

## 🎯 Project Overview

This thesis develops a machine learning system for automatically classifying product descriptions into Harmonized System (HS) codes at the HS6 level and estimating applicable customs duties across multiple jurisdictions (EU, Canada, Switzerland).

### Key Innovation
- **Hierarchical Neural Classifier** that exploits the natural taxonomy structure of HS codes (Chapter → Heading → Subheading)
- **Synthetic Data Generation** to overcome the single-instance-per-class problem in customs datasets
- **Multi-Country Tariff Integration** with real-time duty estimation

## 📁 Project Structure

```
thesis-ml-customs/
├── data/                       # Data storage
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned, model-ready data
│   └── embeddings/             # Precomputed embeddings
├── src/                        # Source code
│   ├── data/                   # Data acquisition and preprocessing
│   ├── models/                 # Model implementations
│   ├── evaluation/             # Evaluation metrics and analysis
│   └── app/                    # Web interface
├── models/                     # Trained model artifacts
│   ├── baseline/               # Sentence-BERT baseline
│   ├── hierarchical/           # Custom hierarchical model
│   └── evaluation/             # Performance metrics
├── notebooks/                  # Jupyter notebooks for experiments
├── tests/                      # Unit tests
├── configs/                    # Configuration files
└── docs/                       # Documentation

```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (optional)
python -m spacy download en_core_web_sm
```

### 2. Data Acquisition

```bash
# Run data download scripts
python src/data/acquisition.py --sources wto wco
```

### 3. Run Baseline Model

```bash
# Train Sentence-BERT baseline
python src/models/baseline.py --train

# Evaluate baseline
python src/models/baseline.py --evaluate
```

### 4. Train Hierarchical Model

```bash
# Train custom hierarchical classifier
python src/models/hierarchical.py --train --epochs 20

# Evaluate hierarchical model
python src/models/hierarchical.py --evaluate
```

### 5. Launch Web Interface

```bash
# Start Streamlit app
streamlit run src/app/streamlit_app.py
```

## 📊 Models

### Baseline: Semantic Retrieval (Sentence-BERT)
- **Approach**: Embed HS descriptions, retrieve top-K by cosine similarity
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Advantages**: Fast, interpretable, zero-shot capable
- **Limitations**: Doesn't learn from data, flat classification

### Proposed: Hierarchical Neural Classifier
- **Architecture**: 
  - Shared encoder (DistilBERT)
  - Three prediction heads (Chapter, Heading, Subheading)
  - Hierarchical loss function
- **Training**: Multi-task learning with weighted losses
- **Advantages**: Exploits HS structure, learns from data
- **Innovation**: Novel architecture for customs classification

## 📈 Evaluation Metrics

- **Top-K Accuracy** (K=1,3,5): Standard retrieval metrics
- **Mean Reciprocal Rank (MRR)**: Ranks position of correct answer
- **Hierarchical Accuracy**: Accuracy at each HS level (2-digit, 4-digit, 6-digit)
- **Tariff-Weighted Accuracy**: Weights errors by duty rate impact

## 💾 Data Sources

1. **WTO Analytical Database (ADB)**: Applied MFN tariff rates
2. **WCO HS Nomenclature (HS2022)**: Official HS descriptions
3. **Synthetic Product Descriptions**: Generated for training/evaluation

## 🎓 Thesis Contributions

1. **Novel Hierarchical Architecture** for HS classification
2. **Synthetic Data Generation Method** for customs domain
3. **Comprehensive Evaluation Framework** with multiple metrics
4. **Working Prototype** integrating classification + tariff estimation

## 📝 Citation

```bibtex
@mastersthesis{leon2025customs,
  author = {Carlos Leon},
  title = {Machine Learning-Based Estimation and Analysis of Customs Duties and Compliance Requirements},
  school = {Hochschule Luzern},
  year = {2025},
  type = {Master's Thesis}
}
```

## 📧 Contact

- **Author**: Carlos Leon ([carlos.leon@stud.hslu.ch](mailto:carlos.leon@stud.hslu.ch))
- **Supervisor**: Oliver Staubli ([oliver.staubli@revolytics.com](mailto:oliver.staubli@revolytics.com))
- **Client**: Sofia Viale, On AG ([sofia.viale@on-running.com](mailto:sofia.viale@on-running.com))

## 📄 License

This project is part of a Master's thesis at Hochschule Luzern.

---

**Status**: Active Development (October 2025)

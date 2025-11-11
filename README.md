# Hierarchical Transformer Models for Automated HS Code Classification and Trade Compliance

**Master's Thesis Project**  
Carlos Leon | Hochschule Luzern - Wirtschaft | December 2025

Supervised by: Oliver Staubli (Revolytics)  
Client: On AG

---

## 📋 Overview

This repository contains the implementation of a machine learning-based system for automated classification of products into Harmonized System (HS) codes and assigning of customs duties. The project addresses the challenge of manual, rule-based customs classification by leveraging transformer-based neural networks to predict HS codes from natural language product descriptions.

### Research Question

*How can machine learning and natural language processing methods be applied to automate the classification of product descriptions into HS codes and support the analysis of customs compliance in international trade?*

### Key Features

- **Dual-Model Architecture**: Baseline retrieval model + hierarchical neural classifier
- **Hierarchical Classification**: Predicts at three levels (Chapter → Heading → HS6)
- **Tariff Integration**: Automatic duty rate lookup for Canada, EU, and Switzerland
- **Web Interface**: Interactive Streamlit application for real-time classification
- **High Accuracy**: 97.19% validation accuracy on HS6 classification

---

## 🎯 Problem Statement

Multinational companies face significant complexity in customs classification:
- **5,612 unique HS6 codes** to navigate globally
- Manual classification is error-prone and resource-intensive
- Static tariff tables don't adapt to regulatory changes
- Misclassification leads to financial penalties and shipment delays

This system automates HS code prediction using state-of-the-art NLP techniques, reducing classification time and improving accuracy.

---

## 🏗️ System Architecture

### Model 1: Baseline (Sentence-BERT)
- **Architecture**: Semantic similarity retrieval
- **Base Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings)
- **Method**: Cosine similarity search over encoded WCO descriptions
- **Advantages**: Fast inference (<50ms), no training required, interpretable

### Model 2: Hierarchical Classifier (DistilBERT)
- **Architecture**: Multi-output neural network with three parallel classification heads
- **Base Model**: `distilbert-base-uncased` (6 layers, 768-dim hidden size)
- **Training Data**: 179,000 augmented examples from 5,612 HS6 codes
- **Loss Function**: Weighted multi-level cross-entropy (0.2 Chapter + 0.3 Heading + 0.5 HS6)
- **Performance**: 
  - Chapter (2-digit): 99.77% accuracy
  - Heading (4-digit): 98.79% accuracy
  - HS6 (6-digit): 97.19% accuracy

---

## 📊 Dataset

### WCO Harmonized System Nomenclature
- **Source**: World Customs Organization official descriptions
- **Coverage**: 5,612 HS6 codes across 96 chapters and 1,228 headings
- **Augmentation**: ~32 synthetic variations per code using category-specific rules
- **Total Examples**: 179,184 training samples

### Tariff Data
- **Source**: WTO Analytical Database
- **Countries**: Canada, European Union, Switzerland
- **Year**: 2024 (HS22 classification)
- **Coverage**: 16,797 MFN tariff rates

### Data Augmentation Strategy
Context-aware rules for generating paraphrases:
- Synonym substitution (horses ↔ equines, cattle ↔ bovine)
- Prefix/suffix additions ("imported", "for commercial use")
- Simplification (removing parenthetical clauses)
- Domain-specific templates (Live Animals, Food, Textiles, Machinery, Chemicals)

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (recommended for training)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/thesis-clean.git
cd thesis-clean
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download WCO data**
```bash
python src/data/download_wco.py
```

4. **Generate augmented training data**
```bash
python src/data/augment_data.py
```

### Training Models

**Train Baseline Model:**
```bash
python src/models/baseline.py
```

**Train Hierarchical Model:**
```bash
python src/models/train_hierarchical.py
```
- Training time: 2-4 hours on GPU
- Model size: ~250 MB
- Best model saved at epoch with highest validation HS6 accuracy

### Running the Web Application

```bash
cd src/app
streamlit run streamlit_app.py
```

Access at: `http://localhost:8501`

---

## 📁 Repository Structure

```
thesis-clean/
├── src/
│   ├── data/
│   │   ├── download_wco.py          # WCO data scraper
│   │   └── augment_data.py          # Data augmentation script
│   ├── models/
│   │   ├── baseline.py              # Sentence-BERT retrieval model
│   │   └── train_hierarchical.py    # Training script
│   └── app/
│       └── streamlit_app.py         # Web interface
├── data/
│   └── processed/
│       ├── wco_hs_descriptions.csv      # Official HS descriptions
│       └── wto_model_can_eu_che.csv     # Tariff data
├── models/
│   ├── baseline/
│   └── hierarchical/
│       ├── best_model.pt            # Trained model weights (not included due to file size constraints)
│       ├── label_mappings.json      # Class mappings
│       └── training_log.txt         # Training metrics
├── notebooks/
│   └── 02_model_eval.ipynb          # Model evaluation & analysis
├── requirements.txt
└── README.md
```

---

## 🔬 Model Performance

### Hierarchical Classifier Results (Validation Set)

| Level | Classes | Accuracy |
|-------|---------|----------|
| **Chapter (2-digit)** | 96 | 99.77% |
| **Heading (4-digit)** | 1,228 | 98.79% |
| **HS6 (6-digit)** | 5,612 | 97.19% |

### Training Dynamics
- **Convergence**: Best model at epoch 4 of 30
- **Training Loss**: 0.1522 → 0.0537 (epochs 4 → 30)
- **Validation Loss**: 0.0939 → 0.1510 (mild overfitting after epoch 4)

### Generalization Performance
- **Strong on WCO-style descriptions**: Both models near-perfect
- **Moderate on natural language**: Performance degrades on simplified user queries
- **Low confidence on ambiguous inputs**: Model recognizes distribution shift

---

## 💡 Use Cases

1. **Automated Product Classification**: Replace manual HS code lookup
2. **Customs Compliance**: Reduce misclassification penalties
3. **Duty Estimation**: Instant tariff rate lookup for multiple countries
4. **Supply Chain Optimization**: Faster customs clearance through accurate pre-classification
5. **Trade Analytics**: Analyze product portfolios by HS classification

---

## 🛠️ Technical Details

### Baseline Model
- **Framework**: sentence-transformers
- **Encoding**: 384-dimensional dense vectors
- **Index**: FAISS for efficient similarity search
- **Inference**: <50ms per query on CPU

### Hierarchical Model
- **Framework**: PyTorch 2.x + Hugging Face Transformers
- **Optimizer**: AdamW (lr=2e-5, batch_size=32)
- **Schedule**: Linear decay with 500 warmup steps
- **Regularization**: Dropout (p=0.1) after CLS pooling
- **Hardware**: CUDA-enabled GPU with 8GB+ VRAM

---

## ⚠️ Limitations

1. **Data Leakage**: Augmentation applied before train/validation split may inflate validation metrics
2. **Distribution Shift**: Performance degrades on natural language queries vs. technical WCO descriptions
3. **No Ablation Studies**: Optimal loss weights and architecture variants not tested
4. **Limited Countries**: Tariff data only available for Canada, EU, and Switzerland
5. **No NLP Integration**: Regulatory text analysis not implemented (planned future work)

---

## 🔮 Future Work

### Short-term Improvements
- [ ] Implement proper train/test splitting before augmentation
- [ ] Create curated test set of real user queries
- [ ] Add precision, recall, F1 metrics for per-class analysis
- [ ] Expand tariff coverage to more countries
- [ ] Implement hierarchical inference (use chapter to constrain HS6 predictions)

### Long-term Enhancements
- [ ] Fine-tune with domain-specific BERT (LEGAL-BERT)
- [ ] Integrate regulatory text analysis (NLP for compliance documents)
- [ ] Add multi-language support
- [ ] Implement active learning for continuous improvement
- [ ] Deploy as production API with authentication

---

## 📚 Key References

- **World Customs Organization**: [HS Nomenclature](https://www.wcoomd.org/)
- **WTO Tariff Data**: [Analytical Database](https://www.wto.org/)
- **Hugging Face Transformers**: [Documentation](https://huggingface.co/docs/transformers/)
- **DistilBERT Paper**: Sanh et al. (2019) "DistilBERT, a distilled version of BERT"

---

## 📄 License

This project is part of a Master's thesis at Hochschule Luzern.  
For academic or commercial use, please contact the author.

---

## 👤 Author

**Carlos Leon**  
Master of Science in Applied Information and Data Science  
Hochschule Luzern - Wirtschaft

📧 carlos.leon@stud.hslu.ch  
🔗 [LinkedIn](https://www.linkedin.com/in/carlosgleonr/) | [GitHub](https://github.com/cgleonr/)

---

## 🙏 Acknowledgments

- **Supervisor**: Oliver Staubli (Revolytics)
- **Client Partner**: Sofia Viale (On AG)
- **Institution**: Hochschule Luzern - Wirtschaft

Special thanks to the World Customs Organization for making HS nomenclature data publicly available, and to the Hugging Face team for their excellent open-source tools.

---

*Last updated: December 2025*
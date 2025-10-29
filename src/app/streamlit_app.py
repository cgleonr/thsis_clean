"""
Improved Streamlit Web Application
Fixed hierarchical model integration and better error handling
"""

import streamlit as st
import pandas as pd
import sys
import torch
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.baseline import BaselineHSClassifier

# Try to import hierarchical model components
try:
    from transformers import AutoTokenizer, AutoModel
    import torch.nn as nn
    HIERARCHICAL_AVAILABLE = True
except Exception as e:
    logger.warning(f"Could not import hierarchical dependencies: {e}")
    HIERARCHICAL_AVAILABLE = False


# Hierarchical model class
class HierarchicalHSClassifier(nn.Module):
    """Hierarchical neural classifier"""
    
    def __init__(self, base_model_name, num_chapters, num_headings, num_hs6, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.chapter_head = nn.Linear(hidden_size, num_chapters)
        self.heading_head = nn.Linear(hidden_size, num_headings)
        self.hs6_head = nn.Linear(hidden_size, num_hs6)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        chapter_logits = self.chapter_head(pooled)
        heading_logits = self.heading_head(pooled)
        hs6_logits = self.hs6_head(pooled)
        return chapter_logits, heading_logits, hs6_logits


# Page configuration
st.set_page_config(
    page_title="HS Code Classifier",
    page_icon="📦",
    layout="wide"
)

# Title and description
st.title("🌍 Machine Learning-Based HS Code Classification")
st.markdown("""
This tool uses machine learning to automatically classify product descriptions into 
Harmonized System (HS) codes and estimate applicable customs duties.

**Master's Thesis Project** by Carlos Leon | Hochschule Luzern | 2025
""")

st.divider()


@st.cache_resource
def load_baseline_model():
    """Load the baseline model (cached)"""
    try:
        # Try fixed data first
        data_dir_options = ["data/processed", "../../data/processed", "../data/processed"]
        model_dir_options = ["models/baseline", "../../models/baseline", "../models/baseline"]
        
        for data_dir in data_dir_options:
            for model_dir in model_dir_options:
                try:
                    classifier = BaselineHSClassifier(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        data_dir=data_dir,
                        model_dir=model_dir
                    )
                    
                    # Try to load index
                    classifier.load_index()
                    
                    # Try to load tariff data (prefer fixed version)
                    tariff_files = [
                        Path(data_dir) / "wto_tariffs_fixed.csv",
                        Path(data_dir) / "wto_tariffs_clean.csv"
                    ]
                    
                    for tariff_file in tariff_files:
                        if tariff_file.exists():
                            classifier.load_tariff_data(str(tariff_file))
                            logger.info(f"Loaded tariff data from {tariff_file}")
                            break
                    
                    logger.info(f"✅ Baseline model loaded from {data_dir}, {model_dir}")
                    return classifier
                    
                except Exception as e:
                    continue
        
        st.error("❌ Could not load baseline model. Please ensure data is preprocessed.")
        return None
        
    except Exception as e:
        st.error(f"Error loading baseline model: {e}")
        return None


@st.cache_resource
def load_hierarchical_model():
    """Load the hierarchical model (cached)"""
    if not HIERARCHICAL_AVAILABLE:
        return None
    
    try:
        model_dir_options = ["models/hierarchical", "../../models/hierarchical", "../models/hierarchical"]
        
        for model_dir in model_dir_options:
            model_dir_path = Path(model_dir)
            model_file = model_dir_path / "best_model.pt"
            mappings_file = model_dir_path / "label_mappings.json"
            
            if not model_file.exists() or not mappings_file.exists():
                continue
            
            try:
                # Load mappings
                with open(mappings_file, 'r') as f:
                    mappings = json.load(f)
                
                # Determine which BERT model was used
                checkpoint = torch.load(model_file, map_location='cpu')
                base_model = checkpoint.get('model_config', {}).get('base_model_name', 'prajjwal1/bert-tiny')
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(base_model)
                
                # Initialize model
                model = HierarchicalHSClassifier(
                    base_model_name=base_model,
                    num_chapters=len(mappings['chapter_to_idx']),
                    num_headings=len(mappings['heading_to_idx']),
                    num_hs6=len(mappings['hs6_to_idx'])
                )
                
                # Load weights
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                logger.info(f"✅ Hierarchical model loaded from {model_dir}")
                
                return {
                    'model': model,
                    'tokenizer': tokenizer,
                    'mappings': mappings,
                    'base_model': base_model
                }
                
            except Exception as e:
                logger.error(f"Error loading from {model_dir}: {e}")
                continue
        
        logger.warning("⚠️ Hierarchical model not found. Train it first.")
        return None
        
    except Exception as e:
        logger.error(f"Error loading hierarchical model: {e}")
        return None


def predict_hierarchical(query_text, hierarchical_dict, baseline_classifier, reporter_name, top_k=3):
    """Make predictions with hierarchical model and add tariff data"""
    
    model = hierarchical_dict['model']
    tokenizer = hierarchical_dict['tokenizer']
    mappings = hierarchical_dict['mappings']
    
    # Tokenize
    encoding = tokenizer(
        query_text.lower().strip(),
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    # Predict
    with torch.no_grad():
        chapter_logits, heading_logits, hs6_logits = model(
            encoding['input_ids'],
            encoding['attention_mask']
        )
    
    # Get top-K predictions for HS6
    hs6_probs = torch.softmax(hs6_logits, dim=1)[0]
    top_k_actual = min(top_k, len(hs6_probs))
    top_k_probs, top_k_indices = torch.topk(hs6_probs, top_k_actual)
    
    # Also get chapter and heading predictions
    chapter_probs = torch.softmax(chapter_logits, dim=1)[0]
    heading_probs = torch.softmax(heading_logits, dim=1)[0]
    
    # Convert to readable codes
    idx_to_hs6 = mappings['idx_to_hs6']
    idx_to_chapter = mappings['idx_to_chapter']
    idx_to_heading = mappings['idx_to_heading']
    
    results = []
    for i, (prob, idx) in enumerate(zip(top_k_probs, top_k_indices)):
        hs6_code = idx_to_hs6[str(idx.item())]
        chapter_code = hs6_code[:2]
        heading_code = hs6_code[:4]
        
        # Get description from baseline's data
        description = "Hierarchical model prediction"
        if baseline_classifier and baseline_classifier.hs_descriptions is not None:
            desc_match = baseline_classifier.hs_descriptions[
                baseline_classifier.hs_descriptions['hs6'] == hs6_code
            ]
            if len(desc_match) > 0:
                description = desc_match.iloc[0]['description']
        
        # Get tariff
        year, rate = None, None
        if baseline_classifier:
            year, rate = baseline_classifier.get_tariff(reporter_name, hs6_code)
        
        results.append({
            'rank': i + 1,
            'hs6': hs6_code,
            'chapter': chapter_code,
            'heading': heading_code,
            'description': description,
            'similarity': float(prob.item()),
            'model_type': 'hierarchical',
            'reporter': reporter_name,
            'tariff_year': year,
            'mfn_rate_percent': rate
        })
    
    return pd.DataFrame(results)


# Load models
baseline_classifier = load_baseline_model()
hierarchical_dict = load_hierarchical_model()

# Sidebar
st.sidebar.header("⚙️ Settings")

# Country selection
country_options = {
    "European Union": "European Union",
    "Canada": "Canada",
    "Switzerland": "Switzerland"
}
selected_country = st.sidebar.selectbox(
    "Select Country",
    options=list(country_options.keys()),
    help="Choose the country for tariff lookup"
)

# Top-K selection
top_k = st.sidebar.slider(
    "Number of Predictions",
    min_value=1,
    max_value=10,
    value=3,
    help="Number of HS code predictions to display"
)

# Model selection
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Selection")

model_options = []
if baseline_classifier is not None:
    model_options.append("Baseline (Sentence-BERT)")

if hierarchical_dict is not None:
    model_options.append("Hierarchical (Neural Network)")
elif HIERARCHICAL_AVAILABLE:
    st.sidebar.warning("⚠️ Hierarchical model not trained yet")
else:
    st.sidebar.warning("⚠️ Hierarchical dependencies not available")

if not model_options:
    st.sidebar.error("❌ No models available!")
    model_type = None
else:
    model_type = st.sidebar.radio(
        "Select Model",
        options=model_options,
        help="Choose the classification model"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 About
This system uses:
- **Baseline**: Sentence-BERT semantic retrieval
- **Hierarchical**: Custom neural network exploiting HS taxonomy
- **Data**: WTO ADB tariffs + WCO descriptions

**Thesis**: Machine Learning-Based Estimation and Analysis of 
Customs Duties and Compliance Requirements
""")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Product Description")
    
    # Text input
    product_description = st.text_area(
        "Enter a product description:",
        height=100,
        placeholder="e.g., leather handbag with shoulder strap, smartphone with touchscreen, coffee beans roasted...",
        help="Describe the product you want to classify"
    )
    
    # Example buttons
    st.markdown("**Quick Examples:**")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    with col_ex1:
        if st.button("👜 Leather Handbag"):
            product_description = "leather handbag with shoulder strap"
            st.rerun()
    
    with col_ex2:
        if st.button("👕 Cotton T-Shirt"):
            product_description = "cotton t-shirt for men, short sleeves"
            st.rerun()
    
    with col_ex3:
        if st.button("📱 Smartphone"):
            product_description = "smartphone with touchscreen and camera"
            st.rerun()
    
    # Additional examples
    col_ex4, col_ex5, col_ex6 = st.columns(3)
    
    with col_ex4:
        if st.button("☕ Coffee Beans"):
            product_description = "roasted coffee beans, arabica"
            st.rerun()
    
    with col_ex5:
        if st.button("👞 Leather Shoes"):
            product_description = "leather shoes for women, high heels"
            st.rerun()
    
    with col_ex6:
        if st.button("📚 Printed Book"):
            product_description = "printed book, hardcover, educational"
            st.rerun()
    
    # Classify button
    classify_button = st.button("🔍 Classify Product", type="primary", use_container_width=True)

with col2:
    st.header("ℹ️ Information")
    
    if model_type and "Hierarchical" in model_type:
        st.success("""
        **Hierarchical Model Active!**
        
        This model uses a custom neural network that:
        - Predicts at 3 levels (Chapter → Heading → HS6)
        - Exploits HS code taxonomy structure
        - Learns from training data patterns
        
        *This is the main academic contribution of the thesis.*
        """)
    elif model_type and "Baseline" in model_type:
        st.info("""
        **Baseline Model Active**
        
        Uses semantic similarity to find matching HS codes.
        
        - Fast and interpretable
        - No training required
        - Good baseline performance
        - Similarity scores show match quality
        """)
    else:
        st.warning("""
        **No Model Selected**
        
        Please train the models first or check data preprocessing.
        """)

# Results section
if classify_button and product_description:
    if model_type is None:
        st.error("❌ No models available. Please check your data and model setup.")
    else:
        st.divider()
        st.header("🎯 Classification Results")
        
        # Determine which model to use
        use_hierarchical = "Hierarchical" in model_type and hierarchical_dict is not None
        use_baseline = "Baseline" in model_type and baseline_classifier is not None
        
        with st.spinner("Classifying..."):
            try:
                # Get predictions
                if use_hierarchical:
                    st.info("🧠 Using Hierarchical Neural Network")
                    results = predict_hierarchical(
                        product_description,
                        hierarchical_dict,
                        baseline_classifier,
                        country_options[selected_country],
                        top_k=top_k
                    )
                    
                elif use_baseline:
                    st.info("🔍 Using Baseline (Sentence-BERT)")
                    results = baseline_classifier.predict(
                        query_text=product_description,
                        reporter_name=country_options[selected_country],
                        top_k=top_k
                    )
                    results['model_type'] = 'baseline'
                
                else:
                    st.error("❌ Selected model not available")
                    st.stop()
                
                # Display results
                if len(results) == 0:
                    st.warning("⚠️ No predictions generated. Try a different query.")
                else:
                    st.success(f"Found {len(results)} matching HS codes!")
                    
                    # Display each prediction
                    for idx, row in results.iterrows():
                        with st.container():
                            col_rank, col_info, col_tariff = st.columns([1, 5, 2])
                            
                            with col_rank:
                                st.markdown(f"### #{row['rank']}")
                                confidence_pct = row['similarity'] * 100
                                
                                # Color code confidence
                                if confidence_pct >= 50:
                                    color = "normal"
                                elif confidence_pct >= 30:
                                    color = "normal"
                                else:
                                    color = "off"
                                
                                st.metric(
                                    "Confidence",
                                    f"{confidence_pct:.1f}%",
                                    delta=None
                                )
                                
                                # Show quality indicator
                                if confidence_pct >= 50:
                                    st.success("High")
                                elif confidence_pct >= 30:
                                    st.info("Medium")
                                else:
                                    st.warning("Low")
                            
                            with col_info:
                                st.markdown(f"**HS Code:** `{row['hs6']}`")
                                st.markdown(f"**Chapter:** {row['chapter']} | **Heading:** {row['heading']}")
                                
                                if 'description' in row and pd.notna(row['description']):
                                    desc = str(row['description'])
                                    if len(desc) > 150:
                                        st.markdown(f"**Description:** {desc[:150]}...")
                                    else:
                                        st.markdown(f"**Description:** {desc}")
                                
                                # Show model type
                                if 'model_type' in row:
                                    model_badge = "🧠 Hierarchical" if row['model_type'] == 'hierarchical' else "🔍 Baseline"
                                    st.caption(f"Model: {model_badge}")
                            
                            with col_tariff:
                                if 'mfn_rate_percent' in row and pd.notna(row['mfn_rate_percent']):
                                    st.metric(
                                        "MFN Tariff Rate",
                                        f"{row['mfn_rate_percent']:.2f}%",
                                        help=f"Based on {int(row['tariff_year'])} data for {row['reporter']}" if pd.notna(row.get('tariff_year')) else ""
                                    )
                                else:
                                    st.metric(
                                        "MFN Tariff Rate",
                                        "N/A",
                                        help="No tariff data available for this HS code and country"
                                    )
                            
                            st.divider()
                    
                    # Download results
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"hs_classification_results_{selected_country.lower().replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"❌ Error during classification: {str(e)}")
                st.exception(e)
                logger.error(f"Classification error: {e}", exc_info=True)

elif classify_button and not product_description:
    st.warning("⚠️ Please enter a product description first")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Master's Thesis Project - Hochschule Luzern 2025</p>
    <p>Student: Carlos Leon | Supervisor: Oliver Staubli | Client: On AG</p>
</div>
""", unsafe_allow_html=True)

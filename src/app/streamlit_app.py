"""
Streamlit Web Application - FIXED BUTTONS
Example buttons now properly populate the text input
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
    """Load the baseline model"""
    try:
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
                    
                    classifier.load_index()
                    
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
        model_dir_options = ["data/models/hierarchical", "models/hierarchical", "../../data/models/hierarchical", "../data/models/hierarchical"]
        
        for model_dir in model_dir_options:
            model_dir_path = Path(model_dir)
            model_file = model_dir_path / "best_model.pt"
            mappings_file = model_dir_path / "label_mappings.json"
            
            if not model_file.exists() or not mappings_file.exists():
                continue
            
            try:
                with open(mappings_file, 'r') as f:
                    mappings = json.load(f)
                
                checkpoint = torch.load(model_file, map_location='cpu')
                
                # Try multiple keys for base model name
                if 'model_config' in checkpoint:
                    config = checkpoint['model_config']
                    base_model = config.get('base_model') or config.get('base_model_name', 'distilbert-base-uncased')
                else:
                    base_model = 'distilbert-base-uncased'
                
                logger.info(f"Loading model with base: {base_model}")
                
                tokenizer = AutoTokenizer.from_pretrained(base_model)
                
                model = HierarchicalHSClassifier(
                    base_model_name=base_model,
                    num_chapters=len(mappings['chapter_to_idx']),
                    num_headings=len(mappings['heading_to_idx']),
                    num_hs6=len(mappings['hs6_to_idx'])
                )
                
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
    
    encoding = tokenizer(
        query_text.lower().strip(),
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        chapter_logits, heading_logits, hs6_logits = model(
            encoding['input_ids'],
            encoding['attention_mask']
        )
    
    hs6_probs = torch.softmax(hs6_logits, dim=1)[0]
    top_k_actual = min(top_k, len(hs6_probs))
    top_k_probs, top_k_indices = torch.topk(hs6_probs, top_k_actual)
    
    idx_to_hs6 = {v: k for k, v in mappings['hs6_to_idx'].items()}
    
    predictions = []
    for rank, (prob, idx) in enumerate(zip(top_k_probs, top_k_indices), 1):
        hs6_code = idx_to_hs6[idx.item()]
        
        predictions.append({
            'rank': rank,
            'hs6': hs6_code,
            'chapter': hs6_code[:2],
            'heading': hs6_code[:4],
            'similarity': prob.item(),
            'model_type': 'hierarchical'
        })
    
    results_df = pd.DataFrame(predictions)
    
    # Load HS descriptions from WCO file
    try:
        hs_desc_paths = [
            'data/processed/wco_hs_descriptions.csv',
            'data/processed/wco_hs_descriptions_clean.csv',
            '../../data/processed/wco_hs_descriptions.csv',
            '../data/processed/wco_hs_descriptions.csv'
        ]
        
        hs_desc_df = None
        for path in hs_desc_paths:
            if Path(path).exists():
                hs_desc_df = pd.read_csv(path)
                hs_desc_df['hs6'] = hs_desc_df['hs6'].astype(str).str.zfill(6)
                break
        
        if hs_desc_df is not None:
            hs_desc_map = dict(zip(hs_desc_df['hs6'], hs_desc_df['description']))
            results_df['description'] = results_df['hs6'].map(hs_desc_map)
        else:
            logger.warning("WCO HS descriptions file not found")
    except Exception as e:
        logger.warning(f"Could not load HS descriptions: {e}")
    
    # Try to add tariff data if available
    if baseline_classifier and hasattr(baseline_classifier, 'tariff_data') and baseline_classifier.tariff_data is not None:
        tariff_df = baseline_classifier.tariff_data
        tariff_df['hs6_str'] = tariff_df['hs6'].astype(str).str.zfill(6)
        
        tariff_subset = tariff_df[tariff_df['reporter'] == reporter_name].copy()
        
        if len(tariff_subset) > 0:
            latest_year = tariff_subset['year'].max()
            tariff_latest = tariff_subset[tariff_subset['year'] == latest_year]
            
            tariff_map = dict(zip(tariff_latest['hs6_str'], tariff_latest['mfn_rate_percent']))
            year_map = dict(zip(tariff_latest['hs6_str'], tariff_latest['year']))
            reporter_map = dict(zip(tariff_latest['hs6_str'], tariff_latest['reporter']))
            
            results_df['mfn_rate_percent'] = results_df['hs6'].map(tariff_map)
            results_df['tariff_year'] = results_df['hs6'].map(year_map)
            results_df['reporter'] = results_df['hs6'].map(reporter_map)
    
    return results_df


# Load models
baseline_classifier = load_baseline_model()
hierarchical_dict = load_hierarchical_model()

# Determine available models
available_models = []
if hierarchical_dict:
    available_models.append("Hierarchical Neural Network (Thesis Contribution)")
if baseline_classifier:
    available_models.append("Baseline (Sentence-BERT)")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    if available_models:
        # Calculate valid index
        if hierarchical_dict:
            default_idx = 0
        elif baseline_classifier:
            default_idx = 0
        else:
            default_idx = 0
            
        model_type = st.selectbox(
            "Select Model",
            available_models,
            index=default_idx,
            help="Choose which model to use for classification"
        )
    else:
        model_type = None
        st.error("❌ No models available")
    
    country_options = {
        "European Union": "European Union",
        "Canada": "Canada", 
        "Switzerland": "Switzerland"
    }
    
    selected_country = st.selectbox(
        "Country/Region for Tariff Data",
        list(country_options.keys()),
        help="Select the destination country for customs duty rates"
    )
    
    top_k = st.slider(
        "Number of Results",
        min_value=1,
        max_value=10,
        value=5,
        help="How many top predictions to show"
    )
    
    st.divider()
    
    st.markdown("### 📊 Model Status")
    if hierarchical_dict:
        st.success("✅ Hierarchical Model Ready")
    else:
        st.warning("⚠️ Hierarchical Model Not Found")
    
    if baseline_classifier:
        st.success("✅ Baseline Model Ready")
    else:
        st.warning("⚠️ Baseline Model Not Found")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Product Description")
    
    # Text input
    product_description = st.text_area(
        "Enter Product Description",
        value="",
        height=100,
        placeholder="e.g., 'smartphone with 5G connectivity', 'laptop computer', 'leather shoes', 'cotton t-shirt', 'green coffee beans', 'red wine'",
        help="Describe the product you want to classify"
    )
    
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
        
        use_hierarchical = "Hierarchical" in model_type and hierarchical_dict is not None
        use_baseline = "Baseline" in model_type and baseline_classifier is not None
        
        with st.spinner("Classifying..."):
            try:
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
                
                if len(results) == 0:
                    st.warning("⚠️ No predictions generated. Try a different query.")
                else:
                    st.success(f"Found {len(results)} matching HS codes!")
                    
                    for idx, row in results.iterrows():
                        with st.container():
                            col_rank, col_info, col_tariff = st.columns([1, 5, 2])
                            
                            with col_rank:
                                st.markdown(f"### #{row['rank']}")
                                confidence_pct = row['similarity'] * 100
                                
                                st.metric(
                                    "Confidence",
                                    f"{confidence_pct:.1f}%",
                                    delta=None
                                )
                                
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
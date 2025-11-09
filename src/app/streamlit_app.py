"""
Streamlit Web Application with Tariff Integration
Displays MFN tariff rates for predicted HS codes
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


# Page config
st.set_page_config(
    page_title="HS Code Classification",
    page_icon="🌍",
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
def load_tariff_data():
    """Load tariff lookup table"""
    try:
        data_dir_options = ["data/processed", "../../data/processed", "../data/processed"]
        
        for data_dir in data_dir_options:
            tariff_file = Path(data_dir) / "wto_model_can_eu_che.csv"
            if tariff_file.exists():
                df = pd.read_csv(tariff_file, dtype={'hs6': str})
                # Ensure HS6 is 6 digits
                df['hs6'] = df['hs6'].astype(str).str.zfill(6)
                logger.info(f"✅ Loaded tariff data: {len(df)} records")
                return df
        
        logger.warning("⚠️ Tariff file not found")
        return None
        
    except Exception as e:
        logger.error(f"Error loading tariff data: {e}")
        return None


def get_tariff_info(hs6_code, tariff_df, country=None):
    """
    Get tariff information for an HS6 code
    
    Args:
        hs6_code: 6-digit HS code (string)
        tariff_df: DataFrame with tariff data
        country: Specific country or None for all
    
    Returns:
        DataFrame with tariff info or None
    """
    if tariff_df is None:
        return None
    
    # Ensure HS6 is 6 digits
    hs6_code = str(hs6_code).zfill(6)
    
    # Filter by HS6 code
    result = tariff_df[tariff_df['hs6'] == hs6_code].copy()
    
    if country:
        result = result[result['reporter_name'] == country]
    
    if len(result) == 0:
        return None
    
    return result[['reporter_name', 'year', 'mfn_rate_percent']].sort_values('reporter_name')


@st.cache_resource
def load_baseline_model():
    """Load the baseline model (cached)"""
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
                    logger.info(f"✅ Baseline model loaded from {data_dir}, {model_dir}")
                    return classifier
                    
                except Exception as e:
                    continue
        
        st.error("❌ Could not load baseline model")
        return None
        
    except Exception as e:
        st.error(f"Error loading baseline model: {e}")
        return None


@st.cache_resource
def load_hierarchical_model():
    """Load the hierarchical model (cached)"""
    if not HIERARCHICAL_AVAILABLE:
        return None, None, None
    
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
                
                checkpoint = torch.load(model_file, map_location='cpu')
                base_model = checkpoint.get('model_config', {}).get('base_model', 'distilbert-base-uncased')
                
                logger.info(f"Loading model with base: {base_model}")
                
                tokenizer = AutoTokenizer.from_pretrained(base_model)
                
                model = HierarchicalHSClassifier(
                    base_model,
                    num_chapters=len(mappings['chapter_to_idx']),
                    num_headings=len(mappings['heading_to_idx']),
                    num_hs6=len(mappings['hs6_to_idx']),
                    dropout=0.1
                )
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                logger.info(f"✅ Hierarchical model loaded from {model_dir}")
                return model, tokenizer, mappings
                
            except Exception as e:
                logger.error(f"Error loading from {model_dir}: {e}")
                continue
        
        logger.warning("⚠️ Hierarchical model not found")
        return None, None, None
        
    except Exception as e:
        logger.error(f"Error in load_hierarchical_model: {e}")
        return None, None, None


# Load models and tariff data
baseline_classifier = load_baseline_model()
hierarchical_model, tokenizer, mappings = load_hierarchical_model()
tariff_df = load_tariff_data()

# Show model status
st.sidebar.header("📊 Model Status")
if baseline_classifier:
    st.sidebar.success("✅ Baseline Model")
else:
    st.sidebar.error("❌ Baseline Model")

if hierarchical_model:
    st.sidebar.success("✅ Hierarchical Model")
else:
    st.sidebar.warning("⚠️ Hierarchical Model Not Available")

if tariff_df is not None:
    st.sidebar.success(f"✅ Tariff Data ({len(tariff_df)} records)")
    st.sidebar.info(f"Countries: {', '.join(tariff_df['reporter_name'].unique())}")
else:
    st.sidebar.warning("⚠️ Tariff Data Not Available")

st.sidebar.divider()

# Model selection
st.sidebar.header("⚙️ Settings")
available_models = []
if baseline_classifier:
    available_models.append("Baseline (Sentence-BERT)")
if hierarchical_model:
    available_models.append("Hierarchical (DistilBERT)")

if not available_models:
    st.error("No models available. Please check your setup.")
    st.stop()

selected_model = st.sidebar.radio("Select Model:", available_models)
top_k = st.sidebar.slider("Number of predictions:", 1, 10, 5)

# Country selection for tariff display
if tariff_df is not None:
    countries = ["All"] + sorted(tariff_df['reporter_name'].unique().tolist())
    selected_country = st.sidebar.selectbox("Tariff Country:", countries)
    if selected_country == "All":
        selected_country = None
else:
    selected_country = None

st.sidebar.divider()

# Main input area
st.header("🔍 Product Description Input")

query = st.text_area(
    "Enter product description:",
    placeholder="e.g., 'cotton t-shirt for men' or 'smartphone with 5G connectivity'",
    height=100
)

classify_button = st.button("🚀 Classify Product", type="primary", use_container_width=True)

st.divider()

# Prediction logic
if classify_button and query.strip():
    with st.spinner("Classifying..."):
        if selected_model == "Baseline (Sentence-BERT)" and baseline_classifier:
            try:
                predictions = baseline_classifier.predict(query, top_k=top_k)
                
                st.success(f"✅ Classification Complete ({selected_model})")
                
                st.subheader("📋 Top Predictions")
                
                for idx, row in predictions.iterrows():
                    with st.expander(f"**#{row['rank']} - HS6 {row['hs6']}** (Similarity: {row['similarity']:.3f})", expanded=(idx==0)):
                        col_desc, col_tariff = st.columns([2, 1])
                        
                        with col_desc:
                            st.markdown(f"**Description:**")
                            st.write(row['description'])
                            st.markdown(f"**Chapter:** {row['chapter']} | **Heading:** {row['heading']}")
                        
                        with col_tariff:
                            if tariff_df is not None:
                                tariff_info = get_tariff_info(row['hs6'], tariff_df, selected_country)
                                if tariff_info is not None and len(tariff_info) > 0:
                                    st.markdown("**📊 MFN Tariff Rates:**")
                                    for _, t_row in tariff_info.iterrows():
                                        st.metric(
                                            label=t_row['reporter_name'],
                                            value=f"{t_row['mfn_rate_percent']:.2f}%",
                                            help=f"Year: {t_row['year']}"
                                        )
                                else:
                                    st.info("No tariff data available")
                            else:
                                st.info("Tariff data not loaded")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        
        elif selected_model == "Hierarchical (DistilBERT)" and hierarchical_model:
            try:
                # Tokenize
                encoding = tokenizer(
                    query.lower().strip(),
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                
                # Predict
                with torch.no_grad():
                    chapter_logits, heading_logits, hs6_logits = hierarchical_model(
                        encoding['input_ids'],
                        encoding['attention_mask']
                    )
                
                # Get top-K HS6 predictions
                hs6_probs = torch.softmax(hs6_logits, dim=1)[0]
                top_k_actual = min(top_k, len(hs6_probs))
                top_probs, top_indices = torch.topk(hs6_probs, top_k_actual)
                
                # Create reverse mappings
                idx_to_hs6 = {v: k for k, v in mappings['hs6_to_idx'].items()}
                
                # Get descriptions
                desc_df = None
                desc_file_options = [
                    "data/processed/wco_hs_descriptions.csv",
                    "../../data/processed/wco_hs_descriptions.csv",
                    "../data/processed/wco_hs_descriptions.csv"
                ]
                for desc_file in desc_file_options:
                    if Path(desc_file).exists():
                        desc_df = pd.read_csv(desc_file, dtype={'hs6': str})
                        desc_df['hs6'] = desc_df['hs6'].str.zfill(6)
                        break
                
                st.success(f"✅ Classification Complete ({selected_model})")
                st.subheader("📋 Top Predictions")
                
                for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
                    hs6_code = idx_to_hs6[idx.item()]
                    confidence = prob.item()
                    
                    # Get description
                    if desc_df is not None:
                        desc_match = desc_df[desc_df['hs6'] == hs6_code]
                        description = desc_match['description'].iloc[0] if len(desc_match) > 0 else "No description available"
                    else:
                        description = "Description file not found"
                    
                    with st.expander(f"**#{rank} - HS6 {hs6_code}** (Confidence: {confidence:.3f})", expanded=(rank==1)):
                        col_desc, col_tariff = st.columns([2, 1])
                        
                        with col_desc:
                            st.markdown(f"**Description:**")
                            st.write(description)
                            st.markdown(f"**Chapter:** {hs6_code[:2]} | **Heading:** {hs6_code[:4]}")
                        
                        with col_tariff:
                            if tariff_df is not None:
                                tariff_info = get_tariff_info(hs6_code, tariff_df, selected_country)
                                if tariff_info is not None and len(tariff_info) > 0:
                                    st.markdown("**📊 MFN Tariff Rates:**")
                                    for _, t_row in tariff_info.iterrows():
                                        st.metric(
                                            label=t_row['reporter_name'],
                                            value=f"{t_row['mfn_rate_percent']:.2f}%",
                                            help=f"Year: {t_row['year']}"
                                        )
                                else:
                                    st.info("No tariff data available")
                            else:
                                st.info("Tariff data not loaded")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                import traceback
                st.code(traceback.format_exc())

elif classify_button:
    st.warning("⚠️ Please enter a product description")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <small>
    Master's Thesis Project - Machine Learning-Based HS Code Classification<br>
    Carlos Leon | Hochschule Luzern | 2025
    </small>
</div>
""", unsafe_allow_html=True)
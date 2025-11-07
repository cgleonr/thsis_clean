"""
Baseline Model: Sentence-BERT Semantic Retrieval
Uses pre-trained sentence embeddings for HS code classification
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import pickle
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineHSClassifier:
    """
    Baseline HS6 classifier using Sentence-BERT embeddings
    and cosine similarity retrieval
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        data_dir: str = "data/processed",
        model_dir: str = "models/baseline"
    ):
        """
        Initialize classifier
        
        Args:
            model_name: Name of Sentence-BERT model
            data_dir: Directory containing processed data
            model_dir: Directory to save/load model artifacts
        """
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Sentence-BERT model
        logger.info(f"Loading Sentence-BERT model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        
        # Placeholders for embeddings and metadata
        self.hs_descriptions = None
        self.hs_embeddings = None
        self.tariff_lookup = None
    
    def build_index(self, descriptions_file: Optional[str] = None):
        """
        Build embedding index from HS descriptions
        
        Args:
            descriptions_file: Path to HS descriptions CSV
        """
        if descriptions_file is None:
            descriptions_file = self.data_dir / "wco_hs_descriptions_clean.csv"
        else:
            descriptions_file = Path(descriptions_file)
        
        if not descriptions_file.exists():
            raise FileNotFoundError(f"Descriptions file not found: {descriptions_file}")
        
        logger.info(f"Loading HS descriptions from {descriptions_file}")
        self.hs_descriptions = pd.read_csv(descriptions_file, dtype={'hs6': str})
        
        logger.info(f"Loaded {len(self.hs_descriptions)} descriptions")
        logger.info(f"Unique HS6 codes: {self.hs_descriptions['hs6'].nunique()}")
        
        # Encode descriptions
        logger.info("Encoding descriptions...")
        texts = self.hs_descriptions['description'].tolist()
        self.hs_embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        logger.info(f"Created embeddings of shape {self.hs_embeddings.shape}")
        
        # Save embeddings
        embeddings_file = self.model_dir / "hs_embeddings.npy"
        metadata_file = self.model_dir / "hs_metadata.csv"
        
        np.save(embeddings_file, self.hs_embeddings)
        self.hs_descriptions.to_csv(metadata_file, index=False)
        
        logger.info(f"Saved embeddings to {embeddings_file}")
        logger.info(f"Saved metadata to {metadata_file}")
    
    def load_index(self):
        """Load pre-computed embedding index"""
        embeddings_file = self.model_dir / "hs_embeddings.npy"
        metadata_file = self.model_dir / "hs_metadata.csv"
        
        if not embeddings_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(
                "Embedding index not found. Run build_index() first."
            )
        
        logger.info("Loading embedding index...")
        self.hs_embeddings = np.load(embeddings_file)
        self.hs_descriptions = pd.read_csv(metadata_file, dtype={'hs6': str})
        
        logger.info(f"Loaded {len(self.hs_descriptions)} descriptions")
    
    def load_tariff_data(self, tariffs_file: Optional[str] = None):
        """
        Load tariff lookup table
        
        Args:
            tariffs_file: Path to tariffs CSV
        """
        if tariffs_file is None:
            tariffs_file = self.data_dir / "wto_tariffs_clean.csv"
        else:
            tariffs_file = Path(tariffs_file)
        
        if not tariffs_file.exists():
            logger.warning(f"Tariffs file not found: {tariffs_file}")
            return
        
        logger.info(f"Loading tariffs from {tariffs_file}")
        df_tariffs = pd.read_csv(tariffs_file, dtype={'hs6': str})
        
        # Get latest tariff for each (reporter, hs6)
        df_tariffs['year'] = pd.to_numeric(df_tariffs['year'], errors='coerce')
        df_tariffs = df_tariffs.sort_values(
            ['reporter_name', 'hs6', 'year'],
            ascending=[True, True, False]
        )
        df_latest = df_tariffs.drop_duplicates(
            subset=['reporter_name', 'hs6'],
            keep='first'
        )
        
        # Create lookup dictionary
        self.tariff_lookup = df_latest.set_index(
            ['reporter_name', 'hs6']
        )[['year', 'mfn_rate_percent']].to_dict('index')
        
        logger.info(f"Loaded {len(self.tariff_lookup)} tariff records")
    
    def get_tariff(
        self,
        reporter_name: str,
        hs6_code: str
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Get tariff for a country and HS6 code
        
        Args:
            reporter_name: Country name (e.g., "European Union")
            hs6_code: 6-digit HS code
        
        Returns:
            Tuple of (year, tariff_rate_percent) or (None, None)
        """
        if self.tariff_lookup is None:
            return None, None
        
        key = (reporter_name, hs6_code)
        if key in self.tariff_lookup:
            data = self.tariff_lookup[key]
            return int(data['year']), float(data['mfn_rate_percent'])
        
        return None, None
    
    def predict(
        self,
        query_text: str,
        reporter_name: str = "European Union",
        top_k: int = 3
    ) -> pd.DataFrame:
        """
        Predict top-K HS6 codes for a product description
        
        Args:
            query_text: Product description
            reporter_name: Country for tariff lookup
            top_k: Number of predictions to return
        
        Returns:
            DataFrame with predictions, similarities, and tariffs
        """
        if self.hs_embeddings is None or self.hs_descriptions is None:
            raise ValueError("Index not loaded. Run load_index() first.")
        
        # Encode query
        query_embedding = self.encoder.encode(
            [query_text.lower().strip()],
            normalize_embeddings=True
        )
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.hs_embeddings)[0]
        
        # Get top-K indices
        if top_k >= len(similarities):
            top_indices = np.argsort(-similarities)
        else:
            top_indices = np.argpartition(-similarities, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]
        
        # Build results
        results = []
        for idx in top_indices:
            hs6 = self.hs_descriptions.iloc[idx]['hs6']
            description = self.hs_descriptions.iloc[idx]['description']
            similarity = float(similarities[idx])
            
            # Get tariff
            year, rate = self.get_tariff(reporter_name, hs6)
            
            results.append({
                'rank': len(results) + 1,
                'hs6': hs6,
                'chapter': hs6[:2],
                'heading': hs6[:4],
                'description': description,
                'similarity': round(similarity, 4),
                'reporter': reporter_name,
                'tariff_year': year,
                'mfn_rate_percent': rate
            })
        
        return pd.DataFrame(results)
    
    def save_model(self):
        """Save model configuration"""
        config = {
            'model_name': self.model_name,
            'embeddings_shape': self.hs_embeddings.shape if self.hs_embeddings is not None else None,
            'num_hs_codes': len(self.hs_descriptions) if self.hs_descriptions is not None else 0
        }
        
        config_file = self.model_dir / "config.pkl"
        with open(config_file, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"Saved model config to {config_file}")


def demo():
    """Run demonstration of baseline model"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train baseline HS classifier')
    parser.add_argument('--data', type=str, default='data/processed/wco_hs_descriptions.csv',
                        help='Path to HS descriptions file')
    parser.add_argument('--tariffs', type=str, default='data/processed/wto_tariffs_fixed.csv',
                        help='Path to tariffs file')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("BASELINE MODEL TRAINING")
    logger.info("=" * 60)
    
    # Initialize classifier
    classifier = BaselineHSClassifier(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        data_dir="data/processed",
        model_dir="models/baseline"
    )
    
    # Build index from specified data file
    logger.info(f"Building index from: {args.data}")
    classifier.build_index(descriptions_file=args.data)
    
    # Load tariff data
    logger.info(f"Loading tariffs from: {args.tariffs}")
    classifier.load_tariff_data(tariffs_file=args.tariffs)
    
    # Save model
    classifier.save_model()
    
    # Test predictions
    logger.info("\n" + "=" * 60)
    logger.info("TEST PREDICTIONS")
    logger.info("=" * 60)
    
    test_queries = [
        "leather handbag with shoulder strap",
        "cotton t-shirt for men",
        "smartphone with touchscreen",
        "coffee beans, roasted",
        "wooden dining table",
        "bovine animals for breeding",
        "equines for commercial use"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        logger.info("-" * 60)
        
        results = classifier.predict(
            query,
            reporter_name="European Union",
            top_k=3
        )
        
        for _, row in results.iterrows():
            logger.info(
                f"  {row['rank']}. HS6 {row['hs6']} | "
                f"Sim: {row['similarity']:.4f} | "
                f"Tariff: {row['mfn_rate_percent']}%"
            )
            logger.info(f"     {row['description'][:80]}...")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✓ Baseline model trained on {len(classifier.hs_descriptions)} HS codes")
    logger.info(f"✓ Model saved to: models/baseline")
    logger.info("=" * 60)


if __name__ == "__main__":
    demo()
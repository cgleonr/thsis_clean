"""
Data Preprocessing Module - FIXED VERSION
Cleans, normalizes, and harmonizes customs datasets
Automatically detects column names in your data files
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles preprocessing of customs datasets with flexible column detection"""
    
    def __init__(
        self, 
        raw_data_dir: str = "data/raw",
        processed_data_dir: str = "data/processed"
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def detect_columns(self, df: pd.DataFrame, col_type: str) -> Optional[str]:
        """
        Automatically detect column names based on keywords
        
        Args:
            df: DataFrame to search
            col_type: Type of column ('hs_code', 'description', 'tariff', etc.)
        
        Returns:
            Detected column name or None
        """
        col_keywords = {
            'hs_code': ['code', 'hs', 'tariff_code', 'classification', 'commodity'],
            'description': ['description', 'desc', 'text', 'label', 'name', 'commodity_description'],
            'tariff': ['tariff', 'duty', 'rate', 'mfn', 'value'],
            'reporter': ['reporter', 'country', 'nation', 'economy'],
            'year': ['year', 'period', 'date']
        }
        
        keywords = col_keywords.get(col_type, [])
        
        for col in df.columns:
            col_lower = col.lower().strip()
            for keyword in keywords:
                if keyword in col_lower:
                    logger.info(f"  Detected {col_type} column: '{col}'")
                    return col
        
        return None
    
    def normalize_hs_code(self, code: str) -> Optional[str]:
        """
        Normalize HS code to 6-digit format
        
        Args:
            code: Raw HS code string
        
        Returns:
            6-digit zero-padded HS code or None
        """
        if pd.isna(code):
            return None
        
        # Convert to string and extract digits only
        digits = ''.join(c for c in str(code) if c.isdigit())
        
        if not digits or len(digits) < 2:
            return None
        
        # Pad or truncate to 6 digits
        if len(digits) < 6:
            return digits.ljust(6, '0')
        else:
            return digits[:6]
    
    def extract_hs_levels(self, hs6: str) -> dict:
        """
        Extract hierarchical HS levels from HS6 code
        
        Args:
            hs6: 6-digit HS code
        
        Returns:
            Dict with chapter (HS2), heading (HS4), subheading (HS6)
        """
        if not isinstance(hs6, str) or len(hs6) != 6:
            return {'chapter': None, 'heading': None, 'subheading': None}
        
        return {
            'chapter': hs6[:2],      # First 2 digits
            'heading': hs6[:4],      # First 4 digits
            'subheading': hs6        # All 6 digits
        }
    
    def clean_text_description(self, text: str) -> str:
        """
        Clean and normalize text descriptions
        
        Args:
            text: Raw description text
        
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:()\-]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def process_wto_adb(self, input_file: Optional[str] = None) -> pd.DataFrame:
        """
        Process WTO ADB tariff data with flexible column detection
        
        Args:
            input_file: Path to raw WTO file (optional)
        
        Returns:
            Processed DataFrame
        """
        if input_file is None:
            input_file = self.raw_data_dir / "wto" / "adb_tariff_data.csv"
        else:
            input_file = Path(input_file)
        
        if not input_file.exists():
            logger.error(f"File not found: {input_file}")
            return pd.DataFrame()
        
        logger.info(f"Processing WTO ADB data from {input_file}")
        
        # Load data
        df = pd.read_csv(input_file, dtype=str)
        logger.info(f"Loaded {len(df)} rows")
        
        # Detect columns
        logger.info("Detecting column names...")
        hs_col = self.detect_columns(df, 'hs_code')
        tariff_col = self.detect_columns(df, 'tariff')
        reporter_col = self.detect_columns(df, 'reporter')
        year_col = self.detect_columns(df, 'year')
        
        if not hs_col:
            logger.error("Could not detect HS code column!")
            logger.info(f"Available columns: {list(df.columns)}")
            return pd.DataFrame()
        
        # Create standardized column names
        df_clean = pd.DataFrame()
        
        # HS code (required)
        df_clean['hs6'] = df[hs_col].apply(self.normalize_hs_code)
        
        # Reporter (optional)
        if reporter_col:
            df_clean['reporter_name'] = df[reporter_col].astype(str)
            df_clean['reporter_code'] = df_clean['reporter_name']  # Simplified
        else:
            df_clean['reporter_name'] = 'Unknown'
            df_clean['reporter_code'] = 'UNK'
        
        # Year (optional)
        if year_col:
            df_clean['year'] = pd.to_numeric(df[year_col], errors='coerce').astype('Int64')
        else:
            df_clean['year'] = pd.NA
        
        # Tariff rate (optional)
        if tariff_col:
            df_clean['mfn_rate_percent'] = pd.to_numeric(df[tariff_col], errors='coerce')
            df_clean['mfn_rate'] = df_clean['mfn_rate_percent'] / 100.0
        else:
            df_clean['mfn_rate_percent'] = pd.NA
            df_clean['mfn_rate'] = pd.NA
        
        # Extract hierarchical levels
        hs_levels = df_clean['hs6'].apply(self.extract_hs_levels)
        df_clean['chapter'] = [x['chapter'] for x in hs_levels]
        df_clean['heading'] = [x['heading'] for x in hs_levels]
        df_clean['subheading'] = [x['subheading'] for x in hs_levels]
        
        # Add classification version
        df_clean['classification_version'] = 'HS2022'
        
        # Clean and filter
        df_clean = df_clean[df_clean['hs6'].notna()]
        df_clean = df_clean.drop_duplicates(subset=['reporter_name', 'year', 'hs6'])
        
        logger.info(f"Processed {len(df_clean)} rows")
        
        # Save
        output_file = self.processed_data_dir / "wto_tariffs_clean.csv"
        df_clean.to_csv(output_file, index=False)
        logger.info(f"Saved to {output_file}")
        
        return df_clean
    
    def process_wco_hs_descriptions(
        self, 
        input_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process WCO HS descriptions with flexible column detection
        
        Args:
            input_file: Path to raw WCO file (optional)
        
        Returns:
            Processed DataFrame
        """
        if input_file is None:
            input_file = self.raw_data_dir / "wco" / "wco_hs_hs2022.csv"
        else:
            input_file = Path(input_file)
        
        if not input_file.exists():
            logger.error(f"File not found: {input_file}")
            return pd.DataFrame()
        
        logger.info(f"Processing WCO HS descriptions from {input_file}")
        
        # Load data
        df = pd.read_csv(input_file, dtype=str)
        logger.info(f"Loaded {len(df)} rows")
        
        # Detect columns
        logger.info("Detecting column names...")
        hs_col = self.detect_columns(df, 'hs_code')
        desc_col = self.detect_columns(df, 'description')
        
        if not hs_col:
            logger.error("Could not detect HS code column!")
            logger.info(f"Available columns: {list(df.columns)}")
            logger.info("\nFirst row sample:")
            print(df.head(1).T)
            return pd.DataFrame()
        
        if not desc_col:
            logger.error("Could not detect description column!")
            logger.info(f"Available columns: {list(df.columns)}")
            return pd.DataFrame()
        
        # Create clean dataframe
        df_clean = pd.DataFrame()
        
        # Normalize HS codes
        df_clean['hs_code_clean'] = df[hs_col].apply(self.normalize_hs_code)
        
        # Clean descriptions
        df_clean['description_clean'] = df[desc_col].apply(self.clean_text_description)
        
        # Extract hierarchical levels
        hs_levels = df_clean['hs_code_clean'].apply(self.extract_hs_levels)
        df_clean['chapter'] = [x['chapter'] for x in hs_levels]
        df_clean['heading'] = [x['heading'] for x in hs_levels]
        df_clean['hs6'] = [x['subheading'] for x in hs_levels]
        
        # Filter: keep only valid HS6 codes (6 digits)
        df_clean = df_clean[df_clean['hs_code_clean'].notna()]
        df_clean = df_clean[df_clean['hs_code_clean'].str.len() == 6]
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates(subset=['hs6', 'description_clean'])
        
        # Select and rename columns
        df_clean = df_clean[['hs6', 'chapter', 'heading', 'description_clean']].copy()
        df_clean = df_clean.rename(columns={'description_clean': 'description'})
        
        # Remove empty descriptions
        df_clean = df_clean[df_clean['description'].str.len() > 10]
        
        logger.info(f"Processed {len(df_clean)} rows with {df_clean['hs6'].nunique()} unique HS6 codes")
        
        # Save
        output_file = self.processed_data_dir / "wco_hs_descriptions_clean.csv"
        df_clean.to_csv(output_file, index=False)
        logger.info(f"Saved to {output_file}")
        
        return df_clean
    
    def create_unified_dataset(self) -> pd.DataFrame:
        """
        Merge WTO tariffs and WCO descriptions into unified dataset
        
        Returns:
            Unified DataFrame ready for modeling
        """
        logger.info("Creating unified dataset")
        
        # Load processed files
        tariffs_file = self.processed_data_dir / "wto_tariffs_clean.csv"
        descriptions_file = self.processed_data_dir / "wco_hs_descriptions_clean.csv"
        
        if not tariffs_file.exists() or not descriptions_file.exists():
            logger.error("Processed files not found. Run preprocessing first.")
            return pd.DataFrame()
        
        df_tariffs = pd.read_csv(tariffs_file, dtype={'hs6': str})
        df_descriptions = pd.read_csv(descriptions_file, dtype={'hs6': str})
        
        logger.info(f"Tariffs: {len(df_tariffs)} rows")
        logger.info(f"Descriptions: {len(df_descriptions)} rows")
        
        # Get latest tariff for each (reporter, hs6)
        df_tariffs['year'] = pd.to_numeric(df_tariffs['year'], errors='coerce')
        df_tariffs = df_tariffs.sort_values(['reporter_name', 'hs6', 'year'], ascending=[True, True, False])
        df_latest = df_tariffs.drop_duplicates(subset=['reporter_name', 'hs6'], keep='first')
        
        # Merge with descriptions
        df_unified = df_descriptions.merge(
            df_latest[['reporter_name', 'hs6', 'year', 'mfn_rate_percent', 'mfn_rate']],
            on='hs6',
            how='left'
        )
        
        logger.info(f"Unified dataset: {len(df_unified)} rows")
        
        # Save
        output_file = self.processed_data_dir / "unified_hs_tariff_dataset.csv"
        df_unified.to_csv(output_file, index=False)
        logger.info(f"Saved to {output_file}")
        
        return df_unified


def main():
    """Run preprocessing workflow"""
    
    preprocessor = DataPreprocessor(
        raw_data_dir="data/raw",
        processed_data_dir="data/processed"
    )
    
    # Process WTO tariffs
    logger.info("=" * 60)
    logger.info("PROCESSING WTO ADB TARIFFS")
    logger.info("=" * 60)
    df_tariffs = preprocessor.process_wto_adb()
    
    # Process WCO descriptions
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING WCO HS DESCRIPTIONS")
    logger.info("=" * 60)
    df_descriptions = preprocessor.process_wco_hs_descriptions()
    
    if df_descriptions.empty:
        logger.error("\n⚠️  WCO processing failed!")
        logger.error("Please run the diagnostic script to check your file:")
        logger.error("  python diagnose_wco_data.py")
        return
    
    # Create unified dataset
    logger.info("\n" + "=" * 60)
    logger.info("CREATING UNIFIED DATASET")
    logger.info("=" * 60)
    df_unified = preprocessor.create_unified_dataset()
    
    # Summary statistics
    if not df_unified.empty:
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total rows: {len(df_unified)}")
        logger.info(f"Unique HS6 codes: {df_unified['hs6'].nunique()}")
        logger.info(f"Unique chapters: {df_unified['chapter'].nunique()}")
        
        if 'reporter_name' in df_unified.columns:
            logger.info(f"\nRecords by country:")
            print(df_unified['reporter_name'].value_counts())
    
    logger.info("\n✅ Preprocessing complete!")
    logger.info("Next steps:")
    logger.info("1. Train baseline model: python src/models/baseline.py")


if __name__ == "__main__":
    main()
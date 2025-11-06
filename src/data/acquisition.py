"""
Data Acquisition Module
Downloads and validates customs datasets from multiple sources
"""

import os
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAcquisition:
    """Handles downloading and initial validation of customs datasets from various sources.
     Args:
        raw_data_dir: Directory to store raw downloaded data
     """
    
    def __init__(self, raw_data_dir: str = "data/raw"):
        """Initialize DataAcquisition with raw data directory
         Args:
            raw_data_dir: Directory to store raw downloaded data
         """
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.raw_data_dir / "wto").mkdir(exist_ok=True)
        (self.raw_data_dir / "wco").mkdir(exist_ok=True)
        (self.raw_data_dir / "synthetic").mkdir(exist_ok=True)
    
    def download_wto_adb(
        self, 
        reporters: List[str] = ["EU", "CAN", "CHE"],
        years: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Download WTO Analytical Database tariff data
        
        Args:
            reporters: List of country codes (EU, CAN, CHE)
            years: Years to download (default: 2020-2024)
        
        Returns:
            DataFrame with harmonized tariff data
        """
        if years is None:
            years = list(range(2020, 2025))
        
        logger.info(f"Downloading WTO ADB data for {reporters}, years {years}")
        
        # TODO: Replace with actual WTO API calls or file downloads
        # For now, create a placeholder structure
        
        output_file = self.raw_data_dir / "wto" / "adb_tariff_data.csv"
        
        if output_file.exists():
            logger.info(f"Loading existing file: {output_file}")
            return pd.read_csv(output_file)
        
        logger.warning("WTO ADB download not implemented. Please manually download from:")
        logger.warning("https://www.wto.org/english/res_e/statis_e/trade_datasets_e.htm")
        logger.warning(f"Save to: {output_file}")
        
        # Return empty DataFrame with expected schema
        return pd.DataFrame(columns=[
            'reporter_code', 'reporter_name', 'year', 
            'classification_version', 'hs6', 'measure',
            'mfn_rate_percent', 'mfn_rate', 'source'
        ])
    
    def load_wco_hs_descriptions(self, version: str = "HS2022") -> pd.DataFrame:
        """Load WCO Harmonized System descriptions
        Args:
            version: HS version (default: HS2022)
        Returns:
            DataFrame with HS codes and descriptions
        """
        logger.info(f"Loading WCO HS descriptions ({version})")
        
        input_file = self.raw_data_dir / "wco" / f"wco_hs_{version.lower()}.csv"
        
        if input_file.exists():
            logger.info(f"Loading existing file: {input_file}")
            df = pd.read_csv(input_file, dtype=str)
            return df
        
        logger.warning("WCO HS descriptions not found. Please download from:")
        logger.warning("https://www.wcoomd.org/en/topics/nomenclature/instrument-and-tools/hs-nomenclature-2022-edition.aspx")
        logger.warning(f"Save to: {input_file}")
        
        # Return empty DataFrame with expected schema
        return pd.DataFrame(columns=['hs_code', 'description', 'level'])
    
    def validate_data_quality(
        self, 
        df: pd.DataFrame, 
        required_columns: List[str],
        name: str = "Dataset"
    ) -> dict:
        """
        Validate data quality and return statistics
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            name: Dataset name for logging
        Returns:
            Dictionary with validation statistics
        """
        logger.info(f"Validating {name}")
        
        stats = {
            'total_rows': len(df),
            'columns': list(df.columns),
            'missing_columns': [],
            'missing_values': {},
            'duplicates': 0
        }
        
        # Check required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            stats['missing_columns'] = list(missing_cols)
            logger.warning(f"Missing columns in {name}: {missing_cols}")
        
        # Check missing values
        for col in required_columns:
            if col in df.columns:
                missing = df[col].isna().sum()
                if missing > 0:
                    stats['missing_values'][col] = missing
                    logger.warning(f"{name}: {col} has {missing} missing values")
        
        # Check duplicates
        duplicates = df.duplicated().sum()
        stats['duplicates'] = duplicates
        if duplicates > 0:
            logger.warning(f"{name}: {duplicates} duplicate rows found")
        
        logger.info(f"{name} validation complete: {len(df)} rows, {len(df.columns)} columns")
        
        return stats
    
    def normalize_hs_code(self, code: str) -> Optional[str]:
        """
        Normalize HS code to 6-digit format
        Args:
            code: Raw HS code string
        Returns:
            6-digit zero-padded HS code or None if invalid
        """
        if pd.isna(code):
            return None
        
        # Extract digits only
        digits = ''.join(c for c in str(code) if c.isdigit())
        
        if not digits:
            return None
        
        # Pad or truncate to 6 digits
        if len(digits) < 6:
            return digits.ljust(6, '0')
        else:
            return digits[:6]


def main():
    """Run data acquisition workflow"""
    
    # Initialize
    acquirer = DataAcquisition(raw_data_dir="data/raw")
    
    # Download WTO ADB data
    df_wto = acquirer.download_wto_adb(
        reporters=["EU", "CAN", "CHE"],
        years=[2022, 2023, 2024]
    )
    
    # Validate
    if not df_wto.empty:
        acquirer.validate_data_quality(
            df_wto,
            required_columns=['reporter_name', 'hs6', 'mfn_rate_percent'],
            name="WTO ADB"
        )
    
    # Load WCO descriptions
    df_wco = acquirer.load_wco_hs_descriptions(version="HS2022")
    
    # Validate
    if not df_wco.empty:
        acquirer.validate_data_quality(
            df_wco,
            required_columns=['hs_code', 'description'],
            name="WCO HS2022"
        )
    
    logger.info("Data acquisition complete!")
    logger.info("Next steps:")
    logger.info("1. Manually download missing datasets (see warnings above)")
    logger.info("2. Run preprocessing: python src/data/preprocessing.py")


if __name__ == "__main__":
    main()

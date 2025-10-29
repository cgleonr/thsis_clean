"""
Comprehensive Diagnostic and Fix Script
Identifies and fixes data quality issues in the HS classification system
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_wco_descriptions(file_path: str):
    """Diagnose WCO descriptions data quality"""
    logger.info("=" * 60)
    logger.info("DIAGNOSING WCO DESCRIPTIONS")
    logger.info("=" * 60)
    
    df = pd.read_csv(file_path, dtype={'hs6': str})
    
    logger.info(f"\n1. Basic Statistics:")
    logger.info(f"   Total rows: {len(df)}")
    logger.info(f"   Columns: {list(df.columns)}")
    logger.info(f"   Unique HS6 codes: {df['hs6'].nunique()}")
    
    logger.info(f"\n2. Data Quality Issues:")
    logger.info(f"   Missing HS6 codes: {df['hs6'].isna().sum()}")
    logger.info(f"   Missing descriptions: {df['description'].isna().sum()}")
    logger.info(f"   Empty descriptions: {(df['description'].str.strip() == '').sum()}")
    
    # Check HS6 format
    invalid_hs6 = df[df['hs6'].str.len() != 6]
    logger.info(f"   Invalid HS6 format (not 6 digits): {len(invalid_hs6)}")
    
    logger.info(f"\n3. Sample Records:")
    for idx in range(min(5, len(df))):
        row = df.iloc[idx]
        logger.info(f"   HS6: {row['hs6']} | Desc: {row['description'][:60]}...")
    
    return df


def diagnose_tariff_data(file_path: str):
    """Diagnose tariff data quality"""
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSING TARIFF DATA")
    logger.info("=" * 60)
    
    df = pd.read_csv(file_path, dtype={'hs6': str})
    
    logger.info(f"\n1. Basic Statistics:")
    logger.info(f"   Total rows: {len(df)}")
    logger.info(f"   Columns: {list(df.columns)}")
    logger.info(f"   Unique reporters: {df['reporter_name'].nunique()}")
    logger.info(f"   Unique HS6 codes: {df['hs6'].nunique()}")
    
    logger.info(f"\n2. Data Quality Issues:")
    logger.info(f"   Missing HS6 codes: {df['hs6'].isna().sum()}")
    logger.info(f"   Missing MFN rates: {df['mfn_rate_percent'].isna().sum()}")
    logger.info(f"   Zero MFN rates: {(df['mfn_rate_percent'] == 0).sum()}")
    logger.info(f"   Non-zero MFN rates: {(df['mfn_rate_percent'] > 0).sum()}")
    
    logger.info(f"\n3. MFN Rate Distribution:")
    logger.info(df['mfn_rate_percent'].describe())
    
    logger.info(f"\n4. Records by Reporter:")
    reporter_counts = df['reporter_name'].value_counts()
    for reporter, count in reporter_counts.items():
        non_zero = (df[df['reporter_name'] == reporter]['mfn_rate_percent'] > 0).sum()
        logger.info(f"   {reporter}: {count} records, {non_zero} with MFN rate > 0")
    
    logger.info(f"\n5. Sample Records with MFN Rates:")
    sample_with_rates = df[df['mfn_rate_percent'] > 0].head(5)
    for idx, row in sample_with_rates.iterrows():
        logger.info(f"   {row['reporter_name']} | HS6: {row['hs6']} | "
                   f"MFN: {row['mfn_rate_percent']}% | Year: {row['year']}")
    
    return df


def fix_wco_descriptions(input_path: str, output_path: str):
    """Clean and fix WCO descriptions"""
    logger.info("\n" + "=" * 60)
    logger.info("FIXING WCO DESCRIPTIONS")
    logger.info("=" * 60)
    
    df = pd.read_csv(input_path, dtype={'hs6': str})
    original_count = len(df)
    
    # Remove records with missing or invalid data
    df = df.dropna(subset=['hs6', 'description'])
    df = df[df['description'].str.strip() != '']
    df = df[df['hs6'].str.len() == 6]
    
    # Clean descriptions
    df['description'] = df['description'].str.strip().str.lower()
    
    # Remove duplicates (keep first)
    df = df.drop_duplicates(subset=['hs6'], keep='first')
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    
    logger.info(f"   Original records: {original_count}")
    logger.info(f"   Cleaned records: {len(df)}")
    logger.info(f"   Removed: {original_count - len(df)}")
    logger.info(f"   Saved to: {output_path}")
    
    return df


def fix_tariff_data(input_path: str, output_path: str):
    """Clean and fix tariff data"""
    logger.info("\n" + "=" * 60)
    logger.info("FIXING TARIFF DATA")
    logger.info("=" * 60)
    
    df = pd.read_csv(input_path, dtype={'hs6': str})
    original_count = len(df)
    
    # Remove records with missing critical data
    df = df.dropna(subset=['hs6', 'reporter_name'])
    df = df[df['hs6'].str.len() == 6]
    
    # Convert year to numeric
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    
    # Handle MFN rates
    df['mfn_rate_percent'] = pd.to_numeric(df['mfn_rate_percent'], errors='coerce')
    df['mfn_rate_percent'] = df['mfn_rate_percent'].fillna(0.0)
    
    # Keep only latest year for each (reporter, hs6)
    df = df.sort_values(['reporter_name', 'hs6', 'year'], ascending=[True, True, False])
    df_latest = df.drop_duplicates(subset=['reporter_name', 'hs6'], keep='first')
    
    # Save cleaned data
    df_latest.to_csv(output_path, index=False)
    
    logger.info(f"   Original records: {original_count}")
    logger.info(f"   After cleaning: {len(df)}")
    logger.info(f"   Latest records only: {len(df_latest)}")
    logger.info(f"   Removed: {original_count - len(df_latest)}")
    logger.info(f"   Saved to: {output_path}")
    
    # Statistics on cleaned data
    logger.info(f"\n   MFN Rate Statistics (cleaned):")
    logger.info(f"   Records with MFN > 0: {(df_latest['mfn_rate_percent'] > 0).sum()}")
    logger.info(f"   Records with MFN = 0: {(df_latest['mfn_rate_percent'] == 0).sum()}")
    
    return df_latest


def test_lookup(wco_path: str, tariff_path: str, test_queries: list):
    """Test the lookup system with sample queries"""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING LOOKUP SYSTEM")
    logger.info("=" * 60)
    
    wco_df = pd.read_csv(wco_path, dtype={'hs6': str})
    tariff_df = pd.read_csv(tariff_path, dtype={'hs6': str})
    
    # Create lookup dict
    tariff_lookup = tariff_df.set_index(['reporter_name', 'hs6'])[['year', 'mfn_rate_percent']].to_dict('index')
    
    logger.info(f"\nLoaded {len(wco_df)} HS descriptions")
    logger.info(f"Loaded {len(tariff_df)} tariff records")
    logger.info(f"Tariff lookup size: {len(tariff_lookup)}")
    
    # Test specific HS codes
    test_codes = ['620342', '851712', '090111', '940350', '640399']
    
    logger.info(f"\n" + "=" * 60)
    logger.info("SAMPLE HS CODE LOOKUPS")
    logger.info("=" * 60)
    
    for hs6 in test_codes:
        # Get description
        desc_row = wco_df[wco_df['hs6'] == hs6]
        if len(desc_row) > 0:
            desc = desc_row.iloc[0]['description']
            logger.info(f"\nHS6: {hs6}")
            logger.info(f"Description: {desc[:80]}...")
            
            # Get tariffs for all reporters
            for reporter in ['European Union', 'Canada', 'Switzerland']:
                key = (reporter, hs6)
                if key in tariff_lookup:
                    data = tariff_lookup[key]
                    logger.info(f"  {reporter}: {data['mfn_rate_percent']}% (Year: {int(data['year'])})")
                else:
                    logger.info(f"  {reporter}: No tariff data")
        else:
            logger.info(f"\nHS6: {hs6} - NOT FOUND in WCO data")


def main():
    """Main diagnostic and fix routine"""
    
    # Define paths
    data_dir = Path("data/processed")
    
    wco_input = data_dir / "wco_hs_descriptions_clean.csv"
    wco_output = data_dir / "wco_hs_descriptions_fixed.csv"
    
    tariff_input = data_dir / "wto_tariffs_clean.csv"
    tariff_output = data_dir / "wto_tariffs_fixed.csv"
    
    # Check if files exist
    if not wco_input.exists():
        logger.error(f"WCO file not found: {wco_input}")
        logger.error("Please run data preprocessing first!")
        return
    
    if not tariff_input.exists():
        logger.error(f"Tariff file not found: {tariff_input}")
        logger.error("Please run data preprocessing first!")
        return
    
    # Run diagnostics
    logger.info("STARTING COMPREHENSIVE DIAGNOSTICS\n")
    
    wco_df = diagnose_wco_descriptions(str(wco_input))
    tariff_df = diagnose_tariff_data(str(tariff_input))
    
    # Fix data
    logger.info("\n\nSTARTING DATA FIXES\n")
    
    wco_fixed = fix_wco_descriptions(str(wco_input), str(wco_output))
    tariff_fixed = fix_tariff_data(str(tariff_input), str(tariff_output))
    
    # Test lookup
    test_lookup(
        str(wco_output),
        str(tariff_output),
        test_queries=["smartphone", "leather handbag", "coffee beans"]
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC AND FIX COMPLETE!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Rebuild baseline model with fixed data:")
    logger.info("   python src/models/baseline.py")
    logger.info("2. Update streamlit app to use fixed files")
    logger.info("3. Test predictions")


if __name__ == "__main__":
    main()

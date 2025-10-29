"""
Emergency Diagnostic Script
Identifies why web app is showing low confidence and wrong codes
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_complete_system():
    """Complete system diagnosis"""
    
    logger.info("=" * 70)
    logger.info("EMERGENCY DIAGNOSIS - COMPLETE SYSTEM CHECK")
    logger.info("=" * 70)
    
    issues_found = []
    
    # 1. Check data files exist
    logger.info("\n1. CHECKING DATA FILES")
    logger.info("-" * 70)
    
    data_files = {
        'WCO Descriptions (fixed)': 'data/processed/wco_hs_descriptions_fixed.csv',
        'WCO Descriptions (clean)': 'data/processed/wco_hs_descriptions_clean.csv',
        'Tariffs (fixed)': 'data/processed/wto_tariffs_fixed.csv',
        'Tariffs (clean)': 'data/processed/wto_tariffs_clean.csv',
    }
    
    existing_files = {}
    for name, path in data_files.items():
        p = Path(path)
        if p.exists():
            logger.info(f"   ✅ {name}: {path} ({p.stat().st_size / 1024:.1f} KB)")
            existing_files[name] = path
        else:
            logger.warning(f"   ❌ {name}: NOT FOUND")
    
    if not existing_files:
        logger.error("   🚨 NO DATA FILES FOUND!")
        issues_found.append("No data files exist")
        return issues_found
    
    # 2. Check which files the app is using
    logger.info("\n2. CHECKING BASELINE MODEL FILES")
    logger.info("-" * 70)
    
    model_files = {
        'Embeddings': 'models/baseline/hs_embeddings.npy',
        'Metadata': 'models/baseline/hs_metadata.csv',
    }
    
    for name, path in model_files.items():
        p = Path(path)
        if p.exists():
            logger.info(f"   ✅ {name}: {path} ({p.stat().st_size / 1024:.1f} KB)")
        else:
            logger.error(f"   ❌ {name}: NOT FOUND")
            issues_found.append(f"Missing {name}")
    
    # 3. Analyze the metadata file
    logger.info("\n3. ANALYZING BASELINE METADATA")
    logger.info("-" * 70)
    
    metadata_path = Path('models/baseline/hs_metadata.csv')
    if metadata_path.exists():
        df_meta = pd.read_csv(metadata_path, dtype={'hs6': str})
        logger.info(f"   Records: {len(df_meta)}")
        logger.info(f"   Unique HS6: {df_meta['hs6'].nunique()}")
        
        # Check for smartphone codes
        logger.info("\n   Checking for expected smartphone codes:")
        expected_smartphone_codes = ['851712', '851762', '847130']
        
        for code in expected_smartphone_codes:
            if code in df_meta['hs6'].values:
                desc = df_meta[df_meta['hs6'] == code]['description'].iloc[0]
                logger.info(f"   ✅ {code}: {desc[:60]}...")
            else:
                logger.error(f"   ❌ {code}: NOT FOUND IN METADATA")
                issues_found.append(f"Missing HS code {code}")
        
        # Show what it found instead
        logger.info("\n   What the model has for chapter 42 (leather goods):")
        ch42 = df_meta[df_meta['hs6'].str.startswith('42')]
        if len(ch42) > 0:
            logger.info(f"   Found {len(ch42)} codes in chapter 42")
            for idx, row in ch42.head(5).iterrows():
                logger.info(f"      {row['hs6']}: {row['description'][:50]}...")
        else:
            logger.warning("   No chapter 42 codes found")
        
        logger.info("\n   What the model has for chapter 85 (electronics):")
        ch85 = df_meta[df_meta['hs6'].str.startswith('85')]
        if len(ch85) > 0:
            logger.info(f"   Found {len(ch85)} codes in chapter 85")
            for idx, row in ch85.head(5).iterrows():
                logger.info(f"      {row['hs6']}: {row['description'][:50]}...")
        else:
            logger.error("   ❌ No chapter 85 codes found!")
            issues_found.append("Missing electronics chapter")
    else:
        logger.error("   ❌ Metadata file not found")
        issues_found.append("No baseline metadata")
    
    # 4. Check tariff file
    logger.info("\n4. ANALYZING TARIFF DATA")
    logger.info("-" * 70)
    
    tariff_files_to_check = [
        'data/processed/wto_tariffs_fixed.csv',
        'data/processed/wto_tariffs_clean.csv'
    ]
    
    tariff_df = None
    for tf in tariff_files_to_check:
        if Path(tf).exists():
            tariff_df = pd.read_csv(tf, dtype={'hs6': str})
            logger.info(f"   Using: {tf}")
            break
    
    if tariff_df is not None:
        logger.info(f"   Total records: {len(tariff_df)}")
        logger.info(f"   Unique HS6 codes: {tariff_df['hs6'].nunique()}")
        logger.info(f"   Countries: {tariff_df['reporter_name'].unique()}")
        
        # Check for MFN rates
        has_mfn = tariff_df['mfn_rate_percent'].notna().sum()
        has_positive_mfn = (tariff_df['mfn_rate_percent'] > 0).sum()
        logger.info(f"   Records with MFN rate: {has_mfn} ({has_mfn/len(tariff_df)*100:.1f}%)")
        logger.info(f"   Records with MFN > 0: {has_positive_mfn} ({has_positive_mfn/len(tariff_df)*100:.1f}%)")
        
        # Check specific smartphone code
        logger.info("\n   Checking smartphone HS code 851712:")
        for country in ['European Union', 'Canada', 'Switzerland']:
            lookup = tariff_df[(tariff_df['hs6'] == '851712') & 
                              (tariff_df['reporter_name'] == country)]
            if len(lookup) > 0:
                rate = lookup.iloc[0]['mfn_rate_percent']
                year = lookup.iloc[0]['year']
                logger.info(f"      {country}: {rate}% (Year: {year})")
            else:
                logger.warning(f"      {country}: NOT FOUND")
                issues_found.append(f"No tariff for 851712 in {country}")
    else:
        logger.error("   ❌ No tariff file found")
        issues_found.append("No tariff data")
    
    # 5. Check hierarchical model
    logger.info("\n5. CHECKING HIERARCHICAL MODEL")
    logger.info("-" * 70)
    
    hier_files = {
        'Model': 'models/hierarchical/best_model.pt',
        'Mappings': 'models/hierarchical/label_mappings.json'
    }
    
    for name, path in hier_files.items():
        p = Path(path)
        if p.exists():
            logger.info(f"   ✅ {name}: {path} ({p.stat().st_size / 1024:.1f} KB)")
        else:
            logger.error(f"   ❌ {name}: NOT FOUND")
            issues_found.append(f"Missing hierarchical {name}")
    
    # 6. Test embedding quality
    logger.info("\n6. TESTING EMBEDDING QUALITY")
    logger.info("-" * 70)
    
    embeddings_path = Path('models/baseline/hs_embeddings.npy')
    if embeddings_path.exists() and metadata_path.exists():
        try:
            embeddings = np.load(embeddings_path)
            logger.info(f"   Embeddings shape: {embeddings.shape}")
            logger.info(f"   Embeddings dtype: {embeddings.dtype}")
            
            # Check if embeddings are normalized
            norms = np.linalg.norm(embeddings, axis=1)
            logger.info(f"   Embedding norms - Mean: {norms.mean():.4f}, Std: {norms.std():.4f}")
            
            if abs(norms.mean() - 1.0) > 0.1:
                logger.warning(f"   ⚠️  Embeddings may not be normalized properly")
                issues_found.append("Embeddings not normalized")
            
            # Check for any NaN or Inf
            if np.isnan(embeddings).any():
                logger.error("   ❌ Embeddings contain NaN values!")
                issues_found.append("NaN in embeddings")
            
            if np.isinf(embeddings).any():
                logger.error("   ❌ Embeddings contain Inf values!")
                issues_found.append("Inf in embeddings")
                
        except Exception as e:
            logger.error(f"   ❌ Error loading embeddings: {e}")
            issues_found.append(f"Embedding load error: {e}")
    
    # 7. Summary
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSIS SUMMARY")
    logger.info("=" * 70)
    
    if issues_found:
        logger.error(f"\n🚨 FOUND {len(issues_found)} ISSUES:")
        for i, issue in enumerate(issues_found, 1):
            logger.error(f"   {i}. {issue}")
        
        logger.info("\n" + "=" * 70)
        logger.info("RECOMMENDED FIXES")
        logger.info("=" * 70)
        
        if "Missing HS code 851712" in issues_found:
            logger.info("\n❌ CRITICAL: Baseline model doesn't have smartphone codes!")
            logger.info("   FIX: Rebuild baseline model with correct data")
            logger.info("   RUN: python src/models/baseline.py")
        
        if "No tariff for 851712" in str(issues_found):
            logger.info("\n❌ CRITICAL: Tariff data missing smartphone entries!")
            logger.info("   FIX: Re-run data preprocessing")
            logger.info("   RUN: python scripts/fix_data_issues.py")
        
        if "Embeddings not normalized" in issues_found:
            logger.info("\n⚠️  WARNING: Embeddings may need regeneration")
            logger.info("   FIX: Rebuild baseline embeddings")
            logger.info("   RUN: python src/models/baseline.py")
        
    else:
        logger.info("\n✅ No critical issues found in system check")
        logger.info("\n   But results are still poor. This suggests:")
        logger.info("   1. Model using wrong/old data files")
        logger.info("   2. Streamlit app not loading correct files")
        logger.info("   3. Need to verify app is using 'fixed' files")
    
    return issues_found


if __name__ == "__main__":
    issues = diagnose_complete_system()
    
    if issues:
        print("\n\n⚠️  ISSUES FOUND - CHECK LOGS ABOVE FOR FIXES")
        exit(1)
    else:
        print("\n\n✅ System check passed - but verify app is using correct files")
        exit(0)
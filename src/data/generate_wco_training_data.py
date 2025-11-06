#!/usr/bin/env python3
"""
Generate Realistic WCO Training Data with Context-Aware Augmentation

Uses intelligent rules based on product categories to generate realistic variations.
"""

import pandas as pd
import random
from pathlib import Path
import re

random.seed(42)

# Context-aware augmentation rules
LIVE_ANIMALS = {
    'prefixes': ['', 'imported', 'certified', 'registered', 'quarantined'],
    'suffixes': ['', 'for breeding', 'for commercial use', 'meeting health standards', 
                 'with veterinary certificate', 'for export', 'certified healthy'],
    'modifiers': ['', 'pure-bred', 'commercial grade', 'certified']
}

FOOD_PRODUCTS = {
    'prefixes': ['', 'fresh', 'frozen', 'dried', 'canned', 'packaged', 'processed', 
                 'raw', 'cooked', 'prepared', 'organic', 'imported'],
    'suffixes': ['', 'for retail sale', 'for commercial use', 'for export', 
                 'in bulk', 'packaged', 'ready to eat', 'for further processing',
                 'meeting food safety standards', 'grade A', 'premium quality'],
    'modifiers': ['', 'premium', 'standard', 'commercial grade', 'high quality']
}

TEXTILES = {
    'prefixes': ['', 'woven', 'knitted', 'processed', 'unprocessed', 'dyed', 
                 'printed', 'plain', 'manufactured', 'imported'],
    'suffixes': ['', 'for garment making', 'for industrial use', 'for retail sale',
                 'in rolls', 'in pieces', 'meeting standards', 'commercial quality'],
    'modifiers': ['', 'premium', 'commercial', 'industrial grade', 'high quality']
}

MACHINERY = {
    'prefixes': ['', 'new', 'used', 'refurbished', 'industrial', 'commercial',
                 'imported', 'manufactured'],
    'suffixes': ['', 'for industrial use', 'for commercial use', 'complete with parts',
                 'with accessories', 'certified', 'meeting safety standards'],
    'modifiers': ['', 'heavy-duty', 'commercial grade', 'industrial', 'professional']
}

CHEMICALS = {
    'prefixes': ['', 'pure', 'refined', 'industrial grade', 'technical grade',
                 'pharmaceutical grade', 'processed', 'synthesized', 'imported'],
    'suffixes': ['', 'for industrial use', 'for commercial use', 'in bulk',
                 'in containers', 'meeting purity standards', 'certified'],
    'modifiers': ['', 'high purity', 'technical grade', 'pharmaceutical grade', 'commercial grade']
}

MANUFACTURED_GOODS = {
    'prefixes': ['', 'manufactured', 'imported', 'commercial', 'retail',
                 'wholesale', 'new', 'packaged'],
    'suffixes': ['', 'for retail sale', 'for commercial use', 'in original packaging',
                 'for export', 'meeting standards', 'certified quality'],
    'modifiers': ['', 'premium', 'standard', 'commercial grade', 'high quality']
}

DEFAULT_RULES = {
    'prefixes': ['', 'imported', 'commercial', 'processed', 'manufactured'],
    'suffixes': ['', 'for commercial use', 'for industrial use', 'for retail sale', 
                 'meeting standards'],
    'modifiers': ['', 'standard', 'commercial grade', 'premium']
}

# Chapter-based categorization
CHAPTER_CATEGORIES = {
    (1, 5): LIVE_ANIMALS,
    (6, 14): FOOD_PRODUCTS,
    (15, 24): FOOD_PRODUCTS,
    (25, 27): MANUFACTURED_GOODS,
    (28, 38): CHEMICALS,
    (39, 40): MANUFACTURED_GOODS,
    (41, 43): MANUFACTURED_GOODS,
    (44, 49): MANUFACTURED_GOODS,
    (50, 63): TEXTILES,
    (64, 67): MANUFACTURED_GOODS,
    (68, 71): MANUFACTURED_GOODS,
    (72, 83): MANUFACTURED_GOODS,
    (84, 85): MACHINERY,
    (86, 89): MACHINERY,
    (90, 92): MANUFACTURED_GOODS,
    (93, 97): MANUFACTURED_GOODS
}


def get_chapter_number(hs6):
    """Extract chapter number from HS6 code"""
    return int(str(hs6)[:2])


def get_augmentation_rules(hs6, description):
    """Get appropriate augmentation rules based on HS code"""
    chapter = get_chapter_number(hs6)
    
    for (start, end), rules in CHAPTER_CATEGORIES.items():
        if start <= chapter <= end:
            return rules
    
    return DEFAULT_RULES


def is_sensible_combination(prefix, description, suffix, modifier):
    """Check if the combination makes sense"""
    combined = f"{modifier} {prefix} {description} {suffix}".lower()
    
    # Nonsensical combinations to avoid
    bad_patterns = [
        (r'(bottled|canned|frozen|dried)\s+(live|living)', 'Cannot can/bottle live animals'),
        (r'(raw|fresh)\s+\w*\s*(machinery|machine|equipment)', 'Machinery is not raw/fresh'),
        (r'cooked\s+\w*\s*(metal|steel|iron|machinery)', 'Cannot cook metals'),
        (r'edible\s+\w*\s*(machinery|equipment|metal)', 'Machinery is not edible'),
        (r'(woven|knitted)\s+\w*\s*(live|animal)', 'Cannot weave animals'),
        (r'(frozen|dried)\s+\w*\s*(machinery|vehicle|equipment)', 'Cannot freeze machinery'),
    ]
    
    for pattern, reason in bad_patterns:
        if re.search(pattern, combined):
            return False
    
    return True


def generate_realistic_variations(base_description, hs6, num_variations=40):
    """Generate realistic variations based on product context"""
    variations = [base_description]
    
    rules = get_augmentation_rules(hs6, base_description)
    
    attempts = 0
    max_attempts = num_variations * 5
    
    while len(variations) < num_variations and attempts < max_attempts:
        attempts += 1
        
        strategy = random.choice(['prefix_suffix', 'modifier', 'simplify', 'rephrase'])
        
        if strategy == 'prefix_suffix':
            prefix = random.choice(rules['prefixes'])
            suffix = random.choice(rules['suffixes'])
            modifier = ''
            
            parts = [p for p in [prefix, base_description] if p]
            var = ' '.join(parts)
            if suffix:
                var = f"{var}, {suffix}"
        
        elif strategy == 'modifier':
            modifier = random.choice(rules['modifiers'])
            prefix = random.choice(rules['prefixes'][:len(rules['prefixes'])//2])
            suffix = ''
            
            parts = [p for p in [modifier, prefix, base_description] if p]
            var = ' '.join(parts)
        
        elif strategy == 'simplify':
            var = base_description
            if '(' in var:
                var = var.split('(')[0].strip()
            if ';' in var:
                var = var.split(';')[0].strip()
            modifier = prefix = suffix = ''
        
        elif strategy == 'rephrase':
            var = base_description.replace(';', ',')
            var = var.replace('n.e.c.', 'not elsewhere classified')
            var = var.replace('n.e.s.', 'not elsewhere specified')
            modifier = prefix = suffix = ''
        
        if is_sensible_combination(prefix, base_description, suffix, modifier):
            if var not in variations:
                variations.append(var)
    
    return variations[:num_variations]


def clean_description(desc):
    """Clean and normalize WCO description"""
    desc = str(desc).strip()
    desc = desc.replace(';;', ';')
    return desc


def main():
    print("="*70)
    print("GENERATING WCO-BASED TRAINING DATA")
    print("="*70)
    
    input_file = Path('data/raw/wto/harmonized-system_blob.csv')
    
    if not input_file.exists():
        print(f"\nERROR: File not found: {input_file}")
        return
    
    print(f"\nLoading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"   Loaded: {len(df):,} rows")
    
    print("\nFiltering to HS6 codes...")
    hs6_df = df[(df['level'] == 6) & (df['section'] != 'TOTAL')].copy()
    print(f"   HS6 codes found: {len(hs6_df):,}")
    
    hs6_df['hs6'] = hs6_df['hscode'].astype(str).str.zfill(6)
    hs6_df['description'] = hs6_df['description'].apply(clean_description)
    
    official_descriptions = hs6_df[['hs6', 'description']].copy()
    output_desc_file = Path('data/processed/wco_hs_descriptions.csv')
    output_desc_file.parent.mkdir(parents=True, exist_ok=True)
    official_descriptions.to_csv(output_desc_file, index=False)
    print(f"\nSaved official descriptions: {output_desc_file}")
    
    print(f"\nGenerating realistic training variations...")
    print("   Target: 40 variations per HS6 code")
    print("   Using context-aware rules to ensure realism")
    
    training_data = []
    
    for idx, row in hs6_df.iterrows():
        hs6 = row['hs6']
        base_desc = row['description']
        
        variations = generate_realistic_variations(base_desc, hs6, num_variations=40)
        
        for var in variations:
            training_data.append({
                'description': var,
                'hs6': hs6
            })
        
        if (idx + 1) % 500 == 0:
            print(f"   Processed {idx + 1:,} / {len(hs6_df):,} HS codes...")
    
    training_df = pd.DataFrame(training_data)
    
    print(f"\nTraining data generated:")
    print(f"   Total examples: {len(training_df):,}")
    print(f"   Unique HS6 codes: {training_df['hs6'].nunique():,}")
    print(f"   Avg examples per code: {len(training_df) / training_df['hs6'].nunique():.1f}")
    
    output_train_file = Path('data/processed/wco_hs_training_data.csv')
    training_df.to_csv(output_train_file, index=False)
    print(f"\nSaved training data: {output_train_file}")
    
    print(f"\nSTATISTICS:")
    print(f"   Description length (avg): {training_df['description'].str.len().mean():.1f} chars")
    print(f"   Word count (avg): {training_df['description'].str.split().str.len().mean():.1f} words")
    
    print(f"\nSAMPLE TRAINING EXAMPLES:")
    
    animal_sample = training_df[training_df['hs6'] == '010121'].head(5)
    print(f"\n   Live Animals (HS6 010121):")
    for desc in animal_sample['description']:
        print(f"   - {desc}")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nNext: python src/models/train_hierarchical.py")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Generate WCO Training Data with Synonym Support

Adds intelligent synonym variations so model learns:
- horses ↔ equines
- cattle ↔ bovine ↔ cows  
- asses ↔ donkeys
- etc.
"""

import pandas as pd
import random
from pathlib import Path
import re

random.seed(42)

# Domain-specific synonym mappings
SYNONYMS = {
    'horses': ['equines', 'horse', 'equine'],
    'horse': ['equine', 'horses', 'equines'],
    'asses': ['donkeys', 'donkey', 'ass'],
    'donkey': ['ass', 'donkeys', 'asses'],
    'mules': ['mule'],
    'cattle': ['bovine', 'bovines', 'cow', 'cows'],
    'bovine': ['cattle', 'cow', 'cows'],
    'swine': ['pigs', 'pig', 'hogs'],
    'pigs': ['swine', 'pig'],
    'sheep': ['ovine', 'lamb', 'lambs'],
    'goats': ['caprine', 'goat'],
    'poultry': ['fowl', 'birds'],
    'vegetables': ['vegetable', 'veggies'],
    'fruits': ['fruit'],
    'meat': ['flesh'],
    'leather': ['hide', 'skin'],
    'textile': ['fabric', 'cloth'],
    'plastics': ['plastic', 'polymer'],
    'apparatus': ['equipment', 'device', 'machinery'],
    'machinery': ['machines', 'machine', 'equipment'],
    'articles': ['goods', 'items', 'products'],
}

# Context rules (same as before)
LIVE_ANIMALS = {
    'prefixes': ['', 'imported', 'certified', 'registered'],
    'suffixes': ['', 'for breeding', 'for commercial use', 'meeting health standards'],
    'modifiers': ['', 'pure-bred', 'certified']
}

FOOD_PRODUCTS = {
    'prefixes': ['', 'fresh', 'frozen', 'dried', 'packaged', 'processed', 'organic'],
    'suffixes': ['', 'for retail sale', 'for commercial use', 'in bulk', 'ready to eat'],
    'modifiers': ['', 'premium', 'commercial grade', 'high quality']
}

TEXTILES = {
    'prefixes': ['', 'woven', 'knitted', 'dyed', 'printed', 'manufactured'],
    'suffixes': ['', 'for garment making', 'for industrial use', 'in rolls'],
    'modifiers': ['', 'premium', 'commercial', 'industrial grade']
}

MACHINERY = {
    'prefixes': ['', 'new', 'used', 'industrial', 'commercial'],
    'suffixes': ['', 'for industrial use', 'complete with parts', 'certified'],
    'modifiers': ['', 'heavy-duty', 'commercial grade', 'industrial']
}

CHEMICALS = {
    'prefixes': ['', 'pure', 'refined', 'industrial grade', 'technical grade'],
    'suffixes': ['', 'for industrial use', 'in bulk', 'in containers'],
    'modifiers': ['', 'high purity', 'technical grade', 'pharmaceutical grade']
}

MANUFACTURED_GOODS = {
    'prefixes': ['', 'manufactured', 'imported', 'commercial', 'new'],
    'suffixes': ['', 'for retail sale', 'for commercial use', 'in original packaging'],
    'modifiers': ['', 'premium', 'standard', 'commercial grade']
}

DEFAULT_RULES = {
    'prefixes': ['', 'imported', 'commercial', 'processed'],
    'suffixes': ['', 'for commercial use', 'for industrial use'],
    'modifiers': ['', 'standard', 'commercial grade']
}

CHAPTER_CATEGORIES = {
    (1, 5): LIVE_ANIMALS, (6, 14): FOOD_PRODUCTS, (15, 24): FOOD_PRODUCTS,
    (25, 27): MANUFACTURED_GOODS, (28, 38): CHEMICALS, (39, 40): MANUFACTURED_GOODS,
    (41, 43): MANUFACTURED_GOODS, (44, 49): MANUFACTURED_GOODS, (50, 63): TEXTILES,
    (64, 67): MANUFACTURED_GOODS, (68, 71): MANUFACTURED_GOODS, (72, 83): MANUFACTURED_GOODS,
    (84, 85): MACHINERY, (86, 89): MACHINERY, (90, 92): MANUFACTURED_GOODS,
    (93, 97): MANUFACTURED_GOODS
}


def get_rules(hs6):
    chapter = int(str(hs6)[:2])
    for (start, end), rules in CHAPTER_CATEGORIES.items():
        if start <= chapter <= end:
            return rules
    return DEFAULT_RULES


def apply_synonym(text):
    """Replace one word with its synonym"""
    text_lower = text.lower()
    for word, syns in SYNONYMS.items():
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, text_lower):
            syn = random.choice(syns)
            return re.sub(pattern, syn, text, flags=re.IGNORECASE, count=1)
    return text


def is_valid(prefix, desc, suffix, modifier):
    combined = f"{modifier} {prefix} {desc} {suffix}".lower()
    bad = [
        r'(bottled|canned|frozen|dried)\s+(live|living)',
        r'(raw|fresh)\s+\w*\s*(machinery|machine)',
        r'cooked\s+\w*\s*(metal|machinery)',
        r'(woven|knitted)\s+\w*\s*(live|animal)',
    ]
    return not any(re.search(p, combined) for p in bad)


def generate_variations(base_desc, hs6, n=60):
    variations = [base_desc]
    rules = get_rules(hs6)
    attempts = 0
    
    while len(variations) < n and attempts < n * 10:
        attempts += 1
        strategy = random.choice(['prefix_suffix', 'modifier', 'simplify', 'synonym', 'syn_combo'])
        
        if strategy == 'prefix_suffix':
            prefix = random.choice(rules['prefixes'])
            suffix = random.choice(rules['suffixes'])
            var = f"{prefix} {base_desc}".strip() if prefix else base_desc
            if suffix:
                var = f"{var}, {suffix}"
            modifier = ''
        
        elif strategy == 'modifier':
            modifier = random.choice(rules['modifiers'])
            var = f"{modifier} {base_desc}".strip() if modifier else base_desc
            prefix = suffix = ''
        
        elif strategy == 'simplify':
            var = base_desc.split('(')[0].strip() if '(' in base_desc else base_desc
            var = var.split(';')[0].strip() if ';' in var else var
            prefix = suffix = modifier = ''
        
        elif strategy == 'synonym':
            var = apply_synonym(base_desc)
            prefix = suffix = modifier = ''
        
        elif strategy == 'syn_combo':
            var = apply_synonym(base_desc)
            prefix = random.choice(rules['prefixes'])
            suffix = random.choice(rules['suffixes'])
            var = f"{prefix} {var}".strip() if prefix else var
            if suffix:
                var = f"{var}, {suffix}"
            modifier = ''
        
        if is_valid(prefix, var, suffix, modifier) and var not in variations:
            variations.append(var)
    
    return variations[:n]


def main():
    print("="*70)
    print("GENERATING WCO TRAINING DATA WITH SYNONYMS")
    print("="*70)
    
    input_file = Path('data/raw/wto/harmonized-system_blob.csv')
    if not input_file.exists():
        print(f"\nERROR: {input_file} not found")
        return
    
    print(f"\nLoading: {input_file}")
    df = pd.read_csv(input_file)
    
    hs6_df = df[(df['level'] == 6) & (df['section'] != 'TOTAL')].copy()
    hs6_df['hs6'] = hs6_df['hscode'].astype(str).str.zfill(6)
    hs6_df['description'] = hs6_df['description'].str.strip()
    
    print(f"   HS6 codes: {len(hs6_df):,}")
    
    # Save descriptions
    output_desc_file = Path('data/processed/wco_hs_descriptions.csv')
    output_desc_file.parent.mkdir(parents=True, exist_ok=True)
    hs6_df[['hs6', 'description']].to_csv(output_desc_file, index=False)
    print(f"\nSaved: {output_desc_file}")
    
    # Generate training data
    print(f"\nGenerating 60 variations per HS6...")
    training_data = []
    
    for idx, row in hs6_df.iterrows():
        variations = generate_variations(row['description'], row['hs6'], n=60)
        for var in variations:
            training_data.append({'description': var, 'hs6': row['hs6']})
        
        if (idx + 1) % 500 == 0:
            print(f"   {idx + 1:,} / {len(hs6_df):,}")
    
    training_df = pd.DataFrame(training_data)
    
    output_train_file = Path('data/processed/wco_hs_training_data.csv')
    training_df.to_csv(output_train_file, index=False)
    
    print(f"\n✓ Saved: {output_train_file}")
    print(f"  Total: {len(training_df):,} examples")
    print(f"  Avg per code: {len(training_df) / training_df['hs6'].nunique():.1f}")
    
    # Show examples
    print(f"\nSYNONYM EXAMPLES:")
    horse_ex = training_df[training_df['hs6'] == '010121']
    for desc in horse_ex['description'].head(8):
        if 'equine' in desc.lower():
            print(f"  - {desc}")
    
    print("\n" + "="*70)
    print("Model will now understand:")
    print("  horses ↔ equines | cattle ↔ bovine | asses ↔ donkeys")
    print("="*70)


if __name__ == '__main__':
    main()
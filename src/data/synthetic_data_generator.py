import pandas as pd
import random
import numpy as np

# --- Configuration ---
INPUT_META_FILE = 'models/embeddings/hs6_meta.csv'
OUTPUT_CSV_FILE = 'data/processed/synthetic_train.csv'
NUM_CODES_TO_SELECT = 5600
EXAMPLES_PER_CODE = 20

# --- Load Base Data ---
try:
    df_meta = pd.read_csv(INPUT_META_FILE, dtype={'hs6': str})
    # Basic cleaning: drop rows with missing codes or descriptions
    df_meta.dropna(subset=['hs6', 'description'], inplace=True)
    # Ensure descriptions are strings and reasonably long
    df_meta['description'] = df_meta['description'].astype(str)
    df_meta = df_meta[df_meta['description'].str.len() > 5]
    # Keep only unique HS6 codes for selection
    df_meta.drop_duplicates(subset=['hs6'], keep='first', inplace=True)
    print(f"Loaded {len(df_meta)} unique HS codes from {INPUT_META_FILE}")
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_META_FILE}' not found.")
    exit()
except Exception as e:
    print(f"Error loading or processing input file: {e}")
    exit()


# --- Select HS Codes ---
if len(df_meta) < NUM_CODES_TO_SELECT:
    print(f"Warning: Only {len(df_meta)} unique codes available, selecting all.")
    selected_codes_df = df_meta
else:
    selected_codes_df = df_meta.sample(n=NUM_CODES_TO_SELECT, random_state=42)

print(f"Selected {len(selected_codes_df)} codes for dataset generation.")

# --- Helper Function for Generating Variations ---
def generate_variations(base_description, hs_code, num_variations=EXAMPLES_PER_CODE):
    variations = set() # Use a set to avoid exact duplicates initially
    variations.add(base_description.strip().lower()) # Add original

    # Attempt to create variations (simple examples)
    words = base_description.lower().split(';') # Often descriptions use semicolons
    main_desc = words[0].strip()
    qualifiers = words[1:] if len(words) > 1 else []

    # 1. Base description only
    variations.add(main_desc)

    # 2. Add simple prefixes/suffixes
    prefixes = ["standard", "bulk", "imported", "domestic", "packaged", "unprocessed", "processed"]
    suffixes = ["for industrial use", "retail pack", "grade A", "sample", "component part"]

    for _ in range(num_variations // 3):
         variations.add(f"{random.choice(prefixes)} {main_desc}")
         variations.add(f"{main_desc}, {random.choice(suffixes)}")

    # 3. Use qualifiers if available
    if qualifiers:
         for _ in range(num_variations // 3):
              variations.add(f"{main_desc} ({random.choice(qualifiers).strip()})")
              if len(qualifiers) > 1:
                   variations.add(f"{main_desc} ({random.choice(qualifiers).strip()}, {random.choice(qualifiers).strip()})") # Combine two

    # 4. Add generic variations
    generic_phrases = [
        f"Item described as: {main_desc}",
        f"Product: {main_desc}",
        f"Shipment contains {main_desc}",
        f"Commercial {main_desc}",
        f"Basic {main_desc}"
    ]
    variations.update(random.sample(generic_phrases, min(len(generic_phrases), num_variations // 4)))


    # 5. Shorten/Fragment variations (if long enough)
    if len(main_desc.split()) > 4:
         short_version = " ".join(main_desc.split()[:3]) + "..."
         variations.add(short_version)

    # Fill up to num_variations, adding slight modifications if needed to reach the count
    variation_list = list(variations)
    while len(variation_list) < num_variations:
         # Add slight variants if pool is too small
         base = random.choice(list(variations))
         variation_list.append(f"{base} type {random.randint(1,5)}")
         if len(variation_list) >= num_variations: break
         variation_list.append(f"high quality {base}")
         if len(variation_list) >= num_variations: break

    # Trim to exactly num_variations and pair with hs_code
    final_variations = [(desc, hs_code) for desc in variation_list[:num_variations]]
    return final_variations


# --- Generate Dataset ---
all_data = []
for index, row in selected_codes_df.iterrows():
    hs_code = row['hs6']
    description = row['description']
    variations = generate_variations(description, hs_code, EXAMPLES_PER_CODE)
    all_data.extend(variations)

# --- Create DataFrame and Save ---
df_output = pd.DataFrame(all_data, columns=['description', 'hs6'])

# Shuffle the dataset
df_output = df_output.sample(frac=1, random_state=42).reset_index(drop=True)

try:
    df_output.to_csv(OUTPUT_CSV_FILE, index=False, quoting=1) # quoting=1 ensures descriptions with commas are quoted
    print(f"\nSuccessfully generated and saved {len(df_output)} examples to '{OUTPUT_CSV_FILE}'.")
    print("\nSample rows:")
    print(df_output.head())
except Exception as e:
    print(f"\nError saving CSV file: {e}")
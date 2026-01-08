"""
Prepare Finnish-English dataset from separate files
Combines EUbookshop.fi and EUbookshop.en into tab-separated format
"""

import os
import random
from tqdm import tqdm

def prepare_dataset(fi_file, en_file, output_file, max_samples=100000, max_length=None):
    """
    Combine Finnish and English files into tab-separated format with no filtering
    """
    
    print(f"Reading files:")
    print(f"  Finnish: {fi_file}")
    print(f"  English: {en_file}")
    
    # Read Finnish sentences
    with open(fi_file, 'r', encoding='utf-8') as f:
        fi_sentences = [line.strip() for line in f if line.strip()]
    
    # Read English sentences
    with open(en_file, 'r', encoding='utf-8') as f:
        en_sentences = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(fi_sentences)} Finnish sentences")
    print(f"Loaded {len(en_sentences)} English sentences")
    
    # Ensure same number of sentences
    min_length = min(len(fi_sentences), len(en_sentences))
    fi_sentences = fi_sentences[:min_length]
    en_sentences = en_sentences[:min_length]
    
    print(f"Using {min_length} sentence pairs")
    
    # No filtering - use all pairs
    filtered_pairs = list(zip(fi_sentences, en_sentences))
    
    print(f"Total pairs: {len(filtered_pairs)}")
    
    # Only limit by max_samples if specified
    if max_samples and len(filtered_pairs) > max_samples:
        print(f"Randomly sampling {max_samples} pairs...")
        random.shuffle(filtered_pairs)
        filtered_pairs = filtered_pairs[:max_samples]
    
    # Write to output file
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for fi, en in tqdm(filtered_pairs):
            f.write(f"{fi}\t{en}\n")
    
    print(f"Dataset prepared: {len(filtered_pairs)} sentence pairs")
    return len(filtered_pairs)

def analyze_dataset(dataset_file):
    """Analyze the prepared dataset"""
    
    print(f"\nðŸ“Š Dataset Analysis: {dataset_file}")
    print("=" * 50)
    
    fi_lengths = []
    en_lengths = []
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                fi_lengths.append(len(parts[0].split()))
                en_lengths.append(len(parts[1].split()))
    
    print(f"Total sentences: {line_num}")
    print(f"Average Finnish length: {sum(fi_lengths)/len(fi_lengths):.1f} words")
    print(f"Average English length: {sum(en_lengths)/len(en_lengths):.1f} words") 
    print(f"Max Finnish length: {max(fi_lengths)} words")
    print(f"Max English length: {max(en_lengths)} words")
    print(f"Min Finnish length: {min(fi_lengths)} words")
    print(f"Min English length: {min(en_lengths)} words")

def main():
    # File paths - update these to match your files
    fi_file = "EUbookshop.fi"
    en_file = "EUbookshop.en" 
    output_file = "finnish_english_100k.txt"
    
    # Check if input files exist
    if not os.path.exists(fi_file):
        print(f"âŒ Finnish file not found: {fi_file}")
        print("Make sure EUbookshop.fi is in the current directory")
        return
    
    if not os.path.exists(en_file):
        print(f"âŒ English file not found: {en_file}")
        print("Make sure EUbookshop.en is in the current directory")
        return
    
    # Prepare dataset
    num_pairs = prepare_dataset(fi_file, en_file, output_file, 
                               max_samples=100000, max_length=150)
    
    # Analyze the prepared dataset
    analyze_dataset(output_file)
    
    print(f"\nðŸŽ¯ Next Steps:")
    print(f"1. Your dataset is ready: {output_file}")
    print(f"2. Update run_experiment.sh to use this file:")
    print(f"   DATA_PATH=\"{output_file}\"")
    print(f"3. Run the training: ./run_experiment.sh")

if __name__ == "__main__":
    random.seed(42)  # For reproducible sampling
    main()
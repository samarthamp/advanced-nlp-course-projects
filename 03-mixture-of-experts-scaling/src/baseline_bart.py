import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
import os

def generate_summaries_bart(model, tokenizer, dataset, device, max_len=64, batch_size=8):
    """Generate summaries using BART"""
    model.eval()
    
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Generating"):
            batch_end = min(i + batch_size, len(dataset))
            batch = dataset[i:batch_end]
            
            # Tokenize
            inputs = tokenizer(
                batch['document'],
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            # Generate
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_len,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_predictions.extend(predictions)
            all_references.extend(batch['summary'])
    
    return all_predictions, all_references


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading BART model...")
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-xsum')
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-xsum').to(device)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset('xsum', trust_remote_code=True)
    test_data = dataset['test']
    
    # Use subset if specified
    if args.num_samples > 0:
        test_data = test_data.select(range(min(args.num_samples, len(test_data))))
    
    print(f"Test size: {len(test_data)}")
    
    # Generate summaries
    print("Generating summaries...")
    predictions, references = generate_summaries_bart(
        model, tokenizer, test_data, device,
        args.max_len, args.batch_size
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        'predictions': predictions,
        'references': references
    }
    
    output_file = os.path.join(args.output_dir, 'predictions_bart.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print sample results
    print("\n=== Sample Results ===")
    for i in range(min(3, len(predictions))):
        print(f"\nSample {i+1}:")
        print(f"Reference: {references[i]}")
        print(f"Prediction: {predictions[i]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BART Baseline Inference')
    
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_len', type=int, default=64, help='Maximum generation length')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples (-1 for all)')
    
    args = parser.parse_args()
    
    main(args)
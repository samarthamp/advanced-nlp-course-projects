import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
import os

def generate_summaries_seq2seq(model, tokenizer, dataset, device, max_len=64, batch_size=8):
    """Generate summaries using seq2seq model (T5, Pegasus)"""
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
                max_length=256,
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


def generate_summaries_causal(model, tokenizer, dataset, device, max_len=64, batch_size=4):
    """Generate summaries using causal LM (Llama, Qwen)"""
    model.eval()
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Generating"):
            batch_end = min(i + batch_size, len(dataset))
            batch = dataset[i:batch_end]
            
            # Create prompts
            prompts = [
                f"Summarize the following article in one sentence:\n\n{doc}\n\nSummary:"
                for doc in batch['document']
            ]
            
            # Tokenize
            inputs = tokenizer(
                prompts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            # Generate
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_len,
                num_beams=4,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Decode (remove prompt)
            predictions = []
            for j, output in enumerate(outputs):
                # Skip the input prompt tokens
                generated_tokens = output[inputs['input_ids'][j].shape[0]:]
                summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                predictions.append(summary.strip())
            
            all_predictions.extend(predictions)
            all_references.extend(batch['summary'])
    
    return all_predictions, all_references


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    
    if args.model_type == 'seq2seq':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
    elif args.model_type == 'causal':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    print("Model loaded successfully")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset('xsum', trust_remote_code=True)
    test_data = dataset['test']
    
    if args.num_samples > 0:
        test_data = test_data.select(range(min(args.num_samples, len(test_data))))
    
    print(f"Test size: {len(test_data)}")
    
    # Generate summaries
    print("Generating summaries...")
    if args.model_type == 'seq2seq':
        predictions, references = generate_summaries_seq2seq(
            model, tokenizer, test_data, device, args.max_len, args.batch_size
        )
    else:
        predictions, references = generate_summaries_causal(
            model, tokenizer, test_data, device, args.max_len, args.batch_size
        )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        'predictions': predictions,
        'references': references
    }
    
    output_file = os.path.join(args.output_dir, f'predictions_{args.model_name}.json')
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
    parser = argparse.ArgumentParser(description='Inference with Fine-tuned Models')
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to fine-tuned model')
    parser.add_argument('--model_type', type=str, required=True, choices=['seq2seq', 'causal'], help='Model type')
    parser.add_argument('--model_name', type=str, required=True, help='Model name for output files')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_len', type=int, default=64, help='Maximum generation length')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples (-1 for all)')
    
    args = parser.parse_args()
    
    main(args)
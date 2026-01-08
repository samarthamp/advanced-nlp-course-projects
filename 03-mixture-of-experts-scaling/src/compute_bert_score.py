# compute_bertscore.py
import json
import torch
from bert_score import score
from datasets import load_dataset
from tqdm import tqdm

def compute_bertscore(predictions_file, output_file):
    """Compute BERTScore for predictions"""
    
    print(f"Loading predictions from {predictions_file}...")
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    predictions = data['predictions']
    references = data['references']
    
    print(f"Computing BERTScore for {len(predictions)} samples...")
    
    # Compute BERTScore
    # Using roberta-large model (recommended for English)
    P, R, F1 = score(
        predictions, 
        references, 
        lang="en", 
        model_type="microsoft/deberta-xlarge-mnli",
        verbose=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Calculate mean and std
    results = {
        "bertscore": {
            "precision": {
                "mean": float(P.mean()),
                "std": float(P.std())
            },
            "recall": {
                "mean": float(R.mean()),
                "std": float(R.std())
            },
            "f1": {
                "mean": float(F1.mean()),
                "std": float(F1.std())
            }
        }
    }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBERTScore Results:")
    print(f"  Precision: {results['bertscore']['precision']['mean']:.4f}")
    print(f"  Recall: {results['bertscore']['recall']['mean']:.4f}")
    print(f"  F1: {results['bertscore']['f1']['mean']:.4f}")
    print(f"\nResults saved to {output_file}")
    
    return results

if __name__ == "__main__":
    models = [
        'moe_topk',
        'moe_hash',
        'bart',
        't5_base',
        'llama_1b'
    ]
    
    for model in models:
        print(f"\n{'='*80}")
        print(f"Computing BERTScore for {model}")
        print('='*80)
        
        predictions_file = f'./results/predictions_{model}.json'
        output_file = f'./evaluations/bertscore_{model}.json'
        
        try:
            compute_bertscore(predictions_file, output_file)
        except Exception as e:
            print(f"Error processing {model}: {e}")
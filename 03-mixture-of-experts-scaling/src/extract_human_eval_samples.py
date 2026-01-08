# extract_human_eval_samples.py
import json
import random
from datasets import load_dataset

# Set seed for reproducibility
random.seed(42)

# Load dataset to get documents
print("Loading XSum dataset...")
dataset = load_dataset('xsum', trust_remote_code=True)
test_documents = dataset['test']['document']

# Models to evaluate
models = [
    ('predictions_hash.json', 'moe_hash'),
    ('predictions_bart.json', 'bart'),
    ('predictions_t5_base.json', 't5_base'),
    ('predictions_llama_1b.json', 'llama_1b')
]

# Extract samples for each model
for pred_file, model_name in models:
    print(f"\n{'='*80}")
    print(f"Processing {model_name}...")
    print('='*80)
    
    # Load predictions
    with open(f'./results/{pred_file}', 'r') as f:
        data = json.load(f)
    
    predictions = data['predictions']
    references = data['references']
    
    # Select 3 random indices
    total_samples = len(predictions)
    random_indices = random.sample(range(total_samples), 3)
    
    # Create human evaluation template
    samples = []
    for idx in random_indices:
        sample = {
            "index": idx,
            "document": test_documents[idx],
            "reference_summary": references[idx],
            "generated_summary": predictions[idx],
            "evaluation": {
                "content_relevance": {
                    "score": None,
                    "comments": ""
                },
                "coherence": {
                    "score": None,
                    "comments": ""
                },
                "fluency": {
                    "score": None,
                    "comments": ""
                },
                "factual_consistency": {
                    "score": None,
                    "comments": ""
                }
            }
        }
        samples.append(sample)
    
    # Save to file
    output_file = f'./evaluations/human_evaluation_template_{model_name}.json'
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"✓ Saved to {output_file}")
    
    # Print samples
    for i, sample in enumerate(samples, 1):
        print(f"\n--- Sample {i} (Index: {sample['index']}) ---")
        print(f"Document: {sample['document'][:200]}...")
        print(f"Reference: {sample['reference_summary']}")
        print(f"Prediction: {sample['generated_summary']}")

print("\n" + "="*80)
print("All human evaluation templates created!")
print("="*80)
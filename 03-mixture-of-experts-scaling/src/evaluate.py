import json
import argparse
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from tqdm import tqdm
import torch
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def compute_rouge_scores(predictions, references):
    """Compute ROUGE scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Computing ROUGE"):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': {
            'mean': np.mean(rouge1_scores),
            'std': np.std(rouge1_scores)
        },
        'rouge2': {
            'mean': np.mean(rouge2_scores),
            'std': np.std(rouge2_scores)
        },
        'rougeL': {
            'mean': np.mean(rougeL_scores),
            'std': np.std(rougeL_scores)
        }
    }


def compute_bleu_scores(predictions, references):
    """Compute BLEU scores"""
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Computing BLEU"):
        # Tokenize
        pred_tokens = word_tokenize(pred.lower())
        ref_tokens = word_tokenize(ref.lower())
        
        # Compute BLEU
        try:
            score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
            bleu_scores.append(score)
        except:
            bleu_scores.append(0.0)
    
    return {
        'mean': np.mean(bleu_scores),
        'std': np.std(bleu_scores)
    }


def compute_bertscore(predictions, references, batch_size=32):
    """Compute BERTScore"""
    print("Computing BERTScore...")
    P, R, F1 = bert_score(
        predictions, 
        references, 
        lang='en', 
        verbose=True,
        batch_size=batch_size,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    return {
        'precision': {
            'mean': P.mean().item(),
            'std': P.std().item()
        },
        'recall': {
            'mean': R.mean().item(),
            'std': R.std().item()
        },
        'f1': {
            'mean': F1.mean().item(),
            'std': F1.std().item()
        }
    }


def compute_compression_ratio(predictions, documents):
    """Compute compression ratio"""
    ratios = []
    
    for pred, doc in zip(predictions, documents):
        pred_len = len(word_tokenize(pred))
        doc_len = len(word_tokenize(doc))
        
        if doc_len > 0:
            ratio = pred_len / doc_len
            ratios.append(ratio)
    
    return {
        'mean': np.mean(ratios),
        'std': np.std(ratios)
    }


def compute_extractiveness(predictions, documents):
    """Compute extractiveness metric"""
    stop_words = set(stopwords.words('english'))
    extractiveness_scores = []
    
    for pred, doc in tqdm(zip(predictions, documents), total=len(predictions), desc="Computing Extractiveness"):
        # Tokenize and remove stopwords
        pred_tokens = set([w.lower() for w in word_tokenize(pred) if w.lower() not in stop_words and w.isalnum()])
        doc_tokens = set([w.lower() for w in word_tokenize(doc) if w.lower() not in stop_words and w.isalnum()])
        
        if len(pred_tokens) > 0:
            overlap = len(pred_tokens.intersection(doc_tokens))
            extractiveness = overlap / len(pred_tokens)
            extractiveness_scores.append(extractiveness)
    
    return {
        'mean': np.mean(extractiveness_scores),
        'std': np.std(extractiveness_scores)
    }


def compute_factuality_summac(predictions, documents):
    """
    Compute factuality using SummaC
    Note: This requires the summac library. Install with: pip install summac
    """
    try:
        from summac.model_summac import SummaCZS
        
        print("Loading SummaC model...")
        model = SummaCZS(granularity="sentence", model_name="vitc", device="cuda" if torch.cuda.is_available() else "cpu")
        
        print("Computing factuality scores...")
        scores = []
        batch_size = 8
        
        for i in tqdm(range(0, len(predictions), batch_size)):
            batch_preds = predictions[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            
            batch_scores = model.score(batch_docs, batch_preds)
            scores.extend(batch_scores['scores'])
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
    except ImportError:
        print("Warning: summac library not found. Skipping factuality computation.")
        print("Install with: pip install summac")
        return None
    except Exception as e:
        print(f"Error computing SummaC scores: {e}")
        return None


def human_evaluation_template(predictions, references, documents, num_samples=3):
    """
    Generate template for human evaluation
    """
    samples = []
    indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
    
    for idx in indices:
        sample = {
            'index': int(idx),
            'document': documents[idx],
            'reference_summary': references[idx],
            'generated_summary': predictions[idx],
            'evaluation': {
                'content_relevance': {
                    'score': None,  # To be filled manually (1-5)
                    'comments': ""
                },
                'coherence': {
                    'score': None,  # To be filled manually (1-5)
                    'comments': ""
                },
                'fluency': {
                    'score': None,  # To be filled manually (1-5)
                    'comments': ""
                },
                'factual_consistency': {
                    'score': None,  # To be filled manually (1-5)
                    'comments': ""
                }
            }
        }
        samples.append(sample)
    
    return samples


def main(args):
    # Load predictions
    print(f"Loading predictions from {args.predictions_file}...")
    with open(args.predictions_file, 'r') as f:
        data = json.load(f)
    
    predictions = data['predictions']
    references = data['references']
    
    # Load documents if needed for compression ratio and extractiveness
    if args.compute_compression or args.compute_extractiveness or args.compute_factuality:
        print("Loading documents from dataset...")
        from datasets import load_dataset
        dataset = load_dataset('EdinburghNLP/xsum')
        documents = dataset['test']['document'][:len(predictions)]
    else:
        documents = None
    
    print(f"Number of samples: {len(predictions)}")
    
    # Compute metrics
    results = {}
    
    # ROUGE scores
    if args.compute_rouge:
        print("\n=== Computing ROUGE Scores ===")
        results['rouge'] = compute_rouge_scores(predictions, references)
    
    # BLEU score
    if args.compute_bleu:
        print("\n=== Computing BLEU Score ===")
        results['bleu'] = compute_bleu_scores(predictions, references)
    
    # BERTScore
    if args.compute_bertscore:
        print("\n=== Computing BERTScore ===")
        import torch
        results['bertscore'] = compute_bertscore(predictions, references, args.batch_size)
    
    # Compression ratio
    if args.compute_compression and documents:
        print("\n=== Computing Compression Ratio ===")
        results['compression_ratio'] = compute_compression_ratio(predictions, documents)
    
    # Extractiveness
    if args.compute_extractiveness and documents:
        print("\n=== Computing Extractiveness ===")
        results['extractiveness'] = compute_extractiveness(predictions, documents)
    
    # Factuality (SummaC)
    if args.compute_factuality and documents:
        print("\n=== Computing Factuality (SummaC) ===")
        import torch
        factuality_scores = compute_factuality_summac(predictions, documents)
        if factuality_scores:
            results['factuality_summac'] = factuality_scores
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'evaluation_{args.model_name}.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(json.dumps(results, indent=2))
    
    # Generate human evaluation template
    if args.generate_human_eval and documents:
        print("\n=== Generating Human Evaluation Template ===")
        human_eval_samples = human_evaluation_template(
            predictions, references, documents, args.num_human_eval_samples
        )
        
        human_eval_file = os.path.join(args.output_dir, f'human_evaluation_template_{args.model_name}.json')
        with open(human_eval_file, 'w') as f:
            json.dump(human_eval_samples, f, indent=2)
        
        print(f"Human evaluation template saved to {human_eval_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Summarization Models')
    
    parser.add_argument('--predictions_file', type=str, required=True, help='Path to predictions JSON file')
    parser.add_argument('--model_name', type=str, default='model', help='Model name for output files')
    parser.add_argument('--output_dir', type=str, default='./evaluations', help='Output directory')
    
    # Metric flags
    parser.add_argument('--compute_rouge', action='store_true', default=True, help='Compute ROUGE scores')
    parser.add_argument('--compute_bleu', action='store_true', default=True, help='Compute BLEU score')
    parser.add_argument('--compute_bertscore', action='store_true', help='Compute BERTScore')
    parser.add_argument('--compute_compression', action='store_true', help='Compute compression ratio')
    parser.add_argument('--compute_extractiveness', action='store_true', help='Compute extractiveness')
    parser.add_argument('--compute_factuality', action='store_true', help='Compute factuality using SummaC')
    
    # Other parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for BERTScore')
    parser.add_argument('--generate_human_eval', action='store_true', help='Generate human evaluation template')
    parser.add_argument('--num_human_eval_samples', type=int, default=3, help='Number of samples for human evaluation')
    
    args = parser.parse_args()
    
    main(args)
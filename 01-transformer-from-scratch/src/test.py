import torch
import argparse
import os
import json
from tqdm import tqdm
import time
import numpy as np

from utils import *
from decoder import Transformer, DecodingStrategy

def batch_translate(model, sentences, src_tokenizer, tgt_tokenizer, device, 
                   decoding_strategy='greedy', batch_size=32, **kwargs):
    """Efficient batch translation for evaluation"""
    
    model.eval()
    all_translations = []
    
    decoder = DecodingStrategy(model, src_tokenizer, tgt_tokenizer, device, 
                             kwargs.get('max_length', 100))
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc=f"Batch {decoding_strategy}"):
            batch_sentences = sentences[i:i + batch_size]
            batch_translations = []
            
            # Process each sentence in the batch individually
            for sentence in batch_sentences:
                if decoding_strategy == 'greedy':
                    translation = decoder.greedy_decode(sentence)
                elif decoding_strategy == 'beam_search':
                    translation = decoder.beam_search_decode(
                        sentence, kwargs.get('beam_size', 5))
                elif decoding_strategy == 'top_k':
                    translation = decoder.top_k_sampling_decode(
                        sentence, kwargs.get('top_k', 50), kwargs.get('temperature', 1.0))
                else:
                    raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")
                
                batch_translations.append(translation)
            
            all_translations.extend(batch_translations)
    
    return all_translations

def evaluate_model(model, test_data, src_tokenizer, tgt_tokenizer, device, 
                  decoding_strategy='greedy', batch_size=32, **kwargs):
    """Evaluate model on test data with batch processing"""
    
    src_sentences = [item[0] for item in test_data]
    references = [item[1] for item in test_data]
    
    print(f"Evaluating with {decoding_strategy} decoding...")
    
    # Generate translations
    predictions = batch_translate(
        model, src_sentences, src_tokenizer, tgt_tokenizer, device,
        decoding_strategy, batch_size, **kwargs
    )
    
    # Calculate BLEU score
    bleu_score = calculate_bleu(references, predictions)
    
    return predictions, references, bleu_score

def analyze_translations(predictions, references, src_sentences, num_examples=20):
    """Analyze translation quality and provide insights"""
    
    analysis = {
        'length_stats': {},
        'quality_examples': [],
        'error_examples': []
    }
    
    # Length statistics
    pred_lengths = [len(pred.split()) for pred in predictions]
    ref_lengths = [len(ref.split()) for ref in references]
    
    analysis['length_stats'] = {
        'avg_pred_length': np.mean(pred_lengths),
        'avg_ref_length': np.mean(ref_lengths),
        'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths),
        'length_variance': np.var(pred_lengths)
    }
    
    # Find best and worst examples based on individual BLEU
    individual_bleus = []
    for ref, pred in zip(references, predictions):
        bleu = calculate_bleu([ref], [pred])
        individual_bleus.append(bleu)
    
    # Sort by BLEU score
    sorted_indices = sorted(range(len(individual_bleus)), 
                           key=lambda i: individual_bleus[i], reverse=True)
    
    # Best examples
    for i in sorted_indices[:num_examples//2]:
        analysis['quality_examples'].append({
            'source': src_sentences[i],
            'reference': references[i],
            'prediction': predictions[i],
            'bleu': individual_bleus[i]
        })
    
    # Worst examples  
    for i in sorted_indices[-num_examples//2:]:
        analysis['error_examples'].append({
            'source': src_sentences[i],
            'reference': references[i],
            'prediction': predictions[i],
            'bleu': individual_bleus[i]
        })
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description='Test Transformer for Machine Translation')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing tokenizers')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data')
    
    # Model architecture parameters (should match training)
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--positional_encoding', type=str, choices=['rope', 'relative'], 
                       default='rope', help='Positional encoding type')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    
    # Decoding parameters
    parser.add_argument('--decoding_strategy', type=str, 
                       choices=['greedy', 'beam_search', 'top_k', 'all'], 
                       default='all', help='Decoding strategy')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search')
    parser.add_argument('--top_k', type=int, default=50, help='K for top-k sampling')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--decode_max_length', type=int, default=100, help='Max length for decoding')
    
    # Performance parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--num_examples', type=int, default=20, help='Number of examples to show')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum test samples to use (None for full test set)')
    parser.add_argument('--detailed_analysis', action='store_true', help='Perform detailed analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Load tokenizers
    print("Loading tokenizers...")
    src_tokenizer = Tokenizer()
    tgt_tokenizer = Tokenizer()
    
    src_tokenizer.load(os.path.join(args.model_dir, 'src_tokenizer.pkl'))
    tgt_tokenizer.load(os.path.join(args.model_dir, 'tgt_tokenizer.pkl'))
    
    print(f"Source vocabulary size: {src_tokenizer.vocab_size}")
    print(f"Target vocabulary size: {tgt_tokenizer.vocab_size}")
    
    # Load test data
    print("Loading test data...")
    src_sentences, tgt_sentences = load_data(args.data_path, args.max_samples)
    
    # Use the test split (assuming the data file contains all data)
    _, _, (test_src, test_tgt) = split_data(src_sentences, tgt_sentences)
    test_data = list(zip(test_src, test_tgt))
    #(train_src, train_tgt), _, _ = split_data(src_sentences, tgt_sentences)
    #test_data = list(zip(train_src, train_tgt))
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    print(f"Test samples: {len(test_data)}")
    
    # Create model
    print("Creating model...")
    model = Transformer(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        max_length=args.max_length,
        dropout=args.dropout,
        positional_encoding=args.positional_encoding
    ).to(device)
    
    # Load trained weights
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle DataParallel models
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('module.') for key in state_dict.keys()):
        # Remove 'module.' prefix from DataParallel
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict)
    print(f"Model loaded from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    
    # Model info for report
    model_info = {
        'model_path': args.model_path,
        'positional_encoding': args.positional_encoding,
        'architecture': {
            'd_model': args.d_model,
            'num_heads': args.num_heads,
            'num_encoder_layers': args.num_encoder_layers,
            'num_decoder_layers': args.num_decoder_layers,
            'd_ff': args.d_ff,
            'max_length': args.max_length
        },
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'checkpoint_epoch': checkpoint['epoch'],
        'checkpoint_loss': checkpoint['loss']
    }
    
    test_info = {
        'num_samples': len(test_data),
        'data_path': args.data_path,
        'max_samples': args.max_samples,
        'batch_size': args.batch_size
    }
    
    # Evaluation
    results = {}
    
    if args.decoding_strategy == 'all':
        strategies = ['greedy', 'beam_search', 'top_k']
    else:
        strategies = [args.decoding_strategy]
    
    print(f"\n{'='*60}")
    print(f"STARTING EVALUATION ON {len(test_data)} SAMPLES")
    print(f"{'='*60}")
    
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Evaluating with {strategy.upper()} decoding")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # Evaluation parameters based on strategy
        eval_kwargs = {
            'max_length': args.decode_max_length
        }
        
        if strategy == 'beam_search':
            eval_kwargs['beam_size'] = args.beam_size
        elif strategy == 'top_k':
            eval_kwargs['top_k'] = args.top_k
            eval_kwargs['temperature'] = args.temperature
        
        predictions, references, bleu_score = evaluate_model(
            model, test_data, src_tokenizer, tgt_tokenizer, device,
            decoding_strategy=strategy,
            batch_size=args.batch_size,
            **eval_kwargs
        )
        
        end_time = time.time()
        eval_time = end_time - start_time
        
        print(f"\n{strategy.upper()} Results:")
        print(f"BLEU Score: {bleu_score:.4f}")
        print(f"Evaluation time: {eval_time:.2f} seconds")
        print(f"Average time per sentence: {eval_time/len(test_data):.3f} seconds")
        print(f"Sentences per second: {len(test_data)/eval_time:.1f}")
        
        # Store results
        results[strategy] = {
            'bleu_score': bleu_score,
            'evaluation_time': eval_time,
            'predictions': predictions,
            'references': references,
            'parameters': eval_kwargs
        }
        
        # Detailed analysis if requested
        if args.detailed_analysis:
            print("Performing detailed analysis...")
            src_sentences = [item[0] for item in test_data]
            analysis = analyze_translations(predictions, references, src_sentences, args.num_examples)
            results[strategy]['analysis'] = analysis
            
            print(f"Average prediction length: {analysis['length_stats']['avg_pred_length']:.1f}")
            print(f"Average reference length: {analysis['length_stats']['avg_ref_length']:.1f}")
            print(f"Length ratio (pred/ref): {analysis['length_stats']['length_ratio']:.3f}")
        
        # Show examples
        print(f"\nExample translations ({strategy}):")
        print("-" * 80)
        for i in range(min(args.num_examples, len(test_data))):
            src_sentence = test_data[i][0]
            reference = references[i]
            prediction = predictions[i]
            
            print(f"Source:     {src_sentence}")
            print(f"Reference:  {reference}")
            print(f"Prediction: {prediction}")
            
            if args.detailed_analysis:
                individual_bleu = calculate_bleu([reference], [prediction])
                print(f"BLEU:       {individual_bleu:.4f}")
            
            print("-" * 80)
        
        # Save detailed results for this strategy
        strategy_results = {
            'strategy': strategy,
            'bleu_score': bleu_score,
            'evaluation_time': eval_time,
            'parameters': eval_kwargs,
            'model_info': model_info,
            'test_info': test_info,
            'examples': []
        }
        
        # Add analysis if available
        if args.detailed_analysis:
            strategy_results['analysis'] = analysis
        
        for i in range(len(test_data)):
            example = {
                'source': test_data[i][0],
                'reference': references[i],
                'prediction': predictions[i]
            }
            
            if args.detailed_analysis:
                example['individual_bleu'] = calculate_bleu([references[i]], [predictions[i]])
            
            strategy_results['examples'].append(example)
        
        # Save to file
        output_file = os.path.join(args.output_dir, f'{strategy}_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(strategy_results, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed results saved to: {output_file}")
    
    # Create comparison table
    print(f"\n{'='*70}")
    print("COMPARISON OF DECODING STRATEGIES")
    print(f"{'='*70}")
    print(f"{'Strategy':<15} {'BLEU':<8} {'Time (s)':<10} {'Sent/s':<8} {'Params':<20}")
    print("-" * 70)
    
    for strategy in strategies:
        result = results[strategy]
        time_per_sent = result['evaluation_time'] / len(test_data)
        sent_per_sec = len(test_data) / result['evaluation_time']
        
        # Parameter summary
        if strategy == 'beam_search':
            params = f"beam_size={result['parameters']['beam_size']}"
        elif strategy == 'top_k':
            params = f"k={result['parameters']['top_k']}, T={result['parameters']['temperature']}"
        else:
            params = "-"
        
        print(f"{strategy.upper():<15} {result['bleu_score']:<8.4f} "
              f"{result['evaluation_time']:<10.2f} {sent_per_sec:<8.1f} {params:<20}")
    
    # Save comparison results
    comparison = {
        'model_info': model_info,
        'test_info': test_info,
        'evaluation_summary': {},
        'performance_analysis': {}
    }
    
    for strategy in strategies:
        comparison['evaluation_summary'][strategy] = {
            'bleu_score': results[strategy]['bleu_score'],
            'evaluation_time': results[strategy]['evaluation_time'],
            'sentences_per_second': len(test_data) / results[strategy]['evaluation_time'],
            'parameters': results[strategy]['parameters']
        }
    
    # Performance analysis
    if len(strategies) > 1:
        best_bleu_strategy = max(strategies, key=lambda s: results[s]['bleu_score'])
        fastest_strategy = min(strategies, key=lambda s: results[s]['evaluation_time'])
        
        comparison['performance_analysis'] = {
            'best_bleu_strategy': best_bleu_strategy,
            'best_bleu_score': results[best_bleu_strategy]['bleu_score'],
            'fastest_strategy': fastest_strategy,
            'fastest_time': results[fastest_strategy]['evaluation_time'],
            'quality_speed_tradeoff': {
                'bleu_difference': results[best_bleu_strategy]['bleu_score'] - results[fastest_strategy]['bleu_score'],
                'time_difference': results[best_bleu_strategy]['evaluation_time'] - results[fastest_strategy]['evaluation_time'],
                'efficiency_ratio': (results[best_bleu_strategy]['bleu_score'] / results[fastest_strategy]['bleu_score']) / 
                                   (results[best_bleu_strategy]['evaluation_time'] / results[fastest_strategy]['evaluation_time'])
            }
        }
        
        print(f"\n{'='*60}")
        print("PERFORMANCE ANALYSIS")
        print(f"{'='*60}")
        print(f"Best BLEU score: {best_bleu_strategy.upper()} "
              f"({results[best_bleu_strategy]['bleu_score']:.4f})")
        print(f"Fastest decoding: {fastest_strategy.upper()} "
              f"({results[fastest_strategy]['evaluation_time']:.2f}s)")
        
        if best_bleu_strategy != fastest_strategy:
            bleu_diff = results[best_bleu_strategy]['bleu_score'] - results[fastest_strategy]['bleu_score']
            time_diff = results[best_bleu_strategy]['evaluation_time'] - results[fastest_strategy]['evaluation_time']
            print(f"\nQuality-Speed Trade-off:")
            print(f"  {best_bleu_strategy.upper()} gives {bleu_diff:.4f} higher BLEU (+{bleu_diff/results[fastest_strategy]['bleu_score']*100:.1f}%)")
            print(f"  but takes {time_diff:.2f}s longer (+{time_diff/results[fastest_strategy]['evaluation_time']*100:.1f}%)")
            
            efficiency = comparison['performance_analysis']['quality_speed_tradeoff']['efficiency_ratio']
            if efficiency > 1.0:
                print(f"  {best_bleu_strategy.upper()} has better quality/speed efficiency: {efficiency:.2f}x")
            else:
                print(f"  {fastest_strategy.upper()} has better quality/speed efficiency: {1/efficiency:.2f}x")
    
    comparison_file = os.path.join(args.output_dir, 'comparison_results.json')
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\nComparison results saved to: {comparison_file}")
    
    print(f"\nEvaluation completed! Results saved in: {args.output_dir}")

if __name__ == '__main__':
    main()
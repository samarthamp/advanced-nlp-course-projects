import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
import os
from moe_transformer import MoETransformer

def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    model = MoETransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args['d_model'],
        nhead=args['nhead'],
        num_encoder_layers=args['num_encoder_layers'],
        num_decoder_layers=args['num_decoder_layers'],
        d_ff=args['d_ff'],
        num_experts=args['num_experts'],
        top_k=args['top_k'],
        router_type=args['router_type'],
        dropout=args['dropout'],
        max_len=args['max_src_len'],
        use_load_balancing=args.get('use_load_balancing', True),
        pad_token_id=tokenizer.pad_token_id
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer


def greedy_decode(model, src, src_mask, max_len, start_token, end_token, device):
    """Greedy decoding"""
    batch_size = src.size(0)
    
    # Encode
    memory, _ = model.encode(src, src_mask)
    
    # Initialize decoder input
    ys = torch.ones(batch_size, 1).fill_(start_token).long().to(device)
    
    for i in range(max_len - 1):
        # Create target mask
        tgt_mask = model.generate_square_subsequent_mask(ys.size(1)).to(device)
        
        # Decode
        out, _ = model.decode(ys, memory, tgt_mask, src_mask, None)
        
        # Project to vocabulary
        logits = model.output_projection(out[:, -1, :])
        
        # Get next token
        next_token = logits.argmax(dim=-1, keepdim=True)
        
        # Append to output
        ys = torch.cat([ys, next_token], dim=1)
        
        # Check if all sequences have generated end token
        if (next_token == end_token).all():
            break
    
    return ys


def beam_search_decode(model, src, src_mask, max_len, start_token, end_token, device, beam_size=5):
    """Beam search decoding"""
    batch_size = src.size(0)
    
    # Encode
    memory, _ = model.encode(src, src_mask)
    
    # Expand for beam search
    memory = memory.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(batch_size * beam_size, -1, model.d_model)
    if src_mask is not None:
        src_mask = src_mask.unsqueeze(1).expand(-1, beam_size, -1).reshape(batch_size * beam_size, -1)
    
    # Initialize beams
    beams = torch.ones(batch_size, beam_size, 1).fill_(start_token).long().to(device)
    beam_scores = torch.zeros(batch_size, beam_size).to(device)
    beam_scores[:, 1:] = float('-inf')  # Only first beam is active initially
    
    finished_beams = [[] for _ in range(batch_size)]
    
    for step in range(max_len - 1):
        # Flatten beams for batch processing
        flat_beams = beams.view(batch_size * beam_size, -1)
        
        # Create target mask
        tgt_mask = model.generate_square_subsequent_mask(flat_beams.size(1)).to(device)
        
        # Decode
        out, _ = model.decode(flat_beams, memory, tgt_mask, src_mask, None)
        
        # Project to vocabulary
        logits = model.output_projection(out[:, -1, :])
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Reshape back
        log_probs = log_probs.view(batch_size, beam_size, -1)
        
        # Compute scores for all possible next tokens
        scores = beam_scores.unsqueeze(-1) + log_probs  # [batch_size, beam_size, vocab_size]
        
        # Flatten and get top-k
        scores_flat = scores.view(batch_size, -1)  # [batch_size, beam_size * vocab_size]
        top_scores, top_indices = torch.topk(scores_flat, beam_size, dim=-1)
        
        # Convert flat indices to beam and token indices
        beam_indices = top_indices // log_probs.size(-1)
        token_indices = top_indices % log_probs.size(-1)
        
        # Update beams
        new_beams = []
        new_scores = []
        
        for b in range(batch_size):
            new_beam = []
            new_score = []
            
            for k in range(beam_size):
                beam_idx = beam_indices[b, k]
                token_idx = token_indices[b, k]
                score = top_scores[b, k]
                
                # Check if beam is finished
                if token_idx == end_token:
                    finished_beams[b].append((beams[b, beam_idx].clone(), score.item()))
                    # Add dummy beam
                    new_beam.append(beams[b, 0].clone())
                    new_score.append(float('-inf'))
                else:
                    # Append token to beam
                    new_seq = torch.cat([beams[b, beam_idx], token_idx.unsqueeze(0)])
                    new_beam.append(new_seq)
                    new_score.append(score)
            
            new_beams.append(torch.stack(new_beam))
            new_scores.append(torch.stack(new_score))
        
        beams = torch.stack(new_beams)
        beam_scores = torch.stack(new_scores)
        
        # Check if all beams are finished
        if all(len(fb) >= beam_size for fb in finished_beams):
            break
    
    # Select best beam for each batch
    results = []
    for b in range(batch_size):
        if finished_beams[b]:
            # Select beam with highest score
            best_beam = max(finished_beams[b], key=lambda x: x[1])
            results.append(best_beam[0])
        else:
            # Select beam with highest current score
            best_idx = beam_scores[b].argmax()
            results.append(beams[b, best_idx])
    
    # Pad to same length
    max_result_len = max(r.size(0) for r in results)
    padded_results = []
    for r in results:
        pad_len = max_result_len - r.size(0)
        if pad_len > 0:
            r = torch.cat([r, torch.full((pad_len,), end_token, dtype=r.dtype, device=r.device)])
        padded_results.append(r)
    
    return torch.stack(padded_results)


def generate_summaries(model, tokenizer, dataset, device, max_len=64, batch_size=8, use_beam_search=False, beam_size=5):
    """Generate summaries for entire dataset"""
    model.eval()
    
    all_predictions = []
    all_references = []
    
    # Reset expert usage tracking
    model.reset_all_expert_usage()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Generating"):
            batch_end = min(i + batch_size, len(dataset))
            batch = dataset[i:batch_end]
            
            # Tokenize sources
            src_encodings = tokenizer(
                batch['document'],
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            src_input_ids = src_encodings['input_ids'].to(device)
            src_attention_mask = src_encodings['attention_mask'].to(device)
            src_key_padding_mask = (src_attention_mask == 0)
            
            # Generate
            if use_beam_search:
                outputs = beam_search_decode(
                    model, src_input_ids, src_key_padding_mask,
                    max_len, tokenizer.bos_token_id or tokenizer.cls_token_id,
                    tokenizer.eos_token_id or tokenizer.sep_token_id,
                    device, beam_size
                )
            else:
                outputs = greedy_decode(
                    model, src_input_ids, src_key_padding_mask,
                    max_len, tokenizer.bos_token_id or tokenizer.cls_token_id,
                    tokenizer.eos_token_id or tokenizer.sep_token_id,
                    device
                )
            
            # Decode
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_predictions.extend(predictions)
            all_references.extend(batch['summary'])
    
    # Get expert usage statistics
    expert_usage = model.get_all_expert_usage()
    
    return all_predictions, all_references, expert_usage


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model(args.checkpoint_path, device)
    
    print(f"Model loaded from {args.checkpoint_path}")
    print(f"Router type: {args.checkpoint_path.split('_')[-1].replace('.pt', '')}")
    
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
    predictions, references, expert_usage = generate_summaries(
        model, tokenizer, test_data, device,
        args.max_len, args.batch_size,
        args.use_beam_search, args.beam_size
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        'predictions': predictions,
        'references': references,
        'expert_usage': {
            'encoder': [usage.tolist() for usage in expert_usage['encoder']],
            'decoder': [usage.tolist() for usage in expert_usage['decoder']]
        }
    }
    
    output_file = os.path.join(args.output_dir, f'predictions_{args.router_type}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print sample results
    print("\n=== Sample Results ===")
    for i in range(min(3, len(predictions))):
        print(f"\nSample {i+1}:")
        print(f"Document: {test_data[i]['document'][:200]}...")
        print(f"Reference: {references[i]}")
        print(f"Prediction: {predictions[i]}")
    
    # Print expert usage
    print("\n=== Expert Usage Statistics ===")
    for layer_type in ['encoder', 'decoder']:
        print(f"\n{layer_type.capitalize()} Layers:")
        for i, usage in enumerate(expert_usage[layer_type]):
            print(f"  Layer {i}: {usage}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference with MoE Transformer')
    
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_len', type=int, default=64, help='Maximum generation length')
    parser.add_argument('--max_src_len', type=int, default=512, help='Maximum source length')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to evaluate (-1 for all)')
    parser.add_argument('--use_beam_search', action='store_true', help='Use beam search instead of greedy decoding')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search')
    parser.add_argument('--router_type', type=str, default='topk', help='Router type (for naming output files)')
    
    args = parser.parse_args()
    
    main(args)
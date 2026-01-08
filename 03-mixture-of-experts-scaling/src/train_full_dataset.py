import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
from moe_transformer import MoETransformer
import gc

class StreamingDataset(IterableDataset):
    """Streaming dataset to avoid loading everything in RAM"""
    def __init__(self, dataset, tokenizer, max_src_len=256, max_tgt_len=48):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
    def __iter__(self):
        for item in self.dataset:
            # Tokenize on-the-fly
            src_encoding = self.tokenizer(
                item['document'],
                max_length=self.max_src_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            tgt_encoding = self.tokenizer(
                item['summary'],
                max_length=self.max_tgt_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            yield {
                'src_input_ids': src_encoding['input_ids'].squeeze(0),
                'src_attention_mask': src_encoding['attention_mask'].squeeze(0),
                'tgt_input_ids': tgt_encoding['input_ids'].squeeze(0),
                'tgt_attention_mask': tgt_encoding['attention_mask'].squeeze(0)
            }


def create_padding_mask(attention_mask):
    """Create padding mask from attention mask"""
    return attention_mask == 0


def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, 
                load_balance_weight=0.01, grad_clip=1.0, gradient_accumulation_steps=1,
                epoch=0, total_steps=0):
    """Memory-optimized training for one epoch"""
    model.train()
    total_loss = 0
    total_lm_loss = 0
    total_lb_loss = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    # For full dataset, we don't know exact length, so we'll use a counter
    progress_bar = tqdm(enumerate(dataloader), desc=f"Training Epoch {epoch+1}")
    
    for batch_idx, batch in progress_bar:
        # Move to device
        src_input_ids = batch['src_input_ids'].to(device)
        src_attention_mask = batch['src_attention_mask'].to(device)
        tgt_input_ids = batch['tgt_input_ids'].to(device)
        tgt_attention_mask = batch['tgt_attention_mask'].to(device)
        
        # Prepare decoder input (shift right)
        decoder_input_ids = tgt_input_ids[:, :-1]
        decoder_attention_mask = tgt_attention_mask[:, :-1]
        labels = tgt_input_ids[:, 1:]
        
        # Create padding masks
        src_key_padding_mask = create_padding_mask(src_attention_mask)
        tgt_key_padding_mask = create_padding_mask(decoder_attention_mask)
        
        # Forward pass with mixed precision
        with autocast():
            logits, lb_loss = model(
                src_input_ids,
                decoder_input_ids,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            
            # Compute language modeling loss
            lm_loss = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            
            # Total loss
            loss = lm_loss
            if lb_loss is not None and model.use_load_balancing:
                loss = loss + load_balance_weight * lb_loss
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
        
        # Backward pass with scaled loss
        scaler.scale(loss).backward()
        
        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Unscale before clipping
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
            
            total_steps += 1
        
        total_loss += loss.item() * gradient_accumulation_steps
        total_lm_loss += lm_loss.item()
        if lb_loss is not None:
            total_lb_loss += lb_loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item() * gradient_accumulation_steps,
            'lm_loss': lm_loss.item(),
            'lb_loss': lb_loss.item() if lb_loss is not None else 0.0,
            'lr': optimizer.param_groups[0]['lr'],
            'batches': num_batches
        })
        
        # Periodic memory cleanup
        if batch_idx % 100 == 0 and batch_idx > 0:
            clear_memory()
        
        # Delete intermediate tensors
        del src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask
        del decoder_input_ids, decoder_attention_mask, labels, logits, loss, lm_loss
        if lb_loss is not None:
            del lb_loss
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_lm_loss = total_lm_loss / num_batches if num_batches > 0 else 0
    avg_lb_loss = total_lb_loss / num_batches if num_batches > 0 else 0
    
    return avg_loss, avg_lm_loss, avg_lb_loss, total_steps


def validate(model, dataloader, device, max_batches=None):
    """Memory-optimized validation"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), desc="Validation")
        
        for batch_idx, batch in progress_bar:
            if max_batches and batch_idx >= max_batches:
                break
                
            # Move to device
            src_input_ids = batch['src_input_ids'].to(device)
            src_attention_mask = batch['src_attention_mask'].to(device)
            tgt_input_ids = batch['tgt_input_ids'].to(device)
            tgt_attention_mask = batch['tgt_attention_mask'].to(device)
            
            # Prepare decoder input
            decoder_input_ids = tgt_input_ids[:, :-1]
            decoder_attention_mask = tgt_attention_mask[:, :-1]
            labels = tgt_input_ids[:, 1:]
            
            # Create padding masks
            src_key_padding_mask = create_padding_mask(src_attention_mask)
            tgt_key_padding_mask = create_padding_mask(decoder_attention_mask)
            
            # Forward pass
            with autocast():
                logits, _ = model(
                    src_input_ids,
                    decoder_input_ids,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask
                )
                
                # Compute loss
                loss = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )
            
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'val_loss': loss.item(), 'batches': num_batches})
            
            # Periodic memory cleanup
            if batch_idx % 50 == 0 and batch_idx > 0:
                clear_memory()
            
            # Delete intermediate tensors
            del src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask
            del decoder_input_ids, decoder_attention_mask, labels, logits, loss
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset - streaming mode for memory efficiency
    print("Loading dataset in streaming mode...")
    dataset = load_dataset('xsum', trust_remote_code=True)
    
    train_data = dataset['train']
    val_data = dataset['validation']
    
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    print(f"Note: Using streaming to minimize RAM usage")
    
    # Create streaming datasets
    train_dataset = StreamingDataset(train_data, tokenizer, args.max_src_len, args.max_tgt_len)
    val_dataset = StreamingDataset(val_data, tokenizer, args.max_src_len, args.max_tgt_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,  # Must be 0 for IterableDataset
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False
    )
    
    # Create model
    print("Creating model...")
    model = MoETransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        num_experts=args.num_experts,
        top_k=args.top_k,
        router_type=args.router_type,
        dropout=args.dropout,
        max_len=args.max_src_len,
        use_load_balancing=args.use_load_balancing,
        pad_token_id=tokenizer.pad_token_id
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (fp32)")
    
    # Estimate memory usage
    estimated_vram = (total_params * 4 + args.batch_size * args.max_src_len * args.d_model * 8) / 1024**3
    print(f"Estimated VRAM usage: {estimated_vram:.2f} GB")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with warmup
    # Estimate total steps (204k samples / batch_size / grad_accum)
    estimated_steps_per_epoch = len(train_data) // (args.batch_size * args.gradient_accumulation_steps)
    total_steps = estimated_steps_per_epoch * args.num_epochs
    warmup_steps = int(0.1 * total_steps)
    
    print(f"Estimated steps per epoch: {estimated_steps_per_epoch}")
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, (total_steps - step) / (total_steps - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Initialize training variables
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    total_training_steps = 0
    
    # Resume from checkpoint if specified
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\n{'='*80}")
        print(f"Resuming from checkpoint: {args.resume_from}")
        print(f"{'='*80}")
        
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        total_training_steps = checkpoint.get('total_steps', 0)
        
        # Load training history if exists
        history_path = os.path.join(args.output_dir, f'training_history_{args.router_type}.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
                train_losses = history.get('train_losses', [])
                val_losses = history.get('val_losses', [])
        
        print(f"Resuming from epoch {start_epoch} (completed epoch {checkpoint['epoch']})")
        print(f"Previous best validation loss: {best_val_loss:.4f}")
        print(f"Total steps completed: {total_training_steps}")
        print(f"{'='*80}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if start_epoch == 0:
        print("\nStarting training on FULL dataset...")
        print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
        print(f"This will take approximately {args.num_epochs * 12} to {args.num_epochs * 15} hours")
    else:
        print(f"\nResuming training from epoch {start_epoch}/{args.num_epochs}")
        remaining_epochs = args.num_epochs - start_epoch
        print(f"Approximately {remaining_epochs * 12} to {remaining_epochs * 15} hours remaining")
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_lm_loss, train_lb_loss, total_training_steps = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            args.load_balance_weight, args.grad_clip, args.gradient_accumulation_steps,
            epoch, total_training_steps
        )
        
        # Clear memory before validation
        clear_memory()
        
        # Validate (on subset to save time - full validation would take too long)
        print("\nRunning validation (on subset)...")
        val_loss = validate(model, val_loader, device, max_batches=args.val_batches)
        
        # Log
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (LM: {train_lm_loss:.4f}, LB: {train_lb_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save checkpoint
        if val_loss < best_val_loss or epoch == start_epoch:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'total_steps': total_training_steps,
                'args': vars(args)
            }
            checkpoint_path = os.path.join(args.output_dir, f'best_model_{args.router_type}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved best model to {checkpoint_path}")
        
        # Save regular checkpoint every epoch (to recover from crashes)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'total_steps': total_training_steps,
            'args': vars(args)
        }
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}_{args.router_type}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Saved checkpoint to {checkpoint_path}")
        
        # Save training history after each epoch
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'completed_epochs': epoch + 1,
            'total_steps': total_training_steps
        }
        history_path = os.path.join(args.output_dir, f'training_history_{args.router_type}.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Memory cleanup
        clear_memory()
        
        print(f"\nCompleted {epoch + 1}/{args.num_epochs} epochs")
        print(f"Best validation loss so far: {best_val_loss:.4f}")
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total training steps: {total_training_steps}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoE Transformer on FULL XSum Dataset (8GB VRAM Optimized)')
    
    # Model hyperparameters - TINY configuration for 8GB VRAM
    parser.add_argument('--d_model', type=int, default=96, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=2, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=384, help='FFN dimension')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts')
    parser.add_argument('--top_k', type=int, default=2, help='Number of experts to use per token')
    parser.add_argument('--router_type', type=str, default='topk', choices=['topk', 'hash'], help='Router type')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--use_load_balancing', action='store_true', default=True, help='Use load balancing loss')
    parser.add_argument('--load_balance_weight', type=float, default=0.01, help='Load balancing loss weight')
    
    # Training hyperparameters - optimized for 8GB VRAM + full dataset
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (keep small!)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs (3 is enough for full dataset)')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--max_src_len', type=int, default=256, help='Maximum source length')
    parser.add_argument('--max_tgt_len', type=int, default=48, help='Maximum target length')
    parser.add_argument('--val_batches', type=int, default=200, help='Number of validation batches')
    
    # Resume training
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Other
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    main(args)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import math

from utils import *
from decoder import Transformer
from encoder import Encoder

def setup_cluster_training():
    """Setup optimizations for cluster training"""
    
    # Enable optimized CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Set optimal number of threads
    torch.set_num_threads(min(16, os.cpu_count()))
    
    # Enable TensorFloat-32 for A100/V100 GPUs (massive speedup)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

def create_optimized_dataloader(dataset, batch_size, shuffle=True, num_workers=None):
    """Create optimized DataLoader for cluster training"""
    
    if num_workers is None:
        num_workers = min(16, os.cpu_count())  # Use more workers on cluster
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfers
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4,  # Prefetch more batches
        collate_fn=collate_fn
    )

def setup_model_for_cluster(model, device):
    """Setup model for multi-GPU training if available"""
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    return model

class CosineAnnealingWarmupScheduler:
    """Advanced LR scheduler with warmup and cosine annealing"""
    
    def __init__(self, optimizer, warmup_steps, max_steps, max_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    src_batch = torch.stack([item['src'] for item in batch])
    tgt_batch = torch.stack([item['tgt'] for item in batch])
    target_batch = torch.stack([item['target'] for item in batch])
    
    return {
        'src': src_batch,
        'tgt': tgt_batch,
        'target': target_batch
    }

def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None, scaler=None):
    """Train for one epoch with mixed precision"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        src = batch['src'].to(device, non_blocking=True)
        tgt = batch['tgt'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if scaler:
            with torch.cuda.amp.autocast():
                output, _ = model(src, tgt)
                loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward pass
            output, _ = model(src, tgt)
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            
            # Standard backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Memory cleanup every 100 steps
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / num_batches

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            src = batch['src'].to(device, non_blocking=True)
            tgt = batch['tgt'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            
            # Forward pass
            output, _ = model(src, tgt)
            
            # Calculate loss
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            total_loss += loss.item()
    
    return total_loss / num_batches

def check_gpu_memory():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3      # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

def main():
    parser = argparse.ArgumentParser(description='Train Transformer for Machine Translation')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--positional_encoding', type=str, choices=['rope', 'relative'], 
                       default='rope', help='Positional encoding type')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=8000, help='Warmup steps for scheduler')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--min_freq', type=int, default=2, help='Minimum word frequency for vocab')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of data loading workers')
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use')
    
    # Output parameters
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--save_every', type=int, default=5, help='Save model every N epochs')
    
    # Advanced options
    parser.add_argument('--use_cosine_schedule', action='store_true', help='Use cosine annealing scheduler')
    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Setup cluster optimizations
    setup_cluster_training()
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and split data
    print("Loading data...")
    src_sentences, tgt_sentences = load_data(args.data_path, args.max_samples)
    (train_src, train_tgt), (val_src, val_tgt), (test_src, test_tgt) = split_data(
        src_sentences, tgt_sentences)
    
    print(f"Train samples: {len(train_src)}")
    print(f"Validation samples: {len(val_src)}")
    print(f"Test samples: {len(test_src)}")
    
    # Build or load vocabularies based on resume flag
    if args.resume:
        # Load existing tokenizers when resuming
        print("Loading saved tokenizers for resume...")
        src_tokenizer = Tokenizer()
        tgt_tokenizer = Tokenizer()
        src_tokenizer.load(os.path.join(args.model_dir, 'src_tokenizer.pkl'))
        tgt_tokenizer.load(os.path.join(args.model_dir, 'tgt_tokenizer.pkl'))
    else:
        # Build new vocabularies for fresh training
        print("Building vocabularies...")
        src_tokenizer = Tokenizer()
        tgt_tokenizer = Tokenizer()
        src_tokenizer.build_vocab(train_src, args.min_freq)
        tgt_tokenizer.build_vocab(train_tgt, args.min_freq)
        
        # Save tokenizers
        src_tokenizer.save(os.path.join(args.model_dir, 'src_tokenizer.pkl'))
        tgt_tokenizer.save(os.path.join(args.model_dir, 'tgt_tokenizer.pkl'))
    
    print(f"Source vocabulary size: {src_tokenizer.vocab_size}")
    print(f"Target vocabulary size: {tgt_tokenizer.vocab_size}")
    
    # Create datasets
    train_dataset = Dataset(train_src, train_tgt, src_tokenizer, tgt_tokenizer, args.max_length)
    val_dataset = Dataset(val_src, val_tgt, src_tokenizer, tgt_tokenizer, args.max_length)
    
    # Create optimized data loaders
    train_loader = create_optimized_dataloader(
        train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = create_optimized_dataloader(
        val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers//2 if args.num_workers else None)
    
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
    )
    
    # Setup for multi-GPU
    model = setup_model_for_cluster(model, device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    check_gpu_memory()
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                          betas=(0.9, 0.98), eps=1e-9)
    
    # Create scheduler
    total_steps = len(train_loader) * args.epochs
    
    if args.use_cosine_schedule:
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer, 
            warmup_steps=args.warmup_steps,
            max_steps=total_steps,
            max_lr=args.learning_rate
        )
    else:
        lr_scheduler = LearningRateScheduler(args.d_model, args.warmup_steps)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler)
    
    # Use standard CrossEntropyLoss with label smoothing
    print(f"Using CrossEntropyLoss with label smoothing: {args.label_smoothing}")
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.word2idx['<pad>'], label_smoothing=args.label_smoothing)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    if scaler:
        print("Mixed precision training enabled")
    
    # Resume training if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming training from {args.resume}")
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler, scaler)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        check_gpu_memory()
        
        # Save model if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.model_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, model_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save model every N epochs
        if (epoch + 1) % args.save_every == 0:
            model_path = os.path.join(args.model_dir, f'model_epoch_{epoch + 1}.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, model_path)
            print(f"Saved model at epoch {epoch + 1}")
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, args.epochs - 1, val_losses[-1], final_model_path)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'args': vars(args),
        'best_val_loss': best_val_loss,
        'total_parameters': num_params
    }
    
    with open(os.path.join(args.log_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    
    # Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Curves - {args.positional_encoding.upper()} Positional Encoding')
    plt.legend()
    plt.grid(True)
    
    # Learning rate schedule
    plt.subplot(2, 2, 2)
    lrs = []
    temp_scheduler = CosineAnnealingWarmupScheduler(
        optim.Adam([torch.randn(1, requires_grad=True)], lr=args.learning_rate),
        args.warmup_steps, total_steps, args.learning_rate
    ) if args.use_cosine_schedule else LearningRateScheduler(args.d_model, args.warmup_steps)
    
    for step in range(min(total_steps, 10000)):  # Plot first 10k steps
        if args.use_cosine_schedule:
            temp_scheduler.step()
            lrs.append(temp_scheduler.optimizer.param_groups[0]['lr'])
        else:
            lrs.append(temp_scheduler(step))
    
    plt.plot(lrs[:len(train_losses)*100:100])  # Sample every 100 steps
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    
    # Training vs Validation Loss
    plt.subplot(2, 2, 3)
    plt.plot(np.array(train_losses) - np.array(val_losses), label='Train - Val Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.title('Overfitting Check')
    plt.legend()
    plt.grid(True)
    
    # Model size info
    plt.subplot(2, 2, 4)
    plt.axis('off')
    info_text = f"""Model Configuration:
    Parameters: {num_params:,}
    d_model: {args.d_model}
    Layers: {args.num_encoder_layers}E + {args.num_decoder_layers}D
    Heads: {args.num_heads}
    Batch Size: {args.batch_size}
    Max Length: {args.max_length}
    Positional Encoding: {args.positional_encoding.upper()}
    Label Smoothing: {args.label_smoothing}
    
    Best Val Loss: {best_val_loss:.4f}
    Total Epochs: {args.epochs}"""
    
    plt.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, f'training_analysis_{args.positional_encoding}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training analysis saved to: {args.log_dir}")

if __name__ == '__main__':
    main()
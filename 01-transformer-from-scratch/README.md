# Transformer from Scratch: Finnish-English Machine Translation

A complete implementation of the Transformer architecture from scratch for machine translation, featuring two positional encoding methods and multiple decoding strategies.

## Overview

This project implements the full Transformer encoder-decoder architecture without using pre-built PyTorch transformer modules. The implementation includes:

- **Positional Encodings**: Rotary Position Embeddings (RoPE) and Relative Position Bias
- **Decoding Strategies**: Greedy, Beam Search, and Top-k Sampling
- **Training Infrastructure**: Multi-GPU support, mixed precision, and advanced scheduling
- **Evaluation Framework**: BLEU scoring and comprehensive analysis tools

## Architecture Details

### Model Specifications
- **Parameters**: ~114 million trainable parameters
- **Dimensions**: d_model=512, d_ff=2048, 8 attention heads
- **Layers**: 6 encoder + 6 decoder layers
- **Vocabulary**: ~67k Finnish tokens, ~34k English tokens
- **Max Sequence Length**: 128 tokens

### Key Components
- Multi-head self-attention and cross-attention mechanisms
- Position-wise feed-forward networks
- Layer normalization and residual connections
- Custom tokenizer with frequency-based vocabulary building

## File Structure

```
├── src/
│   ├── train.py              # Training script with multi-GPU support
│   ├── test.py               # Evaluation script with all decoding strategies
│   ├── encoder.py            # Encoder implementation with positional encodings
│   ├── decoder.py            # Decoder and decoding strategies
│   ├── utils.py              # Tokenizer, dataset, and utility functions
│   └── prepare_dataset.py    # Dataset preparation and preprocessing
├── models/
│   ├── rope_final/           # RoPE model weights and tokenizers
│   └── relative_final/       # Relative position model weights and tokenizers
├── logs/                     # Training logs and analysis plots
├── README.md
├── report.pdf
└── requirements.txt
```

## Usage

### Training

Train RoPE model:
```bash
python src/train.py \
    --data_path finnish_english_100k.txt \
    --positional_encoding rope \
    --batch_size 32 \
    --epochs 10 \
    --learning_rate 1e-4 \
    --model_dir ./models/rope_final \
    --log_dir ./logs/rope_final \
    --label_smoothing 0.1 \
    --warmup_steps 1000 \
    --mixed_precision \
    --use_cosine_schedule
```

Train Relative Position model:
```bash
python src/train.py \
    --data_path finnish_english_100k.txt \
    --positional_encoding relative \
    --batch_size 32 \
    --epochs 10 \
    --learning_rate 1e-4 \
    --model_dir ./models/relative_final \
    --log_dir ./logs/relative_final \
    --label_smoothing 0.1 \
    --warmup_steps 1000 \
    --mixed_precision \
    --use_cosine_schedule
```

### Evaluation

Test RoPE model with all decoding strategies:
```bash
python src/test.py \
    --model_path models/rope_final/best_model.pt \
    --model_dir models/rope_final \
    --data_path finnish_english_100k.txt \
    --positional_encoding rope \
    --decoding_strategy all \
    --detailed_analysis \
    --batch_size 16
```

Test Relative Position model with all decoding strategies:
```bash
python src/test.py \
    --model_path models/relative_final/best_model.pt \
    --model_dir models/relative_final \
    --data_path finnish_english_100k.txt \
    --positional_encoding relative \
    --decoding_strategy all \
    --detailed_analysis \
    --batch_size 16
```

### Key Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--positional_encoding` | rope or relative | rope |
| `--batch_size` | Training batch size | 64 |
| `--learning_rate` | Initial learning rate | 1e-4 |
| `--label_smoothing` | Label smoothing factor | 0.1 |
| `--warmup_steps` | LR warmup steps | 8000 |
| `--mixed_precision` | Enable mixed precision training | False |
| `--use_cosine_schedule` | Use cosine annealing scheduler | False |

## Data Format

The training data should be a tab-separated file with Finnish and English sentence pairs:
```
Finnish sentence 1	English sentence 1
Finnish sentence 2	English sentence 2
...
```

## Results

### Training Performance
| Model | Best Validation Loss | Training Time | Convergence |
|-------|---------------------|---------------|-------------|
| RoPE | 5.4355 | ~2.5 hours | Stable |
| Relative Position | 5.5179 | ~2.5 hours | Stable |

### Key Findings
- Both positional encoding methods achieved similar performance
- RoPE showed slightly better validation loss
- Models successfully learned domain vocabulary and sentence structure
- Translation quality limited by training duration (10 epochs insufficient for full task mastery)
- Clear differences observed between decoding strategies

## Implementation Notes

### Positional Encoding Comparison
- **RoPE**: Parameter-efficient, rotation-based, better theoretical properties
- **Relative Position**: Learnable bias table, simpler implementation, comparable performance

### Decoding Strategies
- **Greedy**: Fastest, deterministic, prone to repetition
- **Beam Search**: Systematic exploration, better quality for short sequences
- **Top-k Sampling**: Most diverse outputs, good for creative applications

### Technical Optimizations
- Multi-GPU training with DataParallel
- Mixed precision training for memory efficiency
- Gradient clipping for training stability
- Advanced learning rate scheduling with warmup

## Requirements

```
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.4.0
tqdm>=4.62.0
```

For CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Hardware Recommendations

- **Training**: 8GB+ GPU memory, multi-GPU setup preferred
- **Inference**: 4GB+ GPU memory sufficient
- **CPU**: Multi-core processor for data loading
- **Storage**: 5GB+ for models and datasets

## Model Files

Pre-trained models available:
- `models/rope_final/best_model.pt` - RoPE model (1.37GB)
- `models/relative_final/best_model.pt` - Relative position model (1.37GB)
- Corresponding tokenizer files (`.pkl`) required for inference

## Known Limitations

- Translation quality requires longer training (20+ epochs recommended)
- Large vocabulary size demands substantial training data
- Memory intensive for longer sequences
- Current implementation optimized for research/educational use

## Citation

This implementation is based on:
- Vaswani et al. "Attention is All You Need" (2017)
- Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- Shaw et al. "Self-Attention with Relative Position Representations" (2018)

## License

Educational use only. Implementation created for Advanced NLP coursework.


Rope google colab weight:-https://drive.google.com/drive/folders/11FLNtDlP8-8DTfvpAsuOWjNinXopW5Ff?usp=drive_link
Relative google colab weight-https://drive.google.com/drive/folders/1QRbK8ZDWpYsUJkl5LJRNTtG2MSf3HbzP?usp=drive_link
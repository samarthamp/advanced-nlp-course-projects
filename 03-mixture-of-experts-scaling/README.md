
# Mixture-of-Experts Transformers for Abstractive Summarization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the implementation and experiments for comparing Mixture-of-Experts (MoE) routing strategies (Top-K vs Hash-based) on abstractive text summarization using the XSum dataset.

**Weights**:-https://drive.google.com/drive/folders/1iR8VFrrx8XkUP5VFtYUwR5cFiE5sJLce?usp=sharing

## 📋 Overview

We implement MoE transformer models from scratch and compare them against pre-trained baselines (BART, T5-Base, Llama-3.2-1B) to investigate:
- Trade-offs between learned Top-K routing and deterministic hash-based routing
- Impact of pre-training on summarization performance
- Effect of load balancing loss on generation quality

**Key Findings:**
- Hash routing achieves perfect load distribution (50% per expert) and 5% higher ROUGE-1 than Top-K
- Pre-trained models outperform from-scratch MoE by 40-100%
- Load balancing loss acts as a critical regularizer preventing degenerate token repetition

## 🏗️ Repository Structure

```
.
├── src/
│   ├── baseline_bart.py               # BART inference script
│   ├── compute_bert_score.py          # BERTScore computation script
│   ├── config.py                      # Configuration settings
│   ├── evaluate.py                    # Comprehensive evaluation metrics
│   ├── extract_human_eval_samples.py  # Extract samples for human evaluation
│   ├── train_full_dataset.py          # Training script for MoE models
│   ├── finetune_baseline_llama.py     # LoRA adaptation for Llama
│   ├── finetune_baseline.py           # LoRA fine-tuning for T5-Base
│   ├── inference_finetuned.py         # Inference for fine-tuned baselines
│   ├── inference.py                   # Inference script for MoE models
│   ├── moe_layer.py                   # MoE layer implementation
│   ├── moe_transformer.py             # MoE Transformer model
│   ├── monitor_training.py            # Training monitoring utilities
│   └── visualize.py                   # Visualization generation
├── outputs/                       # Model checkpoints (gitignored)
├── evaluations/                   # Evaluation metric results
├── results/                       # Prediction JSON files
├── results_no_lb/                 # Prediction JSON files without load balancing
├── visualizations/                # Generated plots and charts
├── commands.md                    # Command snippets for training, evaluation, visualization
├── README.md                      # This README file
├── report.pdf                     # Project report
└── requirements.txt               # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# NVIDIA GPU with 8GB+ VRAM (for training)
nvidia-smi
```

### Installation

```bash
# Clone the repository
git clone https://github.com/ANLP2025/assignment3-Geekonatrip123.git
cd assignment3-Geekonatrip123

# Create virtual environment
python -m venv a3
source a3/bin/activate  # On Windows: a3\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets rouge-score bert-score peft accelerate
pip install nltk matplotlib seaborn tqdm
```

### Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 🎯 Training Models

### MoE Models (From Scratch)

**Top-K Routing (with Load Balancing):**
```bash
python src/train_full_dataset.py \
    --router_type topk \
    --d_model 96 \
    --nhead 4 \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --d_ff 384 \
    --num_experts 4 \
    --top_k 2 \
    --use_load_balancing \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_epochs 3 \
    --output_dir ./outputs
```

**Hash Routing:**
```bash
python src/train_full_dataset.py \
    --router_type hash \
    --d_model 96 \
    --nhead 4 \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --d_ff 384 \
    --num_experts 4 \
    --top_k 2 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_epochs 3 \
    --output_dir ./outputs
```

**Top-K without Load Balancing (Bonus Experiment):**
```bash
python src/train_full_dataset.py \
    --router_type topk \
    --d_model 96 \
    --nhead 4 \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --d_ff 384 \
    --num_experts 4 \
    --top_k 2 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_epochs 3 \
    --output_dir ./outputs_no_lb
```

**Training Time:** ~4.5 hours per model on RTX 4060 8GB

### Baseline Models

**T5-Base (LoRA Fine-tuning):**
```bash
python src/finetune_baseline.py \
    --model_name google-t5/t5-base \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --lora_r 8 \
    --output_dir ./outputs_t5
```

**Llama-3.2-1B (LoRA Adaptation):**
```bash
python src/finetune_baseline_llama.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_epochs 1 \
    --learning_rate 2e-5 \
    --lora_r 8 \
    --output_dir ./outputs_llama
```

**BART (Pre-trained, No Training Required):**
```bash
# Uses facebook/bart-large-xsum directly for inference
```

## 📊 Evaluation

### Run Inference

**MoE Models:**
```bash
python src/inference.py \
    --checkpoint_path ./outputs/best_model_topk.pt \
    --router_type topk \
    --batch_size 8 \
    --output_dir ./results
```

**Fine-tuned Baselines:**
```bash
python src/inference_finetuned.py \
    --checkpoint_path ./outputs_t5 \
    --model_type t5 \
    --batch_size 8 \
    --output_dir ./results
```

**BART:**
```bash
python src/baseline_bart.py \
    --batch_size 8 \
    --output_dir ./results
```

### Compute Metrics

```bash
python src/evaluate.py \
    --predictions_file ./results/predictions_topk.json \
    --model_name moe_topk \
    --compute_rouge \
    --compute_bleu \
    --compute_bertscore \
    --compute_compression \
    --compute_extractiveness \
    --batch_size 32 \
    --output_dir ./evaluations
```

**Metrics Computed:**
- ROUGE-1, ROUGE-2, ROUGE-L
- BLEU-4
- BERTScore (Precision, Recall, F1)
- Compression Ratio
- Extractiveness

### Generate Visualizations

```bash
python src/visualize.py \
    --evaluation_dir ./evaluations \
    --output_dir ./visualizations/final_comparison
```

**Outputs:**
- `model_comparison_metrics.png` - Bar chart comparing all metrics
- `bertscore_comparison.png` - BERTScore across models
- `comprehensive_comparison.png` - 6-panel view with radar chart
- `detailed_comparison_table.png` - Numerical comparison table

## 📈 Results Summary

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore-F1 | Params |
|-------|---------|---------|---------|------|--------------|--------|
| **BART** | **0.4514** | **0.2200** | **0.3692** | **0.1365** | **0.9159** | 406M |
| **T5-Base** | 0.3245 | 0.1048 | 0.2516 | 0.0536 | 0.8918 | 223M |
| Llama-1B | 0.2359 | 0.0666 | 0.1712 | 0.0277 | 0.8644 | 1.2B |
| MoE-Hash | 0.2230 | 0.0487 | 0.1798 | 0.0267 | 0.8515 | 16M |
| MoE-TopK | 0.2121 | 0.0459 | 0.1700 | 0.0247 | 0.8375 | 16M |

### Expert Usage

**Hash Routing:** Perfect 50% distribution across all experts
**Top-K Routing:** 44-57% range, showing learned specialization

### Load Balancing Impact

Removing load balancing loss from Top-K routing:
- ✅ Improved validation loss: 4.34 → 4.33
- ❌ Degraded ROUGE-1: 0.2121 → 0.2043 (-3.7%)
- ❌ Caused repetitive token generation ("to to to...")

**Conclusion:** Load balancing loss is essential as a regularizer.

## 🔧 Hardware Requirements

### Minimum Requirements
- **GPU:** NVIDIA GPU with 8GB VRAM (e.g., RTX 4060, RTX 3070)
- **RAM:** 16GB system RAM
- **Storage:** 10GB for datasets and checkpoints
- **CUDA:** 11.8 or higher

### Training Time Estimates
- **MoE models:** ~4-5 hours per model
- **T5-Base (LoRA):** ~8 hours
- **Llama-1B (LoRA):** ~24 hours
- **Inference (all models):** ~20-30 minutes per model on test set

## 📝 Key Arguments

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--router_type` | Routing mechanism: `topk` or `hash` | `topk` |
| `--d_model` | Model dimension | 96 |
| `--num_experts` | Number of experts per layer | 4 |
| `--top_k` | Number of experts to activate | 2 |
| `--use_load_balancing` | Enable load balancing loss | False |
| `--batch_size` | Batch size per GPU | 2 |
| `--gradient_accumulation_steps` | Gradient accumulation | 8 |
| `--num_epochs` | Training epochs | 3 |
| `--learning_rate` | Learning rate | 2e-4 |

### Evaluation Arguments

| Argument | Description |
|----------|-------------|
| `--compute_rouge` | Compute ROUGE scores |
| `--compute_bleu` | Compute BLEU score |
| `--compute_bertscore` | Compute BERTScore |
| `--compute_compression` | Compute compression ratio |
| `--compute_extractiveness` | Compute extractiveness |

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch_size 1 --gradient_accumulation_steps 16

# Or use CPU (slower)
--device cpu
```

### Dependency Conflicts
```bash
# Use exact versions
pip install transformers==4.36.0 datasets==2.15.0 torch==2.1.0
```

### Dataset Download Issues
```bash
# Manually download XSum
from datasets import load_dataset
dataset = load_dataset("EdinburghNLP/xsum", cache_dir="./data")
```

## 📚 References

- **XSum Dataset:** [Narayan et al., 2018](https://arxiv.org/abs/1808.08745)
- **MoE Transformers:** [Shazeer et al., 2017](https://arxiv.org/abs/1701.06538)
- **BART:** [Lewis et al., 2019](https://arxiv.org/abs/1910.13461)
- **T5:** [Raffel et al., 2020](https://arxiv.org/abs/1910.10683)
- **LoRA:** [Hu et al., 2021](https://arxiv.org/abs/2106.09685)

## 🙏 Acknowledgments

- Course instructors and TAs for guidance
- Hugging Face for the Transformers library
- The XSum dataset authors
- IIIT Hyderabad for computational resources

---

**Note:** This project was completed as part of the Advanced NLP coursework at IIIT Hyderabad.
# Complete Command History - MoE Summarization Assignment

## Environment Setup
```powershell
# Create virtual environment
python -m venv a3

# Activate virtual environment
.\a3\Scripts\activate

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets rouge-score nltk bert-score evaluate accelerate peft
pip install matplotlib seaborn pandas tqdm
pip install python-docx openpyxl

# Downgrade datasets library (to fix compatibility issue)
pip install datasets==3.6.0
```

## 1. Training MoE Models (From Scratch)

### MoE with Top-K Routing
```powershell
python train_full_dataset.py `
    --router_type topk `
    --d_model 96 `
    --nhead 4 `
    --num_encoder_layers 2 `
    --num_decoder_layers 2 `
    --d_ff 384 `
    --num_experts 4 `
    --top_k 2 `
    --use_load_balancing `
    --batch_size 2 `
    --gradient_accumulation_steps 8 `
    --num_epochs 3 `
    --max_src_len 256 `
    --max_tgt_len 48 `
    --output_dir ./outputs
```
**Time:** ~4.5 hours  
**Result:** Best Val Loss = 4.3395

### MoE with Hash Routing
```powershell
python train_full_dataset.py `
    --router_type hash `
    --d_model 96 `
    --nhead 4 `
    --num_encoder_layers 2 `
    --num_decoder_layers 2 `
    --d_ff 384 `
    --num_experts 4 `
    --top_k 2 `
    --batch_size 2 `
    --gradient_accumulation_steps 8 `
    --num_epochs 3 `
    --output_dir ./outputs
```
**Time:** ~4.5 hours  
**Result:** Best Val Loss = 4.3868

## 2. MoE Model Inference

### Top-K Inference
```powershell
# Fix inference.py for correct max_src_len (changed 512 to 256)
(Get-Content inference.py) -replace "max_length=512,", "max_length=256," | Set-Content inference.py

python inference.py `
    --checkpoint_path ./outputs/best_model_topk.pt `
    --router_type topk `
    --batch_size 8 `
    --output_dir ./results
```
**Time:** ~20 minutes  
**Output:** `./results/predictions_topk.json`

### Hash Inference
```powershell
python inference.py `
    --checkpoint_path ./outputs/best_model_hash.pt `
    --router_type hash `
    --batch_size 8 `
    --output_dir ./results
```
**Time:** ~10 minutes  
**Output:** `./results/predictions_hash.json`

## 3. BART Baseline (Zero-shot)

```powershell
python baseline_bart.py `
    --output_dir ./results `
    --batch_size 8
```
**Time:** ~52 minutes  
**Output:** `./results/predictions_bart.json`

## 4. Fine-tuning T5-Base with LoRA

### Fix Training Script
```powershell
# Fix deprecated parameter name
(Get-Content finetune_baseline.py) -replace "evaluation_strategy='epoch'", "eval_strategy='epoch'" | Set-Content finetune_baseline.py
```

### Train T5-Base (Attempt 1 - Failed with NaN)
```powershell
# This failed due to FP16 instability - NaN loss at epoch 1.83
python finetune_baseline.py `
    --model_name google-t5/t5-base `
    --model_type encoder-decoder `
    --batch_size 2 `
    --gradient_accumulation_steps 8 `
    --num_epochs 3 `
    --use_lora `
    --lora_r 8 `
    --max_src_len 256 `
    --max_tgt_len 48 `
    --use_fp16 `
    --output_dir ./finetuned_models
```
**Result:** ❌ Failed with NaN loss

### Clean Up Failed Model
```powershell
Remove-Item -Recurse -Force ./finetuned_models/google-t5-t5-base
```

### Train T5-Base (Attempt 2 - Success)
```powershell
# Removed --use_fp16 flag for stability (defaults to FP32)
python finetune_baseline.py `
    --model_name google-t5/t5-base `
    --model_type encoder-decoder `
    --batch_size 2 `
    --gradient_accumulation_steps 8 `
    --num_epochs 3 `
    --use_lora `
    --lora_r 8 `
    --max_src_len 256 `
    --max_tgt_len 48 `
    --output_dir ./finetuned_models
```
**Time:** ~8 hours 16 minutes  
**Result:** ✅ Best Val Loss = 1.3453  
**Trainable params:** 884,736 / 223,788,288 (0.4%)

## 5. T5-Base Inference

```powershell
python inference_finetuned.py `
    --model_path ./finetuned_models/google-t5-t5-base/final_model `
    --model_type seq2seq `
    --model_name t5_base `
    --batch_size 8 `
    --output_dir ./results
```
**Time:** ~35 minutes  
**Output:** `./results/predictions_t5_base.json`

## 6. Fine-tuning Llama-3.2-1B-Instruct

### Set Hugging Face Token (for gated model access)
```powershell
# Option 1: Permanent login (recommended)
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
huggingface-cli login --token "$HF_TOKEN"

```

### Train Llama (Optimized Settings)
```powershell
# Used 1 epoch (domain adaptation), max_len 256, no FP16
python finetune_baseline.py `
    --model_name meta-llama/Llama-3.2-1B-Instruct `
    --model_type causal `
    --batch_size 1 `
    --gradient_accumulation_steps 16 `
    --num_epochs 1 `
    --use_lora `
    --lora_r 8 `
    --max_len 256 `
    --output_dir ./finetuned_models
```
**Time:** ~24-30 hours (estimated)  
**Trainable params:** 851,968 / 1,236,666,368 (0.07%)

## 7. Llama Inference

```powershell
python inference_finetuned.py `
    --model_path ./finetuned_models/meta-llama-Llama-3.2-1B-Instruct/final_model `
    --model_type causal `
    --model_name llama_1b `
    --batch_size 4 `
    --output_dir ./results
```
**Time:** ~2-3 hours (estimated)  
**Output:** `./results/predictions_llama_1b.json`

## 8. Evaluation

### Evaluate MoE Top-K
```powershell
python evaluate.py `
    --predictions_file ./results/predictions_topk.json `
    --model_name moe_topk `
    --compute_rouge `
    --compute_bleu `
    --compute_compression `
    --compute_extractiveness `
    --generate_human_eval `
    --num_human_eval_samples 3 `
    --output_dir ./evaluations
```
**Output:** `./evaluations/evaluation_moe_topk.json`

### Evaluate MoE Hash
```powershell
python evaluate.py `
    --predictions_file ./results/predictions_hash.json `
    --model_name moe_hash `
    --compute_rouge `
    --compute_bleu `
    --compute_compression `
    --compute_extractiveness `
    --output_dir ./evaluations
```
**Output:** `./evaluations/evaluation_moe_hash.json`

### Evaluate BART
```powershell
python evaluate.py `
    --predictions_file ./results/predictions_bart.json `
    --model_name bart `
    --compute_rouge `
    --compute_bleu `
    --compute_compression `
    --compute_extractiveness `
    --output_dir ./evaluations
```
**Output:** `./evaluations/evaluation_bart.json`

### Evaluate T5-Base
```powershell
python evaluate.py `
    --predictions_file ./results/predictions_t5_base.json `
    --model_name t5_base `
    --compute_rouge `
    --compute_bleu `
    --compute_compression `
    --compute_extractiveness `
    --output_dir ./evaluations
```
**Output:** `./evaluations/evaluation_t5_base.json`

### Evaluate Llama
```powershell
python evaluate.py `
    --predictions_file ./results/predictions_llama_1b.json `
    --model_name llama_1b `
    --compute_rouge `
    --compute_bleu `
    --compute_compression `
    --compute_extractiveness `
    --output_dir ./evaluations
```
**Output:** `./evaluations/evaluation_llama_1b.json`

## 9. Visualization

### Create Directories
```powershell
New-Item -ItemType Directory -Force -Path ./visualizations/topk
New-Item -ItemType Directory -Force -Path ./visualizations/hash
New-Item -ItemType Directory -Force -Path ./visualizations/comparison
```

### Visualize MoE Top-K
```powershell
# Expert usage
python visualize.py `
    --predictions_file ./results/predictions_topk.json `
    --output_dir ./visualizations/topk

# Training history
python visualize.py `
    --training_history ./outputs/training_history_topk.json `
    --output_dir ./visualizations/topk

# All metrics
python visualize.py `
    --evaluation_file ./evaluations/evaluation_moe_topk.json `
    --model_name "MoE-TopK" `
    --output_dir ./visualizations/topk
```

### Visualize MoE Hash
```powershell
# Expert usage
python visualize.py `
    --predictions_file ./results/predictions_hash.json `
    --output_dir ./visualizations/hash

# Training history
python visualize.py `
    --training_history ./outputs/training_history_hash.json `
    --output_dir ./visualizations/hash

# All metrics
python visualize.py `
    --evaluation_file ./evaluations/evaluation_moe_hash.json `
    --model_name "MoE-Hash" `
    --output_dir ./visualizations/hash
```

### Compare All Models
```powershell
python visualize.py `
    --compare_models `
    --evaluation_files "./evaluations/evaluation_bart.json,./evaluations/evaluation_t5_base.json,./evaluations/evaluation_llama_1b.json,./evaluations/evaluation_moe_topk.json,./evaluations/evaluation_moe_hash.json" `
    --model_names "BART,T5-Base,Llama-1B,MoE-TopK,MoE-Hash" `
    --output_dir ./visualizations/comparison
```

## 10. Results Summary

### Final Performance (ROUGE-1 Scores)
| Model | ROUGE-1 | Parameters | Training |
|-------|---------|------------|----------|
| BART | 0.4514 | 406M | Pre-trained (zero-shot) |
| T5-Base | 0.3245 | 223M | Pre-trained + LoRA |
| Llama-1B | TBD | 1.2B | Pre-trained + LoRA |
| MoE-Hash | 0.2230 | 16M | From scratch |
| MoE-TopK | 0.2121 | 16M | From scratch |

### Key Findings
1. **Expert Routing Comparison:**
   - Top-K: Learned specialization (44-57% usage per expert)
   - Hash: Perfect load balance (50% usage per expert)

2. **Pre-training Impact:**
   - Pre-trained models: 2x better performance
   - Shows critical importance of pre-training

3. **Training Time:**
   - MoE models: ~4.5 hours each
   - T5-Base: ~8 hours
   - Llama-1B: ~24-30 hours

## Notes
- All MoE models trained from scratch (including embeddings)
- T5 and Llama used LoRA (Parameter-Efficient Fine-Tuning)
- BART used as zero-shot baseline (already fine-tuned on XSum)
- Dataset: XSum (204,045 train, 11,332 validation, 11,334 test)
- Hardware: NVIDIA RTX 4060 Laptop GPU (8GB VRAM)
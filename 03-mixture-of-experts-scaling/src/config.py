"""
Optimized configurations for RTX 4060 8GB VRAM
"""

# Small MoE Configuration (Recommended for 8GB VRAM)
SMALL_MOE_CONFIG = {
    'd_model': 128,
    'nhead': 4,
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'd_ff': 512,
    'num_experts': 4,
    'top_k': 2,
    'dropout': 0.1,
    'batch_size': 4,
    'gradient_accumulation_steps': 4,  # Effective batch size = 16
    'max_src_len': 256,
    'max_tgt_len': 48,
    'learning_rate': 5e-4,
}

# Tiny MoE Configuration (For even tighter memory)
TINY_MOE_CONFIG = {
    'd_model': 96,
    'nhead': 4,
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'd_ff': 384,
    'num_experts': 4,
    'top_k': 2,
    'dropout': 0.1,
    'batch_size': 8,
    'gradient_accumulation_steps': 2,
    'max_src_len': 256,
    'max_tgt_len': 48,
    'learning_rate': 5e-4,
}

# Baseline fine-tuning configurations
BASELINE_CONFIGS = {
    't5-base': {
        'batch_size': 2,
        'gradient_accumulation_steps': 8,
        'max_src_len': 256,
        'max_tgt_len': 48,
        'use_lora': True,
        'lora_r': 8,
    },
    'llama-1b': {
        'batch_size': 1,
        'gradient_accumulation_steps': 16,
        'max_len': 512,
        'use_lora': True,
        'lora_r': 8,
        'use_fp16': True,
    }
}
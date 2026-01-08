import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import argparse
import os
import numpy as np


def preprocess_function_seq2seq(examples, tokenizer, max_src_len, max_tgt_len):
    """Preprocess for encoder-decoder models"""
    inputs = examples['document']
    targets = examples['summary']
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_src_len,
        truncation=True,
        padding='max_length'
    )
    
    labels = tokenizer(
        targets,
        max_length=max_tgt_len,
        truncation=True,
        padding='max_length'
    )
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def preprocess_function_causal(examples, tokenizer, max_len):
    """Preprocess for instruction/causal LM models"""
    inputs = []
    
    for doc, summary in zip(examples['document'], examples['summary']):
        # Format as instruction
        prompt = f"Summarize the following article in one sentence:\n\n{doc}\n\nSummary: {summary}"
        inputs.append(prompt)
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_len,
        truncation=True,
        padding='max_length'
    )
    
    # For causal LM, labels are same as input_ids
    model_inputs['labels'] = model_inputs['input_ids'].copy()
    
    return model_inputs


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    
    if args.model_type == 'encoder-decoder':
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        
        # LoRA configuration for seq2seq
        if args.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q", "v"]  # Adjust based on model architecture
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
    
    elif args.model_type == 'causal':
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if args.use_fp16 else torch.float32,
            device_map='auto' if args.use_device_map else None
        )
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
        
        # LoRA configuration for causal LM
        if args.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "v_proj"]  # Adjust based on model
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset('xsum', trust_remote_code=True)
    
    # Use subset if specified
    if args.use_subset:
        train_data = dataset['train'].select(range(min(args.subset_size, len(dataset['train']))))
        val_data = dataset['validation'].select(range(min(args.subset_size // 10, len(dataset['validation']))))
    else:
        train_data = dataset['train']
        val_data = dataset['validation']
    
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    if args.model_type == 'encoder-decoder':
        train_dataset = train_data.map(
            lambda x: preprocess_function_seq2seq(x, tokenizer, args.max_src_len, args.max_tgt_len),
            batched=True,
            remove_columns=train_data.column_names
        )
        val_dataset = val_data.map(
            lambda x: preprocess_function_seq2seq(x, tokenizer, args.max_src_len, args.max_tgt_len),
            batched=True,
            remove_columns=val_data.column_names
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        
    else:  # causal
        train_dataset = train_data.map(
            lambda x: preprocess_function_causal(x, tokenizer, args.max_len),
            batched=True,
            remove_columns=train_data.column_names
        )
        val_dataset = val_data.map(
            lambda x: preprocess_function_causal(x, tokenizer, args.max_len),
            batched=True,
            remove_columns=val_data.column_names
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Training arguments
    output_dir = os.path.join(args.output_dir, args.model_name.replace('/', '-'))
    
    if args.model_type == 'encoder-decoder':
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_epochs,
            weight_decay=args.weight_decay,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='loss',
            predict_with_generate=True,
            fp16=args.use_fp16,
            logging_steps=100,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
    
    else:  # causal
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_epochs,
            weight_decay=args.weight_decay,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='loss',
            fp16=args.use_fp16,
            logging_steps=100,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model')
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"\nTraining completed! Model saved to {final_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune Baseline Models')
    
    parser.add_argument('--model_name', type=str, required=True, help='Hugging Face model name')
    parser.add_argument('--model_type', type=str, required=True, choices=['encoder-decoder', 'causal'],
                       help='Model type')
    parser.add_argument('--output_dir', type=str, default='./finetuned_models', help='Output directory')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_src_len', type=int, default=512, help='Max source length (encoder-decoder)')
    parser.add_argument('--max_tgt_len', type=int, default=64, help='Max target length (encoder-decoder)')
    parser.add_argument('--max_len', type=int, default=768, help='Max length (causal LM)')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation')
    
    # LoRA parameters
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    
    # Other
    parser.add_argument('--use_subset', action='store_true', help='Use subset of data')
    parser.add_argument('--subset_size', type=int, default=10000, help='Subset size')
    parser.add_argument('--use_fp16', action='store_true', help='Use FP16 training')
    parser.add_argument('--use_device_map', action='store_true', help='Use device map for large models')
    
    args = parser.parse_args()
    
    main(args)
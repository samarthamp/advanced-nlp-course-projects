"""
Task 2.3: Library-Based Quantization with bitsandbytes
Use bitsandbytes for 8-bit and 4-bit NF4 quantization
"""

import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

class BitsAndBytesQuantizer:
    """Quantization using bitsandbytes library"""
    
    def __init__(self, model_path='models/baseline_gpt2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model_path = model_path
    
    def load_8bit_model(self):
        """Load model with 8-bit quantization"""
        print("Loading model with 8-bit quantization...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        
        model = GPT2ForSequenceClassification.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map='auto' if torch.cuda.is_available() else None
        )
        
        print("8-bit model loaded successfully")
        return model
    
    def load_4bit_model(self):
        """Load model with 4-bit NF4 quantization"""
        print("Loading model with 4-bit NF4 quantization...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = GPT2ForSequenceClassification.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map='auto' if torch.cuda.is_available() else None
        )
        
        print("4-bit NF4 model loaded successfully")
        return model
    
    def get_model_size(self, model):
        """Calculate model size in MB"""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        return total_size / (1024**2)
    
    def evaluate(self, model, test_loader):
        """Evaluate the quantized model"""
        all_preds = []
        all_labels = []
        inference_times = []
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['label'].to(model.device)
                
                if torch.cuda.is_available():
                     torch.cuda.synchronize()
                start_time = time.time()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_times.append(time.time() - start_time)
                
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'model_size_mb': self.get_model_size(model),
            'avg_inference_time_ms': np.mean(inference_times) * 1000
        }
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(cm, title, save_path):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")


def prepare_test_data(tokenizer, batch_size=16, max_length=128):
    """Prepare test dataset"""
    print("Loading AG News test dataset...")
    dataset = load_dataset('ag_news')
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    test_dataset = dataset['test'].map(tokenize_function, batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return test_loader


if __name__ == '__main__':
    print("="*70)
    print("TASK 2.3: BITSANDBYTES QUANTIZATION")
    print("="*70)
    
    quantizer = BitsAndBytesQuantizer('models/baseline_gpt2')
    test_loader = prepare_test_data(quantizer.tokenizer)
    
    # 8-bit Quantization
    print("\n" + "-"*70)
    print("8-BIT QUANTIZATION")
    print("-"*70)
    
    model_8bit = quantizer.load_8bit_model()
    metrics_8bit = quantizer.evaluate(model_8bit, test_loader)
    
    print("\nINT8 QUANTIZED MODEL RESULTS (BITSANDBYTES)")
    print("="*70)
    print(f"Accuracy: {metrics_8bit['accuracy']:.4f}")
    print(f"Precision: {metrics_8bit['precision']:.4f}")
    print(f"Recall: {metrics_8bit['recall']:.4f}")
    print(f"F1-Score: {metrics_8bit['f1_score']:.4f}")
    print(f"Model Size: {metrics_8bit['model_size_mb']:.2f} MB")
    print(f"Avg Inference Time: {metrics_8bit['avg_inference_time_ms']:.2f} ms")
    
    quantizer.plot_confusion_matrix(
        metrics_8bit['confusion_matrix'],
        'INT8 (bitsandbytes)',
        'int8_bitsandbytes_confusion_matrix.png'
    )
    
    # Clear memory
    del model_8bit
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 4-bit NF4 Quantization
    print("\n" + "-"*70)
    print("4-BIT NF4 QUANTIZATION")
    print("-"*70)
    
    model_4bit = quantizer.load_4bit_model()
    metrics_4bit = quantizer.evaluate(model_4bit, test_loader)
    
    print("\nNF4 QUANTIZED MODEL RESULTS (BITSANDBYTES)")
    print("="*70)
    print(f"Accuracy: {metrics_4bit['accuracy']:.4f}")
    print(f"Precision: {metrics_4bit['precision']:.4f}")
    print(f"Recall: {metrics_4bit['recall']:.4f}")
    print(f"F1-Score: {metrics_4bit['f1_score']:.4f}")
    print(f"Model Size: {metrics_4bit['model_size_mb']:.2f} MB")
    print(f"Avg Inference Time: {metrics_4bit['avg_inference_time_ms']:.2f} ms")
    
    quantizer.plot_confusion_matrix(
        metrics_4bit['confusion_matrix'],
        'NF4 (bitsandbytes)',
        'nf4_bitsandbytes_confusion_matrix.png'
    )
    
    # Comparison Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Metric':<25} {'INT8':<15} {'NF4':<15}")
    print("-" * 55)
    print(f"{'Accuracy':<25} {metrics_8bit['accuracy']:<15.4f} {metrics_4bit['accuracy']:<15.4f}")
    print(f"{'F1-Score':<25} {metrics_8bit['f1_score']:<15.4f} {metrics_4bit['f1_score']:<15.4f}")
    print(f"{'Model Size (MB)':<25} {metrics_8bit['model_size_mb']:<15.2f} {metrics_4bit['model_size_mb']:<15.2f}")
    print(f"{'Inference Time (ms)':<25} {metrics_8bit['avg_inference_time_ms']:<15.2f} {metrics_4bit['avg_inference_time_ms']:<15.2f}")
    
    print("\n✓ Task 2.3 completed successfully!")
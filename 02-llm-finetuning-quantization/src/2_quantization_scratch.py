"""
Task 2.2: Post-Training Quantization from Scratch
Implement INT8 quantization without using bitsandbytes library
"""

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

class LinearQuantizer:
    """Implements linear quantization from scratch"""
    
    @staticmethod
    def quantize_tensor(tensor, num_bits=8):
        """Quantize a FP32 tensor to INT8"""
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1
        
        scale = (max_val - min_val) / (qmax - qmin)
        if scale == 0:
            scale = 1.0
        
        zero_point = qmin - min_val / scale
        zero_point = int(np.round(zero_point))
        zero_point = np.clip(zero_point, qmin, qmax)
        
        quantized_tensor = torch.round(tensor / scale + zero_point)
        quantized_tensor = torch.clamp(quantized_tensor, qmin, qmax).to(torch.int8)
        
        return quantized_tensor, scale, zero_point
    
    @staticmethod
    def dequantize_tensor(quantized_tensor, scale, zero_point):
        """Dequantize an INT8 tensor back to FP32"""
        dequantized_tensor = (quantized_tensor.float() - zero_point) * scale
        return dequantized_tensor


class QuantizedGPT2Model:
    """Wrapper for quantized GPT-2 model"""
    
    def __init__(self, model_path='models/baseline_gpt2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        print("Loading baseline model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2ForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.quantized_params = {}
        self.quantizer = LinearQuantizer()
    
    def quantize_model(self):
        """Quantize all model parameters to INT8"""
        print("Quantizing model to INT8...")
        
        for name, param in tqdm(self.model.named_parameters(), desc="Quantizing"):
            if param.requires_grad:
                quantized, scale, zero_point = self.quantizer.quantize_tensor(param.data)
                self.quantized_params[name] = {
                    'quantized': quantized,
                    'scale': scale,
                    'zero_point': zero_point,
                    'shape': param.shape
                }
        
        print(f"Quantized {len(self.quantized_params)} parameters")
    
    def dequantize_and_load(self):
        """Dequantize parameters and load them into the model"""
        for name, param in self.model.named_parameters():
            if name in self.quantized_params:
                quant_data = self.quantized_params[name]
                dequantized = self.quantizer.dequantize_tensor(
                    quant_data['quantized'],
                    quant_data['scale'],
                    quant_data['zero_point']
                )
                param.data = dequantized.view(quant_data['shape']).to(self.device)
    
    def get_model_size(self):
        """Calculate model size in MB"""
        quantized_size = sum(
            param['quantized'].numel() * param['quantized'].element_size() + 16
            for param in self.quantized_params.values()
        )
        return quantized_size / (1024**2)
    
    def evaluate(self, test_loader):
        """Evaluate the quantized model"""
        print("Dequantizing and loading parameters...")
        self.dequantize_and_load()
        
        all_preds = []
        all_labels = []
        inference_times = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Same fix:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
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
            'model_size_mb': self.get_model_size(),
            'avg_inference_time_ms': np.mean(inference_times) * 1000
        }
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(cm, save_path='int8_scratch_confusion_matrix.png'):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - INT8 Quantized (Scratch)')
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
    print("TASK 2.2: POST-TRAINING QUANTIZATION FROM SCRATCH")
    print("="*70)
    
    quant_model = QuantizedGPT2Model('models/baseline_gpt2')
    quant_model.quantize_model()
    
    test_loader = prepare_test_data(quant_model.tokenizer)
    
    print("\nEvaluating quantized model...")
    metrics = quant_model.evaluate(test_loader)
    
    print("\n" + "="*70)
    print("INT8 QUANTIZED MODEL RESULTS (FROM SCRATCH)")
    print("="*70)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Model Size: {metrics['model_size_mb']:.2f} MB")
    print(f"Avg Inference Time: {metrics['avg_inference_time_ms']:.2f} ms")
    
    quant_model.plot_confusion_matrix(metrics['confusion_matrix'])
    
    print("\n✓ Task 2.2 completed successfully!")
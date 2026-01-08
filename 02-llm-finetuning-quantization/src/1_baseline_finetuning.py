"""
Task 2.1: Baseline Fine-tuning
Fine-tune GPT-2 Small on AG News dataset for text classification
"""

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

class GPT2Trainer:
    def __init__(self, model_name='gpt2', num_labels=4, max_length=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model.to(self.device)
        self.max_length = max_length
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Full fine-tuning: {trainable_params == total_params}")
        
    def prepare_data(self, batch_size=16):
        """Load and prepare AG News dataset"""
        print("Loading AG News dataset...")
        dataset = load_dataset('ag_news')
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
        
        print("Tokenizing datasets...")
        train_dataset = dataset['train'].map(tokenize_function, batched=True)
        test_dataset = dataset['test'].map(tokenize_function, batched=True)
        
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        return train_loader, test_loader
    
    def train(self, train_loader, epochs=3, learning_rate=2e-5):
        """Train the model with full fine-tuning"""
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
    
    def evaluate(self, test_loader):
        """Evaluate the model and return metrics"""
        print("Evaluating model...")
        self.model.eval()
        all_preds = []
        all_labels = []
        inference_times = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
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
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        cm = confusion_matrix(all_labels, all_preds)
        
        # Model size
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**2)
        
        # Average inference time
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'model_size_mb': model_size,
            'avg_inference_time_ms': avg_inference_time
        }
        
        return metrics
    
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png'):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Baseline FP32')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    
    def save_model(self, path='models/baseline_gpt2'):
        """Save the fine-tuned model"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f'Model saved to {path}')


if __name__ == '__main__':
    print("="*70)
    print("TASK 2.1: BASELINE FINE-TUNING")
    print("="*70)
    
    # Initialize trainer
    trainer = GPT2Trainer(model_name='gpt2', num_labels=4)
    
    # Prepare data
    train_loader, test_loader = trainer.prepare_data(batch_size=16)
    
    # Train model
    print("\nStarting training...")
    trainer.train(train_loader, epochs=3)
    
    # Evaluate model
    print("\nEvaluating baseline model...")
    metrics = trainer.evaluate(test_loader)
    
    # Print results
    print("\n" + "="*70)
    print("BASELINE MODEL RESULTS")
    print("="*70)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Model Size: {metrics['model_size_mb']:.2f} MB")
    print(f"Avg Inference Time: {metrics['avg_inference_time_ms']:.2f} ms")
    
    # Plot confusion matrix
    trainer.plot_confusion_matrix(metrics['confusion_matrix'], 'baseline_confusion_matrix.png')
    
    # Save model
    trainer.save_model('models/baseline_gpt2')
    
    print("\nTask 2.1 completed successfully!")
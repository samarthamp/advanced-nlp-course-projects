"""
Complete comparison of all quantization approaches
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

class ModelComparator:
    def __init__(self):
        self.results = {}
    
    def add_result(self, model_name, metrics):
        self.results[model_name] = metrics
    
    def generate_comparison_table(self):
        data = []
        for model_name, metrics in self.results.items():
            data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'Size (MB)': f"{metrics['model_size_mb']:.2f}",
                'Inference (ms)': f"{metrics['avg_inference_time_ms']:.2f}"
            })
        return pd.DataFrame(data)
    
    def plot_performance_comparison(self, save_path='performance_comparison.png'):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        models = list(self.results.keys())
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
            ax = axes[idx // 2, idx % 2]
            values = [self.results[model][metric] for model in models]
            bars = ax.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_ylabel(title)
            ax.set_title(f'{title} Comparison')
            ax.tick_params(axis='x', rotation=45)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved to {save_path}")
    
    def plot_efficiency_comparison(self, save_path='efficiency_comparison.png'):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        models = list(self.results.keys())
        
        sizes = [self.results[model]['model_size_mb'] for model in models]
        axes[0].bar(models, sizes, color='steelblue')
        axes[0].set_ylabel('Model Size (MB)')
        axes[0].set_title('Model Size Comparison')
        axes[0].tick_params(axis='x', rotation=45)
        
        times = [self.results[model]['avg_inference_time_ms'] for model in models]
        axes[1].bar(models, times, color='coral')
        axes[1].set_ylabel('Avg Inference Time (ms)')
        axes[1].set_title('Inference Time Comparison')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved to {save_path}")
    
    def calculate_compression_ratio(self, baseline='Baseline FP32'):
        if baseline not in self.results:
            return
        baseline_size = self.results[baseline]['model_size_mb']
        print("\n" + "="*70)
        print("COMPRESSION ANALYSIS")
        print("="*70)
        print(f"\nBaseline Size: {baseline_size:.2f} MB\n")
        for model, metrics in self.results.items():
            if model != baseline:
                size = metrics['model_size_mb']
                ratio = baseline_size / size
                reduction = (1 - size/baseline_size) * 100
                print(f"{model}:")
                print(f"  Size: {size:.2f} MB")
                print(f"  Compression: {ratio:.2f}x")
                print(f"  Size Reduction: {reduction:.2f}%\n")
    
    def generate_full_report(self, output_dir='results'):
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "="*70)
        print("GENERATING REPORT")
        print("="*70)
        
        df = self.generate_comparison_table()
        print("\n" + str(df))
        df.to_csv(f'{output_dir}/comparison_table.csv', index=False)
        
        self.plot_performance_comparison(f'{output_dir}/performance_comparison.png')
        self.plot_efficiency_comparison(f'{output_dir}/efficiency_comparison.png')
        self.calculate_compression_ratio()
        
        print(f"\nAll results saved to '{output_dir}/'")

if __name__ == '__main__':
    print("This will be called by run_all_experiments.py")
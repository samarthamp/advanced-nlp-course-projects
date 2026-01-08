import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from matplotlib.patches import Rectangle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_training_history(history_file, output_dir):
    """Plot training and validation loss"""
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training history plot to {output_path}")


def plot_expert_usage(predictions_file, output_dir):
    """Plot expert usage statistics"""
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    if 'expert_usage' not in data:
        print("No expert usage data found in predictions file")
        return
    
    expert_usage = data['expert_usage']
    
    # Plot encoder expert usage
    if 'encoder' in expert_usage:
        fig, axes = plt.subplots(len(expert_usage['encoder']), 1, figsize=(10, 3*len(expert_usage['encoder'])))
        if len(expert_usage['encoder']) == 1:
            axes = [axes]
        
        for i, usage in enumerate(expert_usage['encoder']):
            usage_array = np.array(usage)
            expert_indices = np.arange(len(usage_array))
            
            axes[i].bar(expert_indices, usage_array, color='skyblue', edgecolor='navy', alpha=0.7)
            axes[i].set_xlabel('Expert Index', fontsize=11)
            axes[i].set_ylabel('Usage Frequency', fontsize=11)
            axes[i].set_title(f'Encoder Layer {i+1} - Expert Usage', fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for idx, val in enumerate(usage_array):
                axes[i].text(idx, val, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'encoder_expert_usage.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved encoder expert usage plot to {output_path}")
    
    # Plot decoder expert usage
    if 'decoder' in expert_usage:
        fig, axes = plt.subplots(len(expert_usage['decoder']), 1, figsize=(10, 3*len(expert_usage['decoder'])))
        if len(expert_usage['decoder']) == 1:
            axes = [axes]
        
        for i, usage in enumerate(expert_usage['decoder']):
            usage_array = np.array(usage)
            expert_indices = np.arange(len(usage_array))
            
            axes[i].bar(expert_indices, usage_array, color='lightcoral', edgecolor='darkred', alpha=0.7)
            axes[i].set_xlabel('Expert Index', fontsize=11)
            axes[i].set_ylabel('Usage Frequency', fontsize=11)
            axes[i].set_title(f'Decoder Layer {i+1} - Expert Usage', fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for idx, val in enumerate(usage_array):
                axes[i].text(idx, val, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'decoder_expert_usage.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved decoder expert usage plot to {output_path}")


def plot_expert_usage_heatmap(predictions_file, output_dir):
    """Plot expert usage as heatmap"""
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    if 'expert_usage' not in data:
        print("No expert usage data found in predictions file")
        return
    
    expert_usage = data['expert_usage']
    
    # Create heatmap for encoder
    if 'encoder' in expert_usage:
        encoder_usage = np.array(expert_usage['encoder'])
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(encoder_usage, annot=True, fmt='.3f', cmap='YlOrRd', 
                    xticklabels=[f'E{i}' for i in range(encoder_usage.shape[1])],
                    yticklabels=[f'Layer {i+1}' for i in range(encoder_usage.shape[0])],
                    cbar_kws={'label': 'Usage Frequency'})
        plt.xlabel('Expert Index', fontsize=12)
        plt.ylabel('Encoder Layer', fontsize=12)
        plt.title('Encoder Expert Usage Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'encoder_expert_usage_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved encoder expert usage heatmap to {output_path}")
    
    # Create heatmap for decoder
    if 'decoder' in expert_usage:
        decoder_usage = np.array(expert_usage['decoder'])
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(decoder_usage, annot=True, fmt='.3f', cmap='YlGnBu',
                    xticklabels=[f'E{i}' for i in range(decoder_usage.shape[1])],
                    yticklabels=[f'Layer {i+1}' for i in range(decoder_usage.shape[0])],
                    cbar_kws={'label': 'Usage Frequency'})
        plt.xlabel('Expert Index', fontsize=12)
        plt.ylabel('Decoder Layer', fontsize=12)
        plt.title('Decoder Expert Usage Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'decoder_expert_usage_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved decoder expert usage heatmap to {output_path}")


def compare_metrics(evaluation_files, model_names, output_dir):
    """Compare metrics across different models"""
    # Load all evaluation results
    all_results = []
    for eval_file in evaluation_files:
        with open(eval_file, 'r') as f:
            all_results.append(json.load(f))
    
    # Extract metrics for comparison
    metrics_to_compare = ['rouge1', 'rouge2', 'rougeL', 'bleu']
    
    # Prepare data
    metric_values = {metric: [] for metric in metrics_to_compare}
    
    for results in all_results:
        if 'rouge' in results:
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                metric_values[metric].append(results['rouge'][metric]['mean'])
        
        if 'bleu' in results:
            metric_values['bleu'].append(results['bleu']['mean'])
    
    # Create comparison bar plot
    x = np.arange(len(metrics_to_compare))
    width = 0.8 / len(model_names)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    
    for i, model_name in enumerate(model_names):
        values = [metric_values[metric][i] if len(metric_values[metric]) > i else 0 
                  for metric in metrics_to_compare]
        offset = width * (i - len(model_names)/2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison - Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_to_compare])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved model comparison plot to {output_path}")
    
    # Create detailed comparison table plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = [['Model'] + metrics_to_compare]
    
    for i, model_name in enumerate(model_names):
        row = [model_name]
        for metric in metrics_to_compare:
            if len(metric_values[metric]) > i:
                row.append(f"{metric_values[metric][i]:.4f}")
            else:
                row.append("N/A")
        table_data.append(row)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2] + [0.15] * len(metrics_to_compare))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(len(metrics_to_compare) + 1):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(model_names) + 1):
        for j in range(len(metrics_to_compare) + 1):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Detailed Model Comparison', fontsize=16, fontweight='bold', pad=20)
    output_path = os.path.join(output_dir, 'model_comparison_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved model comparison table to {output_path}")


def plot_all_metrics(evaluation_file, model_name, output_dir):
    """Plot all available metrics for a single model"""
    with open(evaluation_file, 'r') as f:
        results = json.load(f)
    
    # Collect all metrics
    metrics = {}
    
    if 'rouge' in results:
        metrics['ROUGE-1'] = results['rouge']['rouge1']['mean']
        metrics['ROUGE-2'] = results['rouge']['rouge2']['mean']
        metrics['ROUGE-L'] = results['rouge']['rougeL']['mean']
    
    if 'bleu' in results:
        metrics['BLEU'] = results['bleu']['mean']
    
    if 'bertscore' in results:
        metrics['BERTScore-F1'] = results['bertscore']['f1']['mean']
    
    if 'compression_ratio' in results:
        metrics['Compression Ratio'] = results['compression_ratio']['mean']
    
    if 'extractiveness' in results:
        metrics['Extractiveness'] = results['extractiveness']['mean']
    
    if 'factuality_summac' in results:
        metrics['Factuality (SummaC)'] = results['factuality_summac']['mean']
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metric_names)))
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{model_name} - All Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{model_name}_all_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved all metrics plot to {output_path}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot training history if provided
    if args.training_history:
        print("Plotting training history...")
        plot_training_history(args.training_history, args.output_dir)
    
    # Plot expert usage if provided
    if args.predictions_file:
        print("Plotting expert usage...")
        plot_expert_usage(args.predictions_file, args.output_dir)
        plot_expert_usage_heatmap(args.predictions_file, args.output_dir)
    
    # Plot single model metrics
    if args.evaluation_file and args.model_name:
        print(f"Plotting metrics for {args.model_name}...")
        plot_all_metrics(args.evaluation_file, args.model_name, args.output_dir)
    
    # Compare multiple models
    if args.compare_models and args.evaluation_files and args.model_names:
        print("Comparing models...")
        eval_files = args.evaluation_files.split(',')
        model_names = args.model_names.split(',')
        
        if len(eval_files) != len(model_names):
            print("Error: Number of evaluation files must match number of model names")
        else:
            compare_metrics(eval_files, model_names, args.output_dir)
    
    print("\nVisualization completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Model Results')
    
    parser.add_argument('--output_dir', type=str, default='./visualizations', help='Output directory')
    parser.add_argument('--training_history', type=str, help='Path to training history JSON file')
    parser.add_argument('--predictions_file', type=str, help='Path to predictions JSON file (with expert usage)')
    parser.add_argument('--evaluation_file', type=str, help='Path to evaluation JSON file')
    parser.add_argument('--model_name', type=str, help='Model name for single model plots')
    
    # For comparison
    parser.add_argument('--compare_models', action='store_true', help='Compare multiple models')
    parser.add_argument('--evaluation_files', type=str, help='Comma-separated paths to evaluation files')
    parser.add_argument('--model_names', type=str, help='Comma-separated model names')
    
    args = parser.parse_args()
    
    main(args)
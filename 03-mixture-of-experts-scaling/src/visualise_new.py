# visualize.py
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

def load_evaluation_results(eval_files):
    """Load evaluation results from multiple files"""
    results = {}
    for model_name, file_path in eval_files.items():
        with open(file_path, 'r') as f:
            results[model_name] = json.load(f)
    return results

def create_comparison_table(results, output_dir):
    """Create detailed comparison table"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    models = list(results.keys())
    table_data = []
    
    for model in models:
        row = [
            model,
            f"{results[model]['rouge']['rouge1']['mean']:.4f}",
            f"{results[model]['rouge']['rouge2']['mean']:.4f}",
            f"{results[model]['rouge']['rougeL']['mean']:.4f}",
            f"{results[model]['bleu']['mean']:.4f}",
            f"{results[model].get('bertscore', {}).get('f1', {}).get('mean', 0):.4f}"
        ]
        table_data.append(row)
    
    headers = ['Model', 'rouge1', 'rouge2', 'rougeL', 'bleu', 'bertscore-f1']
    
    # Create table with colors
    colors = []
    for i, row in enumerate(table_data):
        if i == 0:  # Assuming first is best (BART)
            colors.append(['#90EE90'] * len(row))
        elif i <= 2:
            colors.append(['#FFB6C1'] * len(row))
        else:
            colors.append(['#E0E0E0'] * len(row))
    
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     cellColours=colors,
                     colColours=['#4CAF50'] * len(headers))
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    plt.title('Detailed Model Comparison', fontsize=16, fontweight='bold', pad=20)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'detailed_comparison_table.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved: {output_dir}/detailed_comparison_table.png")


def create_metric_comparison_chart(results, output_dir):
    """Create bar chart comparing all metrics"""
    models = list(results.keys())
    metrics = ['ROUGE1', 'ROUGE2', 'ROUGEL', 'BLEU']
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(metrics))
    width = 0.15
    
    colors = ['#81C784', '#FF8A80', '#FFD54F', '#BA68C8', '#4FC3F7']
    
    for i, model in enumerate(models):
        values = [
            results[model]['rouge']['rouge1']['mean'],
            results[model]['rouge']['rouge2']['mean'],
            results[model]['rouge']['rougeL']['mean'],
            results[model]['bleu']['mean']
        ]
        
        offset = width * (i - len(models)/2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metric', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Comparison - Evaluation Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_metrics.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved: {output_dir}/model_comparison_metrics.png")


def create_bertscore_comparison(results, output_dir):
    """Create BERTScore comparison chart"""
    models = list(results.keys())
    
    precision = [results[m].get('bertscore', {}).get('precision', {}).get('mean', 0) for m in models]
    recall = [results[m].get('bertscore', {}).get('recall', {}).get('mean', 0) for m in models]
    f1 = [results[m].get('bertscore', {}).get('f1', {}).get('mean', 0) for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#4FC3F7', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#81C784', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1', color='#FFD54F', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('BERTScore', fontsize=13, fontweight='bold')
    ax.set_title('BERTScore Comparison Across Models', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.75, 0.95])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bertscore_comparison.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved: {output_dir}/bertscore_comparison.png")


def create_comprehensive_comparison(results, output_dir):
    """Create comprehensive comparison with all metrics including BERTScore"""
    models = list(results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comprehensive Model Evaluation', fontsize=18, fontweight='bold', y=0.995)
    
    colors = ['#81C784', '#FF8A80', '#FFD54F', '#BA68C8', '#4FC3F7']
    
    # ROUGE-1
    ax = axes[0, 0]
    rouge1 = [results[m]['rouge']['rouge1']['mean'] for m in models]
    bars = ax.bar(models, rouge1, color=colors, alpha=0.8)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_title('ROUGE-1', fontweight='bold')
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    
    # ROUGE-2
    ax = axes[0, 1]
    rouge2 = [results[m]['rouge']['rouge2']['mean'] for m in models]
    bars = ax.bar(models, rouge2, color=colors, alpha=0.8)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_title('ROUGE-2', fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    
    # ROUGE-L
    ax = axes[0, 2]
    rougel = [results[m]['rouge']['rougeL']['mean'] for m in models]
    bars = ax.bar(models, rougel, color=colors, alpha=0.8)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_title('ROUGE-L', fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    
    # BLEU
    ax = axes[1, 0]
    bleu = [results[m]['bleu']['mean'] for m in models]
    bars = ax.bar(models, bleu, color=colors, alpha=0.8)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_title('BLEU', fontweight='bold')
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    
    # BERTScore F1
    ax = axes[1, 1]
    bertscore_f1 = [results[m].get('bertscore', {}).get('f1', {}).get('mean', 0) for m in models]
    bars = ax.bar(models, bertscore_f1, color=colors, alpha=0.8)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_title('BERTScore-F1', fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.8, 0.95])
    
    # Summary radar chart
    ax = axes[1, 2]
    ax.remove()
    ax = fig.add_subplot(2, 3, 6, projection='polar')
    
    categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'BERTScore']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    for i, model in enumerate(models):
        values = [
            results[model]['rouge']['rouge1']['mean'],
            results[model]['rouge']['rouge2']['mean'],
            results[model]['rouge']['rougeL']['mean'],
            results[model]['bleu']['mean'],
            results[model].get('bertscore', {}).get('f1', {}).get('mean', 0)
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance Radar', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved: {output_dir}/comprehensive_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Visualize evaluation results')
    parser.add_argument('--evaluation_dir', type=str, default='./evaluations',
                       help='Directory containing evaluation JSON files')
    parser.add_argument('--output_dir', type=str, default='./visualizations/final_comparison',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Load all evaluation results
    eval_files = {
        'BART': os.path.join(args.evaluation_dir, 'evaluation_bart.json'),
        'T5-Base': os.path.join(args.evaluation_dir, 'evaluation_t5_base.json'),
        'Llama-1B': os.path.join(args.evaluation_dir, 'evaluation_llama_1b.json'),
        'MoE-TopK': os.path.join(args.evaluation_dir, 'evaluation_moe_topk.json'),
        'MoE-Hash': os.path.join(args.evaluation_dir, 'evaluation_moe_hash.json')
    }
    
    print("Loading evaluation results...")
    results = load_evaluation_results(eval_files)
    
    print("\nGenerating visualizations...")
    create_comparison_table(results, args.output_dir)
    create_metric_comparison_chart(results, args.output_dir)
    create_bertscore_comparison(results, args.output_dir)
    create_comprehensive_comparison(results, args.output_dir)
    
    print(f"\n✅ All visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
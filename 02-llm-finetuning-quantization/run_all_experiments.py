"""
Master Script to Run All Experiments
This script runs all components of the assignment sequentially
"""

import os
import sys
import time
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class ExperimentRunner:
    """Run all experiments and collect results"""
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
        self.start_time = time.time()
    
    def log(self, message):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def run_task1_evaluation_metrics(self):
        """Run Task 1: Evaluation Metrics Analysis"""
        self.log("="*70)
        self.log("TASK 1: EVALUATION METRICS ANALYSIS")
        self.log("="*70)
        
        try:
            import importlib
            eval_metrics = importlib.import_module('4_evaluation_metrics')
            
            self.log("Running evaluation metrics demonstrations...")
            eval_metrics.demonstrate_metrics()
            
            self.log("Running custom metric demonstration...")
            custom_metric = eval_metrics.SemanticCoherenceScore()
            custom_metric.explain_advantages()
            
            self.log("✓ Task 1 completed successfully!")
            return True
            
        except Exception as e:
            self.log(f"✗ Task 1 failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_task2_1_baseline(self):
        """Run Task 2.1: Baseline Fine-tuning"""
        self.log("\n" + "="*70)
        self.log("TASK 2.1: BASELINE FINE-TUNING")
        self.log("="*70)
        
        try:
            import importlib
            baseline_module = importlib.import_module('1_baseline_finetuning')
            
            self.log("Initializing GPT-2 trainer...")
            trainer = baseline_module.GPT2Trainer(model_name='gpt2', num_labels=4)
            
            self.log("Loading and preparing AG News dataset...")
            train_loader, test_loader = trainer.prepare_data(batch_size=16)
            
            self.log("Starting fine-tuning (this may take 25-30 minutes on GPU)...")
            trainer.train(train_loader, epochs=3)
            
            self.log("Evaluating baseline model...")
            metrics = trainer.evaluate(test_loader)
            
            self.log("Saving baseline model...")
            trainer.save_model('models/baseline_gpt2')
            
            self.log("Generating confusion matrix...")
            trainer.plot_confusion_matrix(
                metrics['confusion_matrix'],
                f'{self.output_dir}/baseline_confusion_matrix.png'
            )
            
            # Store results
            self.results['Baseline FP32'] = metrics
            
            self.log("✓ Task 2.1 completed successfully!")
            self.log(f"  Accuracy: {metrics['accuracy']:.4f}")
            self.log(f"  Model Size: {metrics['model_size_mb']:.2f} MB")
            
            return True, test_loader
            
        except Exception as e:
            self.log(f"✗ Task 2.1 failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None
    
    def run_task2_2_quantization_scratch(self, test_loader=None):
        """Run Task 2.2: Quantization from Scratch"""
        self.log("\n" + "="*70)
        self.log("TASK 2.2: POST-TRAINING QUANTIZATION FROM SCRATCH")
        self.log("="*70)
        
        try:
            import importlib
            quant_module = importlib.import_module('2_quantization_scratch')
            
            self.log("Loading baseline model for quantization...")
            quant_model = quant_module.QuantizedGPT2Model('models/baseline_gpt2')
            
            self.log("Applying INT8 quantization...")
            quant_model.quantize_model()
            
            if test_loader is None:
                self.log("Preparing test data...")
                test_loader = quant_module.prepare_test_data(quant_model.tokenizer)
            
            self.log("Evaluating quantized model...")
            metrics = quant_model.evaluate(test_loader)
            
            self.log("Generating confusion matrix...")
            quant_model.plot_confusion_matrix(
                metrics['confusion_matrix'],
                f'{self.output_dir}/int8_scratch_confusion_matrix.png'
            )
            
            # Store results
            self.results['INT8 Scratch'] = metrics
            
            self.log("✓ Task 2.2 completed successfully!")
            self.log(f"  Accuracy: {metrics['accuracy']:.4f}")
            self.log(f"  Model Size: {metrics['model_size_mb']:.2f} MB")
            
            if 'Baseline FP32' in self.results:
                compression = self.results['Baseline FP32']['model_size_mb'] / metrics['model_size_mb']
                self.log(f"  Compression: {compression:.2f}x")
            
            return True
            
        except Exception as e:
            self.log(f"✗ Task 2.2 failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_task2_3_bitsandbytes(self, test_loader=None):
        """Run Task 2.3: BitsAndBytes Quantization"""
        self.log("\n" + "="*70)
        self.log("TASK 2.3: BITSANDBYTES QUANTIZATION")
        self.log("="*70)
        
        try:
            import importlib
            bnb_module = importlib.import_module('3_bitsandbytes_quant')
            
            quantizer = bnb_module.BitsAndBytesQuantizer('models/baseline_gpt2')
            
            if test_loader is None:
                self.log("Preparing test data...")
                test_loader = bnb_module.prepare_test_data(quantizer.tokenizer)
            
            # INT8 Quantization
            self.log("Loading and evaluating 8-bit model...")
            model_8bit = quantizer.load_8bit_model()
            metrics_8bit = quantizer.evaluate(model_8bit, test_loader)
            
            quantizer.plot_confusion_matrix(
                metrics_8bit['confusion_matrix'],
                'INT8 (BitsAndBytes)',
                f'{self.output_dir}/int8_bitsandbytes_confusion_matrix.png'
            )
            
            self.results['INT8 BitsAndBytes'] = metrics_8bit
            
            self.log("✓ INT8 quantization completed!")
            self.log(f"  Accuracy: {metrics_8bit['accuracy']:.4f}")
            self.log(f"  Model Size: {metrics_8bit['model_size_mb']:.2f} MB")
            
            # Clear memory
            del model_8bit
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # NF4 Quantization
            self.log("\nLoading and evaluating 4-bit NF4 model...")
            model_4bit = quantizer.load_4bit_model()
            metrics_4bit = quantizer.evaluate(model_4bit, test_loader)
            
            quantizer.plot_confusion_matrix(
                metrics_4bit['confusion_matrix'],
                'NF4 (BitsAndBytes)',
                f'{self.output_dir}/nf4_bitsandbytes_confusion_matrix.png'
            )
            
            self.results['NF4 BitsAndBytes'] = metrics_4bit
            
            self.log("✓ NF4 quantization completed!")
            self.log(f"  Accuracy: {metrics_4bit['accuracy']:.4f}")
            self.log(f"  Model Size: {metrics_4bit['model_size_mb']:.2f} MB")
            
            self.log("✓ Task 2.3 completed successfully!")
            
            return True
            
        except Exception as e:
            self.log(f"✗ Task 2.3 failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_final_report(self):
        """Generate comprehensive comparison report"""
        self.log("\n" + "="*70)
        self.log("GENERATING FINAL REPORT")
        self.log("="*70)
        
        try:
            import importlib
            compare_module = importlib.import_module('5_compare_all_models')
            
            comparator = compare_module.ModelComparator()
            
            # Add all results
            for model_name, metrics in self.results.items():
                comparator.add_result(model_name, metrics)
            
            # Generate full report
            comparator.generate_full_report(self.output_dir)
            
            self.log("✓ Final report generated successfully!")
            
            return True
            
        except Exception as e:
            self.log(f"✗ Report generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_summary(self):
        """Print experiment summary"""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        self.log("\n" + "="*70)
        self.log("EXPERIMENT SUMMARY")
        self.log("="*70)
        self.log(f"Total Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
        self.log(f"Results saved to: {self.output_dir}/")
        
        if self.results:
            self.log("\nFinal Results:")
            self.log("-" * 70)
            
            for model_name, metrics in self.results.items():
                self.log(f"\n{model_name}:")
                self.log(f"  Accuracy: {metrics['accuracy']:.4f}")
                self.log(f"  F1-Score: {metrics['f1_score']:.4f}")
                self.log(f"  Size: {metrics['model_size_mb']:.2f} MB")
                self.log(f"  Inference: {metrics['avg_inference_time_ms']:.2f} ms")
        
        self.log("\n" + "="*70)
        self.log("EXPERIMENT COMPLETED!")
        self.log("="*70)
        self.log("\nCheck the 'results/' directory for:")
        self.log("  - comparison_table.csv (all metrics)")
        self.log("  - performance_comparison.png (charts)")
        self.log("  - efficiency_comparison.png (size/speed)")
        self.log("  - confusion matrices for all models")


def main():
    """Main execution function"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║         Advanced NLP Assignment 2 - Complete Execution            ║
    ║              Fine-tuning and Quantization Analysis                ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    runner = ExperimentRunner(output_dir='results')
    
    # Task 1: Evaluation Metrics
    runner.run_task1_evaluation_metrics()
    
    # Task 2.1: Baseline Fine-tuning
    success, test_loader = runner.run_task2_1_baseline()
    
    if not success:
        print("\n⚠ Baseline training failed. Cannot proceed with quantization.")
        sys.exit(1)
    
    # Task 2.2: Quantization from Scratch
    runner.run_task2_2_quantization_scratch(test_loader)
    
    # Task 2.3: BitsAndBytes Quantization
    runner.run_task2_3_bitsandbytes(test_loader)
    
    # Generate Final Report
    runner.generate_final_report()
    
    # Print Summary
    runner.print_summary()
    
    print("\n✓ All experiments completed! Check the 'results/' directory for outputs.")


if __name__ == '__main__':
    main()
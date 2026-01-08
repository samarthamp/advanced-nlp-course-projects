"""
Environment Verification Script
Checks if all required dependencies are installed and working
"""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    package = package_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {package} installed")
        return True
    except ImportError:
        print(f"✗ {package} NOT installed")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available (Device: {torch.cuda.get_device_name(0)})")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return True
        else:
            print("⚠ CUDA not available (will use CPU)")
            return False
    except:
        print("✗ Cannot check CUDA")
        return False

def check_disk_space():
    """Check available disk space"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (2**30)
        print(f"✓ Free disk space: {free_gb} GB")
        if free_gb < 5:
            print("  ⚠ Warning: Less than 5GB free space")
        return True
    except:
        print("⚠ Cannot check disk space")
        return False

def main():
    """Run all checks"""
    print("="*70)
    print("ENVIRONMENT VERIFICATION FOR ASSIGNMENT 2")
    print("="*70)
    
    print("\nPython Version:")
    print(f"  {sys.version}")
    
    print("\n" + "-"*70)
    print("Checking Core Dependencies:")
    print("-"*70)
    
    all_ok = True
    
    # Core libraries
    core_libs = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
        ('accelerate', 'Accelerate'),
    ]
    
    for module, name in core_libs:
        if not check_import(module, name):
            all_ok = False
    
    print("\n" + "-"*70)
    print("Checking Quantization Libraries:")
    print("-"*70)
    
    check_import('bitsandbytes', 'BitsAndBytes')
    
    print("\n" + "-"*70)
    print("Checking Evaluation Metric Libraries:")
    print("-"*70)
    
    eval_libs = [
        ('rouge_score', 'ROUGE Score'),
        ('nltk', 'NLTK'),
        ('bert_score', 'BERT Score'),
        ('sentence_transformers', 'Sentence Transformers'),
    ]
    
    for module, name in eval_libs:
        check_import(module, name)
    
    print("\n" + "-"*70)
    print("Checking Data Science Libraries:")
    print("-"*70)
    
    ds_libs = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('tqdm', 'TQDM'),
    ]
    
    for module, name in ds_libs:
        check_import(module, name)
    
    print("\n" + "-"*70)
    print("Checking Hardware:")
    print("-"*70)
    
    has_cuda = check_cuda()
    check_disk_space()
    
    print("\n" + "="*70)
    
    if all_ok:
        print("✓ ENVIRONMENT READY!")
        print("\nYou can now run the experiments:")
        print("  python run_all_experiments.py")
        
        if has_cuda:
            print("\n✓ GPU detected! Training will be fast (~50-60 min)")
        else:
            print("\n⚠ No GPU. Training will be slower (~2+ hours)")
            print("  Consider using Google Colab for free GPU access")
    else:
        print("⚠ MISSING DEPENDENCIES!")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
    
    print("="*70)
    
    # Additional suggestions
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠ No GPU detected. Training will be slower.")
            print("  Consider using:")
            print("  - Google Colab (free GPU)")
            print("  - Kaggle Notebooks (free GPU)")
            print("  - Reduce dataset size for faster CPU training")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            if "4060" in gpu_name:
                print(f"✓ {gpu_name} detected - Perfect for this assignment!")
                print("  Expected training time: 50-60 minutes")
            else:
                print(f"✓ {gpu_name} detected - Good for training!")
    except:
        pass
    
    try:
        import nltk
        print("\n💡 Don't forget to download NLTK data:")
        print("  python -c \"import nltk; nltk.download('punkt')\"")
    except:
        pass
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
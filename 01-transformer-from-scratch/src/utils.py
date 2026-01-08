import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from collections import Counter
import pickle
import os

class Tokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def build_vocab(self, sentences, min_freq=1):
        counter = Counter()
        for sentence in sentences:
            words = sentence.strip().split()
            counter.update(words)
        
        # Special tokens
        self.word2idx = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '<unk>': 3
        }
        
        idx = 4
        for word, freq in counter.items():
            if freq >= min_freq:
                self.word2idx[word] = idx
                idx += 1
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
    def encode(self, sentence, max_length=None, add_special_tokens=True):
        words = sentence.strip().split()
        if add_special_tokens:
            words = ['<sos>'] + words + ['<eos>']
        
        indices = [self.word2idx.get(word, self.word2idx['<unk>']) for word in words]
        
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices += [self.word2idx['<pad>']] * (max_length - len(indices))
        
        return indices
    
    def decode(self, indices):
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, '<unk>')
            if word == '<eos>':
                break
            if word not in ['<pad>', '<sos>']:
                words.append(word)
        return ' '.join(words)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab_size': self.vocab_size
            }, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.vocab_size = data['vocab_size']

class Dataset:
    def __init__(self, src_sentences, tgt_sentences, src_tokenizer, tgt_tokenizer, max_length=128):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        src_indices = self.src_tokenizer.encode(src_sentence, self.max_length, add_special_tokens=False)
        tgt_indices = self.tgt_tokenizer.encode(tgt_sentence, self.max_length, add_special_tokens=True)
        
        # Create decoder input (without last token) and target (without first token)
        decoder_input = tgt_indices[:-1]
        target = tgt_indices[1:]
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(decoder_input, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }

def create_padding_mask(seq, pad_idx=0):
    """Create padding mask for attention"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    """Create look-ahead mask for decoder self-attention"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0

def load_data(file_path, max_samples=None):
    """Load parallel data from file"""
    src_sentences = []
    tgt_sentences = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                src_sentences.append(parts[0])
                tgt_sentences.append(parts[1])
    
    return src_sentences, tgt_sentences

def split_data(src_sentences, tgt_sentences, train_ratio=0.8, val_ratio=0.1):
    """Split data into train, validation, and test sets"""
    total_size = len(src_sentences)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Create paired data first, then shuffle
    paired_data = list(zip(src_sentences, tgt_sentences))
    np.random.shuffle(paired_data)  # This keeps pairs together
    
    # Split the shuffled pairs
    train_pairs = paired_data[:train_size]
    val_pairs = paired_data[train_size:train_size + val_size]
    test_pairs = paired_data[train_size + val_size:]
    
    # Unzip back into separate lists
    train_src, train_tgt = zip(*train_pairs) if train_pairs else ([], [])
    val_src, val_tgt = zip(*val_pairs) if val_pairs else ([], [])
    test_src, test_tgt = zip(*test_pairs) if test_pairs else ([], [])
    
    return (list(train_src), list(train_tgt)), (list(val_src), list(val_tgt)), (list(test_src), list(test_tgt))

def calculate_bleu(references, hypotheses):
    """Improved BLEU score calculation with proper smoothing"""
    from collections import Counter
    import math
    
    def get_ngrams(tokens, n):
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    
    def bleu_score(reference, hypothesis, max_n=4):
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if len(hyp_tokens) == 0:
            return 0.0
        
        # Brevity penalty
        ref_len = len(ref_tokens)
        hyp_len = len(hyp_tokens)
        
        if hyp_len > ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
        
        # Calculate precision for each n-gram with smoothing
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = Counter(get_ngrams(ref_tokens, n))
            hyp_ngrams = Counter(get_ngrams(hyp_tokens, n))
            
            if len(hyp_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            # Clipped counts
            clipped_counts = 0
            for ngram in hyp_ngrams:
                clipped_counts += min(hyp_ngrams[ngram], ref_ngrams.get(ngram, 0))
            
            total_hyp_ngrams = sum(hyp_ngrams.values())
            
            if total_hyp_ngrams == 0:
                precision = 0.0
            else:
                precision = clipped_counts / total_hyp_ngrams
            
            # Add smoothing for zero precision
            if precision == 0.0:
                precision = 1e-7
            
            precisions.append(precision)
        
        # Geometric mean
        if any(p == 0 for p in precisions):
            return 0.0
        
        log_precisions = [math.log(p) for p in precisions]
        geo_mean = math.exp(sum(log_precisions) / len(log_precisions))
        
        return bp * geo_mean * 100  # Return as percentage
    
    total_bleu = 0.0
    valid_pairs = 0
    
    for ref, hyp in zip(references, hypotheses):
        if ref.strip() and hyp.strip():  # Skip empty strings
            total_bleu += bleu_score(ref, hyp)
            valid_pairs += 1
    
    return total_bleu / valid_pairs if valid_pairs > 0 else 0.0

class LearningRateScheduler:
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        step = max(1, step)  # Avoid division by zero
        return self.d_model ** -0.5 * min(step ** -0.5, step * self.warmup_steps ** -1.5)

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

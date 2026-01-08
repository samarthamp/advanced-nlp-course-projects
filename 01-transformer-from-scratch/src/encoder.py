import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super().__init__()
        self.d_model = d_model
        
        # Create frequency tensor for rotation
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute position encodings (for compatibility with trained model)
        position = torch.arange(max_length).unsqueeze(1).float()
        freqs = torch.einsum('i,j->ij', position.squeeze(), inv_freq)
        
        # Store cos and sin
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q, k, seq_len):
        """Apply rotary position embedding to query and key tensors."""
        # q, k have shape: (batch, num_heads, seq_len, d_k)
        d_k = q.size(-1)
        device = q.device
        
        # Safely handle cached tensors with bounds checking
        max_cached_len = self.cos_cached.size(0)
        if seq_len > max_cached_len:
            # Fallback to dynamic computation for longer sequences
            position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
            inv_freq = self.inv_freq[:d_k//2].to(device)
            freqs = torch.einsum('i,j->ij', position.squeeze(), inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
        else:
            # Use cached values for normal sequences
            cos = self.cos_cached[:seq_len, :d_k//2].to(device)
            sin = self.sin_cached[:seq_len, :d_k//2].to(device)
        
        # Create cos and sin for full d_k by repeating each element
        cos_full = torch.cat([cos, cos], dim=-1)  # (seq_len, d_k)
        sin_full = torch.cat([sin, sin], dim=-1)  # (seq_len, d_k)
        
        # Reshape for broadcasting to match q, k dimensions
        cos_full = cos_full.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_k)
        sin_full = sin_full.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_k)
        
        # Apply rotation
        q_embed = (q * cos_full) + (self.rotate_half(q) * sin_full)
        k_embed = (k * cos_full) + (self.rotate_half(k) * sin_full)
        
        return q_embed, k_embed

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_length=128):
        super().__init__()
        self.num_heads = num_heads
        self.max_length = max_length
        
        # Simplified relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_length - 1), num_heads))
        
        # Create a simple relative position index
        relative_position_index = torch.zeros(max_length, max_length, dtype=torch.long)
        for i in range(max_length):
            for j in range(max_length):
                relative_position_index[i, j] = i - j + max_length - 1
                
        self.register_buffer("relative_position_index", relative_position_index)
        
    def forward(self, seq_len):
        # Clamp seq_len to max_length to avoid index errors
        seq_len = min(seq_len, self.max_length)
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:seq_len, :seq_len]
        ]  # (seq_len, seq_len, num_heads)
        
        return relative_position_bias.permute(2, 0, 1)  # (num_heads, seq_len, seq_len)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, use_relative_position=False, max_length=128, use_rope=False):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Use separate projections to avoid any dimension issues
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.use_relative_position = use_relative_position
        
        if use_relative_position:
            self.relative_position_bias = RelativePositionBias(num_heads, max_length)
        
        if use_rope:
            self.rope = RotaryPositionalEmbedding(d_model, max_length)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        q_seq_len = query.size(1)  # Query sequence length
        kv_seq_len = key.size(1)   # Key/Value sequence length (can be different!)
        
        # Apply linear transformations
        q = self.q_linear(query)   # (batch, q_seq_len, d_model)
        k = self.k_linear(key)     # (batch, kv_seq_len, d_model) 
        v = self.v_linear(value)   # (batch, kv_seq_len, d_model)
        
        # Reshape using the correct sequence lengths
        q = q.view(batch_size, q_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, kv_seq_len, self.num_heads, self.d_k).transpose(1, 2)  
        v = v.view(batch_size, kv_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE if it's available (only for self-attention where seq lengths match)
        if hasattr(self, 'rope') and q_seq_len == kv_seq_len:
            q, k = self.rope.apply_rotary_pos_emb(q, k, q_seq_len)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative position bias (only for self-attention where q_seq_len == kv_seq_len)
        if self.use_relative_position and q_seq_len == kv_seq_len:
            rel_bias = self.relative_position_bias(q_seq_len)
            scores = scores + rel_bias.unsqueeze(0)
        
        # Apply mask
        if mask is not None:
            if mask.dim() == 2:  # (batch, seq_len)
                if mask.size(1) == q_seq_len:
                    # Padding mask for query
                    mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, q_seq_len)
                elif mask.size(1) == kv_seq_len:
                    # Padding mask for key/value
                    mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, kv_seq_len)
            elif mask.dim() == 3:  # (batch, q_seq_len, kv_seq_len) 
                mask = mask.unsqueeze(1)  # (batch, 1, q_seq_len, kv_seq_len)
            elif mask.dim() == 4:  # Already correct shape
                pass
            else:
                raise ValueError(f"Invalid mask shape: {mask.shape}")
            
            scores = scores.masked_fill(mask == 0, -1e4)
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)  # (batch, num_heads, q_seq_len, d_k)
        
        # Concatenate heads: (batch, num_heads, q_seq_len, d_k) -> (batch, q_seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.d_model)
        
        # Final linear transformation
        output = self.out_linear(context)
        
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_relative_position=False, max_length=128, use_rope=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_relative_position, max_length, use_rope)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, 
                 max_length=5000, dropout=0.1, positional_encoding='rope'):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.positional_encoding = positional_encoding
        if positional_encoding == 'rope':
            self.pos_encoding = None  # Don't add to embeddings
            use_relative_position = False
            use_rope = True
        else:  # relative position bias
            self.pos_encoding = None
            use_relative_position = True
            use_rope = False
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, use_relative_position, max_length, use_rope)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, mask=None):
        x = self.embedding(src) * math.sqrt(self.d_model)
        
        # Don't add positional encoding for RoPE - it's applied in attention
        # Don't add anything for relative position either - it's applied in attention
        
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x
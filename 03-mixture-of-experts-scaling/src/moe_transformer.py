import torch
import torch.nn as nn
import math
from moe_layer import SparseMoELayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class TransformerEncoderLayerWithMoE(nn.Module):
    """Transformer encoder layer with MoE replacing FFN"""
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        router_type: str = 'topk',
        dropout: float = 0.1,
        use_load_balancing: bool = True
    ):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # MoE layer (replaces standard FFN)
        self.moe = SparseMoELayer(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k,
            router_type=router_type,
            dropout=dropout,
            use_load_balancing=use_load_balancing
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        src: torch.Tensor, 
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None
    ):
        """
        Args:
            src: [batch_size, seq_len, d_model]
            src_mask: [seq_len, seq_len]
            src_key_padding_mask: [batch_size, seq_len]
        """
        # Self attention block
        src2, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # MoE block
        src2, load_balance_loss = self.moe(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        return src, load_balance_loss


class TransformerDecoderLayerWithMoE(nn.Module):
    """Transformer decoder layer with MoE replacing FFN"""
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        router_type: str = 'topk',
        dropout: float = 0.1,
        use_load_balancing: bool = True
    ):
        super().__init__()
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # MoE layer
        self.moe = SparseMoELayer(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k,
            router_type=router_type,
            dropout=dropout,
            use_load_balancing=use_load_balancing
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None
    ):
        """
        Args:
            tgt: [batch_size, tgt_len, d_model]
            memory: [batch_size, src_len, d_model]
        """
        # Self attention
        tgt2, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention
        tgt2, _ = self.cross_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        # MoE block
        tgt2, load_balance_loss = self.moe(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, load_balance_loss


class MoETransformer(nn.Module):
    """Transformer model with MoE layers for sequence-to-sequence tasks"""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        num_experts: int = 8,
        top_k: int = 2,
        router_type: str = 'topk',
        dropout: float = 0.1,
        max_len: int = 5000,
        use_load_balancing: bool = True,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.use_load_balancing = use_load_balancing
        
        # Embeddings
        self.encoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithMoE(
                d_model, nhead, d_ff, num_experts, top_k, 
                router_type, dropout, use_load_balancing
            ) for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayerWithMoE(
                d_model, nhead, d_ff, num_experts, top_k,
                router_type, dropout, use_load_balancing
            ) for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None
    ):
        """
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
            src_key_padding_mask: [batch_size, src_len]
            tgt_key_padding_mask: [batch_size, tgt_len]
        """
        # Create target mask
        tgt_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # Encode
        memory, encoder_lb_loss = self.encode(src, src_key_padding_mask)
        
        # Decode
        output, decoder_lb_loss = self.decode(
            tgt, memory, tgt_mask, 
            src_key_padding_mask, tgt_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        # Combine load balancing losses
        total_lb_loss = 0
        if encoder_lb_loss is not None:
            total_lb_loss += encoder_lb_loss
        if decoder_lb_loss is not None:
            total_lb_loss += decoder_lb_loss
            
        return logits, total_lb_loss if total_lb_loss != 0 else None
    
    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None):
        """Encode source sequence"""
        # Embed and add positional encoding
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Pass through encoder layers
        total_lb_loss = 0
        for layer in self.encoder_layers:
            src, lb_loss = layer(src, src_key_padding_mask=src_key_padding_mask)
            if lb_loss is not None:
                total_lb_loss += lb_loss
        
        return src, total_lb_loss if total_lb_loss != 0 else None
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None
    ):
        """Decode target sequence"""
        # Embed and add positional encoding
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_decoder(tgt)
        
        # Pass through decoder layers
        total_lb_loss = 0
        for layer in self.decoder_layers:
            tgt, lb_loss = layer(
                tgt, memory, tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            if lb_loss is not None:
                total_lb_loss += lb_loss
        
        return tgt, total_lb_loss if total_lb_loss != 0 else None
    
    def get_all_expert_usage(self):
        """Get expert usage from all MoE layers"""
        usage_stats = {'encoder': [], 'decoder': []}
        
        for i, layer in enumerate(self.encoder_layers):
            usage_stats['encoder'].append(layer.moe.get_expert_usage())
        
        for i, layer in enumerate(self.decoder_layers):
            usage_stats['decoder'].append(layer.moe.get_expert_usage())
        
        return usage_stats
    
    def reset_all_expert_usage(self):
        """Reset expert usage statistics"""
        for layer in self.encoder_layers:
            layer.moe.reset_expert_usage()
        for layer in self.decoder_layers:
            layer.moe.reset_expert_usage()
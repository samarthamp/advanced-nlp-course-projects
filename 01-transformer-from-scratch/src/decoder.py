import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from encoder import RotaryPositionalEmbedding, RelativePositionBias, MultiHeadAttention, PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_relative_position=False, max_length=128, use_rope=False):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_model, num_heads, dropout, use_relative_position, max_length, use_rope)
        self.cross_attention = MultiHeadAttention(
            d_model, num_heads, dropout, use_relative_position, max_length, use_rope=False)  # Cross-attention doesn't use RoPE
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        # Masked self-attention with residual connection and layer norm
        self_attn_output, _ = self.self_attention(x, x, x, self_attention_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention with residual connection and layer norm
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, cross_attention_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, cross_attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, 
                 max_length=5000, dropout=0.1, positional_encoding='rope'):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = positional_encoding
        if positional_encoding == 'rope':
            self.pos_encoding = None  # Don't add to embeddings
            use_relative_position = False
            use_rope = True
        else:  # relative position bias
            self.pos_encoding = None
            use_relative_position = True
            use_rope = False
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, use_relative_position, max_length, use_rope)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        # Embedding and positional encoding
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # Don't add positional encoding for RoPE - it's applied in attention
        # Don't add anything for relative position either - it's applied in attention
        
        x = self.dropout(x)
        
        # Pass through decoder layers
        cross_attention_weights = []
        for layer in self.layers:
            x, cross_attn_weights = layer(x, encoder_output, self_attention_mask, cross_attention_mask)
            cross_attention_weights.append(cross_attn_weights)
        
        return x, cross_attention_weights

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_length=5000, dropout=0.1, positional_encoding='rope'):
        super().__init__()
        
        from encoder import Encoder
        
        self.encoder = Encoder(
            src_vocab_size, d_model, num_heads, num_encoder_layers, d_ff,
            max_length, dropout, positional_encoding
        )
        
        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_heads, num_decoder_layers, d_ff,
            max_length, dropout, positional_encoding
        )
        
        self.projection = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self.init_parameters()
        
    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_masks(self, src, tgt):
        # Source padding mask
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # Target padding mask
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        
        # Target look-ahead mask
        seq_len = tgt.size(1)
        look_ahead_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0).to(tgt.device)
        
        # Combined target mask
        tgt_mask = tgt_mask & ~look_ahead_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.create_masks(src, tgt)
        
        # Encoder
        encoder_output = self.encoder(src, src_mask)
        
        # Decoder
        decoder_output, cross_attention_weights = self.decoder(
            tgt, encoder_output, tgt_mask, src_mask)
        
        # Output projection
        output = self.projection(decoder_output)
        
        return output, cross_attention_weights
    
    def encode(self, src):
        """Encode source sequence"""
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return self.encoder(src, src_mask)
    
    def decode_step(self, tgt, encoder_output, src_mask):
        """Single decoding step"""
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        
        # Look-ahead mask
        look_ahead_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0).to(tgt.device)
        tgt_mask = tgt_mask & ~look_ahead_mask
        
        # Decoder
        decoder_output, _ = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        
        # Output projection
        output = self.projection(decoder_output)
        
        return output

class DecodingStrategy:
    def __init__(self, model, src_tokenizer, tgt_tokenizer, device, max_length=100):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        self.max_length = max_length
        
    def greedy_decode(self, src_sentence):
        """Greedy decoding strategy"""
        self.model.eval()
        
        # Tokenize source
        src_indices = self.src_tokenizer.encode(src_sentence, add_special_tokens=False)
        src = torch.tensor([src_indices], dtype=torch.long).to(self.device)
        
        # Encode source
        encoder_output = self.model.encode(src)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # Initialize target with SOS token
        tgt = torch.tensor([[self.tgt_tokenizer.word2idx['<sos>']]], dtype=torch.long).to(self.device)
        
        for _ in range(self.max_length - 1):
            # Get next token probabilities
            output = self.model.decode_step(tgt, encoder_output, src_mask)
            
            # Select token with highest probability
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Append to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == self.tgt_tokenizer.word2idx['<eos>']:
                break
        
        # Decode to string
        output_indices = tgt[0].cpu().numpy().tolist()
        return self.tgt_tokenizer.decode(output_indices)
    
    def beam_search_decode(self, src_sentence, beam_size=5):
        """Beam search decoding strategy"""
        self.model.eval()
        
        # Tokenize source
        src_indices = self.src_tokenizer.encode(src_sentence, add_special_tokens=False)
        src = torch.tensor([src_indices], dtype=torch.long).to(self.device)
        
        # Encode source
        encoder_output = self.model.encode(src)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # Initialize beams
        sos_token = self.tgt_tokenizer.word2idx['<sos>']
        eos_token = self.tgt_tokenizer.word2idx['<eos>']
        
        beams = [(torch.tensor([[sos_token]], dtype=torch.long).to(self.device), 0.0)]
        completed_beams = []
        
        for step in range(self.max_length - 1):
            new_beams = []
            
            for tgt_seq, score in beams:
                if tgt_seq[0, -1].item() == eos_token:
                    completed_beams.append((tgt_seq, score))
                    continue
                
                # Get next token probabilities
                output = self.model.decode_step(tgt_seq, encoder_output, src_mask)
                log_probs = F.log_softmax(output[:, -1, :], dim=-1)
                
                # Get top-k tokens
                top_log_probs, top_indices = log_probs.topk(beam_size)
                
                for i in range(beam_size):
                    next_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                    new_score = score + top_log_probs[0, i].item()
                    new_seq = torch.cat([tgt_seq, next_token], dim=1)
                    new_beams.append((new_seq, new_score))
            
            # Keep top beam_size beams
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]
            
            # Check if all beams ended
            if len(completed_beams) >= beam_size:
                break
        
        # Add remaining beams to completed
        completed_beams.extend(beams)
        
        # Return best sequence
        completed_beams.sort(key=lambda x: x[1], reverse=True)
        best_seq = completed_beams[0][0]
        
        # Decode to string
        output_indices = best_seq[0].cpu().numpy().tolist()
        return self.tgt_tokenizer.decode(output_indices)
    
    def top_k_sampling_decode(self, src_sentence, k=50, temperature=1.0):
        """Top-k sampling decoding strategy"""
        self.model.eval()
        
        # Tokenize source
        src_indices = self.src_tokenizer.encode(src_sentence, add_special_tokens=False)
        src = torch.tensor([src_indices], dtype=torch.long).to(self.device)
        
        # Encode source
        encoder_output = self.model.encode(src)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # Initialize target with SOS token
        tgt = torch.tensor([[self.tgt_tokenizer.word2idx['<sos>']]], dtype=torch.long).to(self.device)
        
        for _ in range(self.max_length - 1):
            # Get next token probabilities
            output = self.model.decode_step(tgt, encoder_output, src_mask)
            logits = output[:, -1, :] / temperature
            
            # Apply top-k filtering
            top_k_logits, top_k_indices = logits.topk(k)
            
            # Sample from top-k
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, 1)
            next_token = top_k_indices.gather(1, next_token_idx)
            
            # Append to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == self.tgt_tokenizer.word2idx['<eos>']:
                break
        
        # Decode to string
        output_indices = tgt[0].cpu().numpy().tolist()
        return self.tgt_tokenizer.decode(output_indices)

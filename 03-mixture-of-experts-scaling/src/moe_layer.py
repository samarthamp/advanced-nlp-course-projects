import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class Expert(nn.Module):
    """Single expert FFN"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.relu(self.w1(x))))


class HashRouter(nn.Module):
    """Hash-based routing mechanism"""
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            expert_indices: [batch_size, seq_len, top_k]
            expert_weights: [batch_size, seq_len, top_k]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Simple hash function based on input features
        hash_values = torch.sum(x, dim=-1).abs()  # [batch_size, seq_len]
        hash_values = (hash_values * 1000).long() % self.num_experts
        
        # Create top_k indices (hash + offset)
        expert_indices = []
        for k in range(self.top_k):
            indices = (hash_values + k) % self.num_experts
            expert_indices.append(indices)
        
        expert_indices = torch.stack(expert_indices, dim=-1)  # [batch_size, seq_len, top_k]
        
        # Equal weights for hash routing
        expert_weights = torch.ones_like(expert_indices, dtype=x.dtype) / self.top_k
        
        return expert_indices, expert_weights


class TopKRouter(nn.Module):
    """Token choice Top-K routing mechanism"""
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            expert_indices: [batch_size, seq_len, top_k]
            expert_weights: [batch_size, seq_len, top_k]
        """
        # Compute gating scores
        logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        
        # Apply softmax
        probs = F.softmax(logits, dim=-1)
        
        # Select top-k experts
        expert_weights, expert_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # Renormalize weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        return expert_indices, expert_weights


class LoadBalancer:
    """Load balancing loss computation"""
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        
    def compute_load_balancing_loss(
        self, 
        expert_indices: torch.Tensor, 
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing loss to encourage uniform expert usage
        
        Args:
            expert_indices: [batch_size, seq_len, top_k]
            expert_weights: [batch_size, seq_len, top_k]
        Returns:
            loss: scalar tensor
        """
        batch_size, seq_len, top_k = expert_indices.shape
        
        # Count how many times each expert is used
        expert_counts = torch.zeros(
            self.num_experts, 
            device=expert_indices.device, 
            dtype=expert_weights.dtype
        )
        
        # Accumulate weights for each expert
        for k in range(top_k):
            expert_counts.scatter_add_(
                0, 
                expert_indices[:, :, k].flatten(), 
                expert_weights[:, :, k].flatten()
            )
        
        # Compute mean and variance
        total_tokens = batch_size * seq_len
        expert_counts = expert_counts / total_tokens
        
        # Loss is variance of expert usage (encourage uniformity)
        mean_count = expert_counts.mean()
        loss = ((expert_counts - mean_count) ** 2).mean()
        
        return loss * self.num_experts  # Scale by number of experts


class SparseMoELayer(nn.Module):
    """Sparse Mixture of Experts Layer"""
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        num_experts: int, 
        top_k: int = 2,
        router_type: str = 'topk',  # 'topk' or 'hash'
        dropout: float = 0.1,
        use_load_balancing: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_load_balancing = use_load_balancing
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        
        # Create router
        if router_type == 'topk':
            self.router = TopKRouter(d_model, num_experts, top_k)
        elif router_type == 'hash':
            self.router = HashRouter(d_model, num_experts, top_k)
        else:
            raise ValueError(f"Unknown router type: {router_type}")
        
        # Load balancer
        self.load_balancer = LoadBalancer(num_experts)
        
        # Track expert usage
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
            load_balance_loss: scalar tensor or None
        """
        batch_size, seq_len, d_model = x.shape
        
        # Route tokens to experts
        expert_indices, expert_weights = self.router(x)  # Both [batch_size, seq_len, top_k]
        
        # Update expert usage statistics (for visualization)
        with torch.no_grad():
            for k in range(self.top_k):
                for e in range(self.num_experts):
                    mask = (expert_indices[:, :, k] == e)
                    self.expert_usage[e] += mask.sum().float()
            self.total_tokens += batch_size * seq_len
        
        # Flatten for easier processing
        flat_x = x.view(-1, d_model)  # [batch_size * seq_len, d_model]
        flat_expert_indices = expert_indices.view(-1, self.top_k)  # [batch_size * seq_len, top_k]
        flat_expert_weights = expert_weights.view(-1, self.top_k)  # [batch_size * seq_len, top_k]
        
        # Initialize output
        output = torch.zeros_like(flat_x)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (flat_expert_indices == expert_idx)  # [batch_size * seq_len, top_k]
            token_indices, k_indices = torch.where(expert_mask)
            
            if len(token_indices) == 0:
                continue
            
            # Get tokens for this expert
            expert_input = flat_x[token_indices]  # [num_tokens, d_model]
            
            # Process through expert
            expert_output = self.experts[expert_idx](expert_input)  # [num_tokens, d_model]
            
            # Get weights for these tokens
            weights = flat_expert_weights[token_indices, k_indices].unsqueeze(-1)  # [num_tokens, 1]
            
            # Accumulate weighted output
            output.index_add_(0, token_indices, expert_output * weights)
        
        # Reshape output
        output = output.view(batch_size, seq_len, d_model)
        
        # Compute load balancing loss if needed
        load_balance_loss = None
        if self.use_load_balancing and self.training:
            load_balance_loss = self.load_balancer.compute_load_balancing_loss(
                expert_indices, expert_weights
            )
        
        return output, load_balance_loss
    
    def get_expert_usage(self):
        """Get expert usage statistics"""
        if self.total_tokens > 0:
            return (self.expert_usage / self.total_tokens).cpu().numpy()
        return self.expert_usage.cpu().numpy()
    
    def reset_expert_usage(self):
        """Reset expert usage statistics"""
        self.expert_usage.zero_()
        self.total_tokens.zero_()
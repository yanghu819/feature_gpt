import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
import argparse
import os
import logging
from datetime import datetime

# ==============================
# Helper Functions
# ==============================

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return (t, t) if not isinstance(t, tuple) else t

# ==============================
# Neural Memory Components
# ==============================

class MultiheadRMSNorm(nn.Module):
    """Multi-head RMSNorm for neural memory."""
    def __init__(self, dim, heads):
        super().__init__()
        self.rmsnorm = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma = nn.Parameter(torch.zeros(heads, 1, dim))

    def forward(self, x):
        return self.rmsnorm(x) * (self.gamma + 1.)

class BidirectionalNeuralMemory(nn.Module):
    """Neural memory module with bidirectional processing for BERT."""
    def __init__(
        self,
        dim,
        chunk_size=1,
        dim_head=None,
        heads=8,
        pre_rmsnorm=True,
        post_rmsnorm=False,
        qk_rmsnorm=False,
        model=None,
        accept_weight_residual=False,
        momentum=True,
        momentum_order=1,
        init_adaptive_step_bias=-2,
        init_momentum_bias=0.9,
        init_decay_bias=-1.0,
        default_step_transform_max_lr=0.1,
    ):
        super().__init__()
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        
        # Store model configuration
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.store_chunk_size = chunk_size
        self.retrieve_chunk_size = chunk_size
        self.momentum = momentum
        self.momentum_order = momentum_order
        
        # Normalization layers
        self.store_norm = nn.LayerNorm(dim) if pre_rmsnorm else nn.Identity()
        self.retrieve_norm = nn.LayerNorm(dim) if pre_rmsnorm else nn.Identity()
        self.multihead_rmsnorm = MultiheadRMSNorm(dim_head, heads) if post_rmsnorm else nn.Identity()
        
        self.q_norm = MultiheadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()
        self.k_norm = MultiheadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()
        
        # Multi-head projection
        self.to_queries = nn.Linear(dim, inner_dim, bias=False)
        self.to_keys = nn.Linear(dim, inner_dim, bias=False)
        self.to_values = nn.Linear(dim, inner_dim, bias=False)
        
        # Head splitting and merging
        self.split_heads = nn.Sequential(
            nn.Linear(inner_dim, inner_dim, bias=False),
            Rearrange('b n (h d) -> b h n d', h=heads)
        )
        
        self.merge_heads = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(inner_dim, dim, bias=False)
        )
        
        # Memory model (MLP by default)
        self.memory_model = default(model, MemoryMLP(dim_head, depth=2, expansion_factor=4))
        
        # Initialize memory parameters
        self._init_memory_parameters()
        
        # Adaptive learning and momentum
        self.to_adaptive_step = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n 1')
        )
        
        self.to_decay_factor = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n 1')
        )
        
        self.to_momentum = nn.Sequential(
            nn.Linear(dim, heads * momentum_order),
            Rearrange('b n (h o) -> o b h n 1', o=momentum_order)
        ) if momentum else None
        
        # Set initial biases
        if exists(init_adaptive_step_bias):
            nn.init.constant_(self.to_adaptive_step[0].bias, init_adaptive_step_bias)
        
        if exists(init_momentum_bias) and exists(self.to_momentum):
            nn.init.constant_(self.to_momentum[0].bias, init_momentum_bias)
            
        if exists(init_decay_bias):
            nn.init.constant_(self.to_decay_factor[0].bias, init_decay_bias)
            
        # Default learning rate transform
        self.default_step_transform_max_lr = default_step_transform_max_lr
        
        # Weight residual handling
        self.accept_weight_residual = accept_weight_residual
        
        # Register buffer
        self.register_buffer('zero', torch.tensor(0.), persistent=False)
        
        # 添加权重初始化
        self._init_weights()
        
    def _init_memory_parameters(self):
        """Initialize the memory parameters dictionary."""
        memory_params = dict(self.memory_model.named_parameters())
        self.memory_param_names = list(memory_params.keys())
        self.memory_params = nn.ParameterList(memory_params.values())
        
    def init_weights(self, batch_size):
        """Initialize weights for a new batch."""
        weights = {}
        for name, param in zip(self.memory_param_names, self.memory_params):
            weights[name] = param.unsqueeze(0).expand(batch_size * self.heads, *param.shape)
        return weights
    
    def forward(
        self,
        seq,
        store_seq=None,
        state=None,
        prev_weights=None,
    ):
        """
        Bidirectional forward pass for BERT-style processing.
        
        Args:
            seq: Input sequence [batch, seq_len, dim]
            store_seq: Optional sequence to store (defaults to seq)
            state: Previous state (seq_index, weights, cache, states, updates)
            prev_weights: Weights from previous layer
            
        Returns:
            retrieved: The output after memory lookup
            next_state: The updated memory state
        """
        # Handle single token case
        if seq.ndim == 2:
            seq = seq.unsqueeze(1)  # [b, d] -> [b, 1, d]
            
        # Initialize store sequence
        store_seq = default(store_seq, seq)
        batch_size = seq.shape[0]
        
        # Initialize state if not provided
        if not exists(state):
            state = (0, None, None, None, None)
            
        seq_index, weights, cache_store_seq, past_state, updates = state
        
        # Handle cached sequence
        if exists(cache_store_seq):
            store_seq = torch.cat((cache_store_seq, store_seq), dim=1)
            
        # Initialize weights if needed
        if not exists(weights):
            weights = self.init_weights(batch_size)
            
        # Process the whole sequence at once (bidirectional)
        store_seq_len = store_seq.shape[1]
        
        # Update the memory with the full sequence
        updated_weights, next_neural_mem_state = self.store_memories_bidirectional(
            store_seq,
            weights,
            seq_index=seq_index,
            past_state=past_state,
            prev_weights=prev_weights
        )
        
        # Retrieve memories
        retrieved = self.retrieve_memories_bidirectional(
            seq,
            updated_weights
        )
        
        return retrieved, next_neural_mem_state
    
    def store_memories_bidirectional(
        self,
        seq,
        weights=None,
        past_state=None,
        seq_index=0,
        prev_weights=None,
    ):
        """
        Store memories in a bidirectional manner.
        
        Args:
            seq: Input sequence [batch, seq_len, dim]
            weights: Current weights dict
            past_state: Past state tuple
            seq_index: Current sequence index
            prev_weights: Weights from previous layer
            
        Returns:
            updated_weights: The updated weights after processing
            next_state: The next memory state
        """
        batch, seq_len = seq.shape[:2]
        
        # Normalize input
        seq = self.store_norm(seq)
        
        # Project to keys and values
        keys = self.to_keys(seq)
        values = self.to_values(seq)
        
        # Split heads
        keys = self.split_heads(keys)
        values = self.split_heads(values)
        
        # Normalize keys
        keys = self.k_norm(keys)
        
        # Get adaptive learning rates for the sequence
        adaptive_lr = self.to_adaptive_step(seq).sigmoid() * self.default_step_transform_max_lr
        
        # Initialize weights if needed
        if not exists(weights):
            weights = self.init_weights(batch)
            
        # Convert weights to TensorDict-like structure for easier handling
        weights_dict = {k: v.clone() for k, v in weights.items()}
        
        # Compute gradients globally (for all tokens)
        grads = self.compute_global_gradients(keys, values, weights_dict, adaptive_lr)
        
        # Apply updates
        updated_weights = self.apply_global_updates(weights_dict, grads)
        
        # Create next state
        next_state = (updated_weights, None)  # No momentum tracking needed in bidirectional mode
        
        next_store_state = (
            seq_index + seq_len,  # Update sequence index
            updated_weights,      # Updated weights
            None,                 # No cache segment needed
            next_state,           # Next state
            updated_weights       # Memory updates
        )
        
        return updated_weights, next_store_state
    
    def compute_global_gradients(self, keys, values, weights, adaptive_lr):
        """计算全局梯度,添加数值稳定性检查"""
        # 添加输入值范围检查
        if torch.abs(keys).max() > 100 or torch.abs(values).max() > 100:
            keys = F.normalize(keys, dim=-1)
            values = F.normalize(values, dim=-1)
        
        # 检查输入
        if torch.isnan(keys).any() or torch.isnan(values).any():
            print("\n严重错误: 神经记忆输入包含 NaN")
            print("keys NaN 数量:", torch.isnan(keys).sum().item())
            print("values NaN 数量:", torch.isnan(values).sum().item())
            raise ValueError("神经记忆输入包含 NaN,计算终止")
        
        # 准备批处理计算
        batch, heads, seq_len, dim = keys.shape
        
        # 重塑输入张量
        flat_keys = keys.reshape(batch * heads, seq_len, dim)
        flat_values = values.reshape(batch * heads, seq_len, dim)
        
        # 展平权重用于批处理计算
        flat_weights = {}
        for name, w in weights.items():
            if torch.isnan(w).any():
                print(f"\n严重错误: 权重 {name} 包含 NaN")
            flat_weights[name] = w.reshape(batch * heads, *w.shape[1:])
        
        # 计算预测和错误
        preds = {}
        for name, param in flat_weights.items():
            if name == 'weights.0':
                preds[name] = torch.einsum('bsd,bdh->bsh', flat_keys, param)
                preds[name] = torch.clamp(preds[name], -100, 100)  # 限制范围
            elif name == 'weights.1':
                preds[name] = torch.einsum('bsh,bhd->bsd', preds['weights.0'], param)
                preds[name] = torch.clamp(preds[name], -100, 100)  # 限制范围
                
            # 检查预测结果
            if torch.isnan(preds[name]).any():
                print(f"\n严重错误: {name} 的预测包含 NaN")
                print(f"输入形状: keys={flat_keys.shape}, weights={param.shape}")
                print("预测统计:", {
                    "min": preds[name].min().item(),
                    "max": preds[name].max().item(),
                    "mean": preds[name].mean().item()
                })
                raise ValueError(f"{name} 的预测包含 NaN,计算终止")
        
        # 计算预测误差
        errors = {}
        scale_factor = 0.1  # 减小梯度规模
        for name in weights.keys():
            if name == 'weights.0':
                errors[name] = preds[name] * scale_factor
            elif name == 'weights.1':
                errors[name] = (preds[name] - flat_values) * scale_factor
                
            # 检查误差
            if torch.isnan(errors[name]).any():
                print(f"\n严重错误: {name} 的误差包含 NaN")
                print("误差统计:", {
                    "min": errors[name].min().item(),
                    "max": errors[name].max().item(),
                    "mean": errors[name].mean().item()
                })
                raise ValueError(f"{name} 的误差包含 NaN,计算终止")
        
        # 计算梯度
        grads = {}
        flat_lr = adaptive_lr.reshape(batch * heads, seq_len, 1)
        max_grad_norm = 1.0  # 最大梯度范数
        
        for name in weights.keys():
            if name == 'weights.0':
                grad = torch.einsum('bsd,bsh->bdh', flat_keys, errors[name])
                grad = torch.nn.functional.normalize(grad, dim=-1) * max_grad_norm
                grads[name] = grad * flat_lr.mean(dim=1, keepdim=True)
            elif name == 'weights.1':
                grad = torch.einsum('bsh,bsd->bhd', preds['weights.0'], errors[name])
                grad = torch.nn.functional.normalize(grad, dim=-1) * max_grad_norm
                grads[name] = grad * flat_lr.mean(dim=1, keepdim=True)
                
            # 检查梯度
            if torch.isnan(grads[name]).any():
                print(f"\n严重错误: {name} 的梯度包含 NaN")
                print("梯度形状:", grads[name].shape)
                print("梯度统计:", {
                    "min": grads[name].min().item(),
                    "max": grads[name].max().item(),
                    "mean": grads[name].mean().item()
                })
                raise ValueError(f"{name} 的梯度包含 NaN,计算终止")
        
        return grads
    
    def apply_global_updates(self, weights, grads):
        """Apply updates based on global gradients."""
        # 获取批次大小和头数
        batch_size = next(iter(weights.values())).shape[0] // self.heads
        
        # 生成一个临时序列用于计算衰减因子
        token_seq = torch.ones(batch_size, 1, self.dim, device=weights[self.memory_param_names[0]].device)
        
        # 计算衰减因子并调整维度
        decay_factor = self.to_decay_factor(token_seq).sigmoid()  # [batch, heads, 1, 1]
        decay_factor = decay_factor.view(batch_size * self.heads, 1, 1)  # [batch*heads, 1, 1]
        
        # 应用权重衰减和梯度更新
        updated_weights = {}
        for name, weight in weights.items():
            # 计算惊异度（负梯度）
            surprise = -grads[name]
            
            # 应用权重衰减和更新
            updated_weights[name] = (1 - decay_factor) * weight + surprise
            
        return updated_weights
    
    def retrieve_memories_bidirectional(self, seq, weights):
        """
        Retrieve memories in a bidirectional manner.
        
        Args:
            seq: Input sequence [batch, seq_len, dim]
            weights: Current weights dict
            
        Returns:
            values: The retrieved memory values
        """
        batch, seq_len = seq.shape[:2]
        
        # Pre-normalize
        seq = self.retrieve_norm(seq)
        
        # Project to queries
        queries = self.to_queries(seq)
        
        # Split heads
        queries = self.split_heads(queries)
        
        # Query normalization
        queries = self.q_norm(queries)
        
        # Compute memory outputs
        flat_queries = queries.reshape(batch * self.heads, seq_len, self.dim_head)
        
        # 使用权重处理查询
        # 第一层变换
        hidden = torch.bmm(
            flat_queries,  # [B*H, N, D]
            weights['weights.0']  # [B*H, D, H]
        )  # [B*H, N, H]
        
        # 第二层变换
        flat_outputs = torch.bmm(
            hidden,  # [B*H, N, H]
            weights['weights.1']  # [B*H, H, D]
        )  # [B*H, N, D]
        
        # 重塑回多头格式
        values = flat_outputs.reshape(batch, self.heads, seq_len, self.dim_head)
        
        # Normalize multi-head output
        values = self.multihead_rmsnorm(values)
        
        # Merge heads
        values = self.merge_heads(values)
        
        return values

    def _init_weights(self):
        """初始化权重以提高数值稳定性"""
        # 初始化投影矩阵
        nn.init.xavier_uniform_(self.to_queries.weight)
        nn.init.xavier_uniform_(self.to_keys.weight)
        nn.init.xavier_uniform_(self.to_values.weight)
        
        # 初始化分割和合并头的线性层
        for layer in self.split_heads:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                
        for layer in self.merge_heads:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

# ==============================
# Attention Components
# ==============================

class Rearrange(nn.Module):
    """Simple rearrange module."""
    def __init__(self, pattern, **kwargs):
        super().__init__()
        self.pattern = pattern
        self.kwargs = kwargs
        
    def forward(self, x):
        if self.pattern == 'b n (h d) -> b h n d':
            b, n, hd = x.shape
            h = self.kwargs['h']
            d = hd // h
            return x.view(b, n, h, d).transpose(1, 2)
        elif self.pattern == 'b h n d -> b n (h d)':
            b, h, n, d = x.shape
            return x.transpose(1, 2).reshape(b, n, h * d)
        elif self.pattern == 'b n h -> b h n 1':
            b, n, h = x.shape
            # 重排维度并添加最后的单维度
            return x.transpose(1, 2).unsqueeze(-1)
        else:
            raise NotImplementedError(f"Pattern {self.pattern} not implemented")

class BidirectionalAttention(nn.Module):
    """Non-causal attention for BERT-style processing."""
    def __init__(
        self,
        dim,
        segment_len,
        num_persist_mem_tokens=0,
        dim_head=64,
        heads=8,
        dropout=0.1
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        
        self.norm = nn.LayerNorm(dim)
        
        # Rotary positional embedding
        self.pos_emb = RotaryPositionalEmbedding(dim_head)
        
        # Projections
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
        # Persistent memory
        self.num_persist_mem_tokens = num_persist_mem_tokens
        self.persistent_memory = nn.Parameter(torch.zeros(2, heads, num_persist_mem_tokens, dim_head))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        b, n, d = x.shape
        h = self.heads
        
        # Pre-normalization
        x = self.norm(x)
        
        # Project to q, k, v
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)
        
        # Apply rotary positional embeddings
        q = self.pos_emb(q)
        k = self.pos_emb(k)
        
        # Add persistent memory to keys and values
        if self.num_persist_mem_tokens > 0:
            pm_k, pm_v = self.persistent_memory
            pm_k = pm_k.unsqueeze(0).expand(b, -1, -1, -1)
            pm_v = pm_v.unsqueeze(0).expand(b, -1, -1, -1)
            
            k = torch.cat((pm_k, k), dim=2)
            v = torch.cat((pm_v, v), dim=2)
            
            if exists(mask):
                # Extend mask for persistent memory tokens
                pm_mask = torch.ones(b, 1, self.num_persist_mem_tokens, device=mask.device, dtype=mask.dtype)
                mask = torch.cat((pm_mask, mask), dim=2)
        
        # Compute attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply mask if provided (non-causal, typically for padding)
        if exists(mask):
            mask = mask.unsqueeze(1)  # [b, 1, n]
            dots = dots.masked_fill(~mask, -1e9)
            
        # Attention weights
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        # Compute output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(out)
        
        return out

class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding implementation."""
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        
    def forward(self, x):
        device = x.device
        
        t = torch.arange(x.shape[2], device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # 只生成一半维度的嵌入，因为我们会分别应用到x的前后两半
        emb = torch.cat((freqs, freqs), dim=-1).to(device)
        
        # 确保cos和sin的维度与输入的一半维度匹配
        cos = emb[:, :self.dim//2].cos().view(1, 1, -1, self.dim//2)
        sin = emb[:, :self.dim//2].sin().view(1, 1, -1, self.dim//2)
        
        # 分割输入用于旋转
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        
        # 应用旋转嵌入
        return torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)
    
    def _rotate_half(self, x, cos, sin):
        # 这个方法在当前实现中未使用，可以移除或保留以备将来使用
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        return torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)

# ==============================
# Feed Forward Components
# ==============================

class FeedForward(nn.Module):
    """Simple feed forward network with GELU activation."""
    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

# ==============================
# Memory Models
# ==============================

class MemoryMLP(nn.Module):
    """Memory MLP as described in the Titans paper."""
    def __init__(self, dim, depth=2, expansion_factor=4.0):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        dims = [dim] + [hidden_dim] * (depth - 1) + [dim]
        
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(dim_in, dim_out) / math.sqrt(dim_in))
            for dim_in, dim_out in zip(dims[:-1], dims[1:])
        ])
        
    def forward(self, x):
        for i, weight in enumerate(self.weights):
            # Apply non-linearity except for the first layer
            if i > 0:
                x = F.gelu(x)
            x = x @ weight
        return x

# ==============================
# Main BidirectionalTitans Model
# ==============================

class BidirectionalTitans(nn.Module):
    """
    BidirectionalTitans model for BERT-style masked language modeling.
    
    Args:
        num_tokens: Number of tokens in vocabulary
        dim: Model dimension
        depth: Number of layers
        segment_len: Size of each segment
        neural_memory_segment_len: Size of neural memory segments
        heads: Number of attention heads
        dim_head: Dimension per head
        memory_expansion_factor: Expansion factor for memory MLP
        memory_depth: Depth of memory MLP
        mlm_probability: Probability for masked language modeling
        mask_token_id: Token ID for [MASK]
    """
    def __init__(
        self,
        num_tokens,
        dim=768,
        depth=12,
        segment_len=512,
        neural_memory_segment_len=None,
        heads=12,
        dim_head=64,
        ff_mult=4,
        num_persist_mem_tokens=0,
        dropout=0.1,
        memory_expansion_factor=4.0,
        memory_depth=2,
        mlm_probability=0.15,
        mask_token_id=103
    ):
        super().__init__()
        
        # Config
        self.dim = dim
        self.mask_token_id = mask_token_id
        self.mlm_probability = mlm_probability
        
        # Token and positional embeddings
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, segment_len, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        
        # Type embeddings (for sentence A/B)
        self.type_emb = nn.Embedding(2, dim)
        
        # Embedding dropout
        self.emb_dropout = nn.Dropout(dropout)
        
        # Neural memory config
        neural_memory_segment_len = default(neural_memory_segment_len, segment_len)
        
        # Create memory model
        memory_model = MemoryMLP(
            dim=dim_head,
            depth=memory_depth,
            expansion_factor=memory_expansion_factor
        )
        
        # Layers
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            # Bidirectional attention
            attn = BidirectionalAttention(
                dim=dim,
                segment_len=segment_len,
                heads=heads,
                dim_head=dim_head,
                num_persist_mem_tokens=num_persist_mem_tokens,
                dropout=dropout
            )
            
            # Neural memory
            mem = BidirectionalNeuralMemory(
                dim=dim,
                chunk_size=neural_memory_segment_len,
                heads=heads,
                dim_head=dim_head,
                model=memory_model,
                momentum=True,
                momentum_order=1
            )
            
            # Feedforward network
            ff = FeedForward(dim, expansion_factor=ff_mult, dropout=dropout)
            
            # Add to layers
            self.layers.append(nn.ModuleList([mem, attn, ff]))
        
        # Final normalization and output projection
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        
        # Pooler for the [CLS] token
        self.pooler = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh()
        )
        
    def forward(
        self,
        x,
        token_type_ids=None,
        mask=None,
        return_mlm_loss=False,
        return_pooled=False
    ):
        """
        Forward pass through BidirectionalTitans.
        
        Args:
            x: Input token IDs [batch, seq_len]
            token_type_ids: Optional segment IDs for sentence A/B
            mask: Attention mask (1 for tokens to attend to, 0 for padding)
            return_mlm_loss: Whether to return MLM loss
            return_pooled: Whether to return pooled [CLS] representation
            
        Returns:
            Depending on arguments:
                - Token representations
                - MLM loss
                - Pooled representation
        """
        b, n = x.shape
        device = x.device
        
        # Store original input for MLM target
        if return_mlm_loss:
            input_ids = x.clone()
            
            # Create masking pattern
            rand = torch.rand(x.shape, device=device)
            mask_indices = (rand < self.mlm_probability) & (x != 0)  # Don't mask padding
            
            # Replace with [MASK] token
            x = x.clone()
            x[mask_indices] = self.mask_token_id
        
        # Token embeddings
        x = self.token_emb(x)
        
        # Add position embeddings (truncate if sequence is longer than trained)
        position_emb = self.pos_emb[:, :n]
        x = x + position_emb
        
        # Add token type embeddings if provided
        if exists(token_type_ids):
            x = x + self.type_emb(token_type_ids)
            
        # Embedding dropout
        x = self.emb_dropout(x)
        
        # Process through layers
        neural_mem_state = None
        
        for mem, attn, ff in self.layers:
            # Apply neural memory
            mem_out, neural_mem_state = mem(x, state=neural_mem_state)
            x = x + mem_out
            
            # Apply attention
            x = x + attn(x, mask=mask)
            
            # Apply feed forward
            x = x + ff(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Return pooled representation if requested
        if return_pooled:
            # Use first token ([CLS]) for pooled representation
            pooled = self.pooler(x[:, 0])
            
        # Calculate MLM loss if requested
        if return_mlm_loss:
            logits = self.to_logits(x)
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                input_ids.view(-1),
                ignore_index=0  # Ignore padding
            )
            
            if return_pooled:
                return loss, pooled
            return loss
            
        if return_pooled:
            return x, pooled
            
        # By default, return full sequence representation
        return x
        
    def predict_masked(self, x, mask_indices=None, token_type_ids=None, attention_mask=None):
        """Run forward pass and return predictions for masked tokens."""
        if mask_indices is None:
            # Find all mask tokens
            mask_indices = (x == self.mask_token_id)
            
        # Forward pass
        with torch.no_grad():
            outputs = self(x, token_type_ids=token_type_ids, mask=attention_mask)
            logits = self.to_logits(outputs)
            
        # Get predictions only for masked positions
        predictions = logits[mask_indices].argmax(dim=-1)
        return predictions

# ==============================
# Dataset
# ==============================

class WikiTextDataset(Dataset):
    """WikiText dataset for masked language modeling."""
    def __init__(self, file_path, tokenizer, seq_len=512):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Load text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize text
        tokens = tokenizer.encode(text)
        
        # Create examples
        self.examples = []
        for i in range(0, len(tokens) - seq_len, seq_len // 2):  # 50% overlap
            self.examples.append(tokens[i:i + seq_len])
            
        # Pad last example if needed
        if len(self.examples[-1]) < seq_len:
            self.examples[-1] = self.examples[-1] + [0] * (seq_len - len(self.examples[-1]))
            
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx])

# ==============================
# Tokenizer (Simple)
# ==============================

class SimpleTokenizer:
    """Simple tokenizer for WikiText."""
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.word_to_idx = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 103}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.next_idx = len(self.word_to_idx)
        
    def train(self, file_path):
        """Build vocabulary from file."""
        word_counts = {}
        
        # Read file and count words
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                words = line.split()
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add to vocabulary (up to vocab_size)
        for word, _ in words:
            if self.next_idx >= self.vocab_size:
                break
                
            if word not in self.word_to_idx:
                self.word_to_idx[word] = self.next_idx
                self.idx_to_word[self.next_idx] = word
                self.next_idx += 1
                
    def encode(self, text):
        """Encode text to token IDs."""
        tokens = []
        words = text.split()
        
        # Add [CLS] at the beginning
        tokens.append(self.word_to_idx["[CLS]"])
        
        # Encode words
        for word in words:
            if word in self.word_to_idx:
                tokens.append(self.word_to_idx[word])
            else:
                tokens.append(self.word_to_idx["[UNK]"])
                
        return tokens
    
    def decode(self, token_ids):
        """Decode token IDs to text."""
        return " ".join([self.idx_to_word.get(idx, "[UNK]") for idx in token_ids])
        
    def __len__(self):
        return len(self.word_to_idx)

# ==============================
# Training Functions
# ==============================

def setup_logging(output_dir, use_mem):
    """设置日志记录"""
    # 创建logs目录
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mem_status = 'with_mem' if use_mem else 'no_mem'
    log_file = log_dir / f'training_{mem_status}_{timestamp}.log'
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger()

def train_epoch(model, dataloader, optimizer, scheduler, device, logger, max_grad_norm=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    
    for i, batch in progress_bar:
        # 检查输入数据
        if torch.isnan(batch).any():
            logger.error("输入数据包含 NaN")
            logger.error(f"批次索引: {i}")
            raise ValueError("输入数据包含 NaN,训练终止")
            
        batch = batch.to(device)
        
        try:
            loss = model(batch, return_mlm_loss=True)
        except Exception as e:
            logger.error("前向传播过程中出现异常")
            logger.error(f"批次索引: {i}")
            logger.error(f"异常信息: {str(e)}")
            raise e
        
        if torch.isnan(loss):
            logger.error("损失函数为 NaN")
            logger.error(f"批次索引: {i}")
            raise ValueError("损失函数为 NaN,训练终止")
            
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_description(f"Loss: {loss.item():.4f}")
        
        # 每100步记录一次详细loss
        if i % 100 == 0:
            logger.info(f"Step {i}, Loss: {loss.item():.4f}")
        
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Average training loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, dataloader, device, logger):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            loss = model(batch, return_mlm_loss=True)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Validation loss: {avg_loss:.4f}")
    return avg_loss

# ==============================
# Main Training Loop
# ==============================

def train_bidirectional_titans(
    train_file,
    val_file,
    model_dir='models',
    vocab_size=30000,
    dim=768,
    depth=6,
    heads=12,
    batch_size=8,
    seq_len=512,
    lr=2e-5,
    epochs=3,
    warmup_steps=10000,
    weight_decay=0.01,
    device='cuda'
):
    """Main training function."""
    # 检查是否使用神经记忆
    use_mem = os.environ.get('USE_MEM', '').lower() in ('true', '1', 't')
    
    # 设置日志
    logger = setup_logging(model_dir, use_mem)
    logger.info(f"使用神经记忆: {use_mem}")
    logger.info(f"开始训练 BidirectionalTitans 模型")
    logger.info(f"模型配置: dim={dim}, depth={depth}, heads={heads}")
    
    # Create directories
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer and datasets
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    logger.info("训练分词器...")
    tokenizer.train(train_file)
    
    logger.info("创建数据集...")
    train_dataset = WikiTextDataset(train_file, tokenizer, seq_len=seq_len)
    val_dataset = WikiTextDataset(val_file, tokenizer, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    logger.info("初始化模型...")
    model = BidirectionalTitans(
        num_tokens=len(tokenizer),
        dim=dim,
        depth=depth,
        heads=heads,
        segment_len=seq_len,
        neural_memory_segment_len=seq_len//8 if use_mem else None,  # 根据USE_MEM决定是否使用神经记忆
        mlm_probability=0.15,
        mask_token_id=tokenizer.word_to_idx["[MASK]"]
    )
    
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training loop
    best_val_loss = float('inf')
    logger.info(f"开始训练,共 {epochs} 轮...")
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, logger)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device, logger)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = model_dir / f"best_model_{'with_mem' if use_mem else 'no_mem'}.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f"保存新的最佳模型,验证损失: {val_loss:.4f}")
            
        # Save checkpoint
        checkpoint_path = model_dir / f"checkpoint_epoch_{epoch+1}_{'with_mem' if use_mem else 'no_mem'}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, checkpoint_path)
        
    logger.info("训练完成!")
    return model, tokenizer

# ==============================
# Running the script
# ==============================

def main(args):
    # 确保数据目录存在
    data_dir = Path("data/wikitext-103")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 检查数据文件是否存在,如果不存在则下载
    if not data_dir.exists() or not (data_dir / "wiki.train.tokens").exists():
        print("数据集未找到,正在下载 WikiText...")
        dataset = load_dataset("wikitext", "wikitext-103-v1")
        
        # 保存数据集到文件
        splits = {
            "train": "wiki.train.tokens",
            "validation": "wiki.valid.tokens",
            "test": "wiki.test.tokens"
        }
        
        for split, filename in splits.items():
            output_file = data_dir / filename
            with open(output_file, "w", encoding="utf-8") as f:
                for item in dataset[split]:
                    f.write(item["text"] + "\n")
        
        print(f"数据集已下载到 {data_dir}")
    
    # 更新文件路径
    if args.train_file is None:
        args.train_file = str(data_dir / "wiki.train.tokens")
    if args.val_file is None:
        args.val_file = str(data_dir / "wiki.valid.tokens")

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Train model
    model, tokenizer = train_bidirectional_titans(
        train_file=args.train_file,
        val_file=args.val_file,
        model_dir=args.output_dir,
        vocab_size=args.vocab_size,
        dim=args.dim,
        depth=args.num_layers,
        heads=args.num_heads,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.learning_rate,
        epochs=args.epochs,
        device=device
    )
    
    # Save tokenizer vocabulary
    tokenizer_path = Path(args.output_dir) / "tokenizer.pt"
    torch.save({
        'word_to_idx': tokenizer.word_to_idx,
        'idx_to_word': tokenizer.idx_to_word
    }, tokenizer_path)
    
    print(f"Training complete. Model and tokenizer saved to {args.output_dir}")
    
    # Test model with a sample input
    test_text = "The capital of France is [MASK]."
    tokens = tokenizer.encode(test_text)
    input_tensor = torch.tensor([tokens]).to(device)
    
    predictions = model.predict_masked(input_tensor)
    predicted_word = tokenizer.idx_to_word[predictions[0].item()]
    
    print(f"Sample prediction: '{test_text}' -> '{predicted_word}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BidirectionalTitans model")
    
    # Dataset arguments
    parser.add_argument("--train_file", type=str, default=None, help="Path to training file")
    parser.add_argument("--val_file", type=str, default=None, help="Path to validation file")
    
    # Model arguments
    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size")
    parser.add_argument("--dim", type=int, default=384, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--output_dir", type=str, default="models/bidirectional_titans", help="Output directory")
    
    args = parser.parse_args()
    main(args)



# python run.py --dim 384 --num_layers 6 --num_heads 6 --batch_size 8 --epochs 3

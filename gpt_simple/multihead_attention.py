import math

import torch
from torch import Block, nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        assert embed_dim % self.num_heads == 0, "invalid heads and embedding dimension configuration"
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.proj_dropout = nn.Dropout(config.ff_dropout)
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.max_len, config.max_len))
            .unsqueeze(0).unsqueeze(0)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        # x.shape == (batch_size, seq_len, embed_dim)
        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape == (batch_size, num_heads, seq_len, head_dim)
        
        attn = torch.matmul(q, k_t) / math.sqrt(q.size(-1))
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        mask = self.mask[:, :, :seq_len, :seq_len]
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.attn_dropout(attn)
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        y = torch.matmul(attn, v)
        # y.shape == (batch_size, num_heads, seq_len, head_dim)
        y = y.transpose(1, 2)
        # y.shape == (batch_size, seq_len, num_heads, head_dim)
        y = y.reshape(batch_size, seq_len, -1)
        # y.shape == (batch_size, seq_len, embed_dim)
        y = self.proj_dropout(self.proj(y))
        return y
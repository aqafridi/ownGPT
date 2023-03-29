import math

import torch
from torch import  nn
import torch.nn.functional as F
from block import Block


class GPT(nn.Module):

    def __init__(self,config=None):
        super().__init__()
        embed_dim =config.embed_dim
        self.max_len=config.max_len
        self.tok_embed = nn.Embedding(
            config.vocab_size, embed_dim
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_len, embed_dim)
        )
        self.dropout = nn.Dropout(config.embed_dropout)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.num_blocks)]
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, config.vocab_size)
    def forward(self, x, target=None):
        # batch_size = x.size(0)
        seq_len = x.size(1)
        assert seq_len <= self.max_len, "sequence longer than model capacity"
        
        tok_embedding = self.tok_embed(x)
        # tok_embedding.shape == (batch_size, seq_len, embed_dim)
        pos_embedding = self.pos_embed[:, :seq_len, :]
        # pos_embedding.shape == (1, seq_len, embed_dim)
        x = self.dropout(tok_embedding + pos_embedding)
        x = self.blocks(x)
        x = self.ln(x)
        x = self.fc(x)
        # x.shape == (batch_size, seq_len, vocab_size)
        return x


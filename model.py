# This implementation of a encoder transformer model is based on the paper "Attention is all you need" by Vaswani et al.
#   and heavily inspired by Andrej Karpathy's implementation of the same paper:
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from tqdm import tqdm
from typing import Optional


@dataclass
class Config:
    vocab_size: int
    n_classes: int
    batch_size: int
    context_length: int
    d_model: int
    n_head: int
    n_layer: int
    dropout: float
    device: torch.device


def init_(m: nn.Module) -> None:
    """Cannot remember where I found this init function, maybe nanoGPT?"""
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.05)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.05)


class SelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super(SelfAttention, self).__init__()
        self.config = config
        
        ## head_size = n_embd // n_head
        ## single Head, q,k,v = nn.Linear(n_embd, head_size, bias=False)
        ## 3 * n_embd = |{q,k,v}| * (n_embd * head_size) * n_head
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T,C = x.size()
        n_head = self.config.n_head
        q, k, v = self.c_attn(x).split(self.config.d_model, dim=2) # nh: num head, hs: head size
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
        
        y: torch.Tensor = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.config.dropout, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y)) ## (B, T, C)
        return y
        

class FeedForward(nn.Module):
    """a simple Linear Layer followed by a non-linearity"""
    def __init__(self, config: Config):
        super(FeedForward, self).__init__()
        self.lin1 = nn.Linear(config.d_model, 4 * config.d_model) ## the 4* is from the paper
        self.lin2 = nn.Linear(4 * config.d_model, config.d_model) ## projection layer going back into residual pathway
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.gelu(x)
        x = self.lin2(x)
        return self.dropout(x)


class Block(nn.Module):
    """Transformer Bock: comm followed by computation"""
    
    def __init__(self, config: Config):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(config.d_model) # normalise features? at initialisation
        self.csa = SelfAttention(config) ## communication
        self.ln2 = nn.LayerNorm(config.d_model) # B,T dims are considered both batch layers
        self.ffwd = FeedForward(config)                 ## computation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.csa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super(Transformer, self).__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.tke = nn.Embedding(config.vocab_size, config.d_model)
        self.pse = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        self.ln_f = nn.LayerNorm(config.d_model) ## std layer norm before final projection
        self.lm_head = nn.Linear(config.d_model, config.n_classes)
        self.apply(init_) ## init weights
        
        self.config = config
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B,T = idx.shape
        
        tok_emb = self.tke(idx) # B,T,C
        pos_emb = self.pse(torch.arange(T, device=self.config.device)) # T,C
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x) # B, T, C
        x = x[:, 0, :].squeeze(1) # B, C ##  only the first token is used for classification
        logits: torch.Tensor = self.lm_head(x) # (B, n_classes)

        if targets is None:
            return logits, None
        
        # B, C = logits.shape
        # logits = logits.view(B*T, C)
        # targets = targets.view(B*T) ## targets.view(-1)
        
        loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return logits.argmax(dim=-1).view(-1)


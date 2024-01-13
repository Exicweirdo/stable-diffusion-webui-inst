### Attention model
import numpy as np
from torch import nn
from modules.attention import CrossAttention
from transformers import CLIPVisionModelWithProjection, AutoProcessor
from modules import shared, devices


class Attentions(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim))

    def forward(self, x, context=None):
        x_1 = self.attn1(x)
        x_2 = self.attn2(x_1, x)
        x_3 = self.net(x_2)
        return x_3
    
class FrozenCLIPVisionencoder:
    def __init__(self, version = "openai/clip-vit-large-patch14", device="cuda", max_length=77):
        self.processor = AutoProcessor.from_pretrained(version)
        self.encoder = CLIPVisionModelWithProjection.from_pretrained(version).eval().requires_grad_(False).to(device)
        self.device = device
        self.max_length = max_length
        self.encoder.to(device)
        self.freeze()
    
    def freeze(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def encode(self, images):
        self.processor()
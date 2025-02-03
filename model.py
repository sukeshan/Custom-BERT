import math

import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        """Cross entropy loss function."""
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, logits, target):
        return self.loss(logits, target)
    
class Embedding(nn.Module):

    """Embedding layer that converts token IDs to vectors."""
    
    def __init__(self, vocab_size, emb_dim):
        super().__init__() 
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
   
    """Add positional encoding to token embeddings."""

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # Efficient broadcasting
        return self.dropout(x)

class MultiHeadAttention(nn.Module):

    """Multi-head attention layer integrated with dropout , layer normalization and Flash Attention-I."""

    def __init__(self, emb_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__() 
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads # 64
        
        # Linear projections without bias (standard practice)
        self.qkv = nn.Linear(emb_dim, 3*emb_dim, bias=False)  # (Emb_dim, 3*Emb_dim)
        self.out = nn.Linear(emb_dim, emb_dim, bias=False) # (Emb_dim, Emb_dim)
        self.dropout = dropout  # Store dropout probability

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # (Batch_size, Seq_len, 3* Seq_len) --> [(Batch_size, Seq_len,Emb_dim),(Batch_size, Seq_len,Emb_dim),(Batch_size, Seq_len,Emb_dim)]
        qkv = self.qkv(x).chunk(3, dim=-1)
        #(Batch_size, Seq_len, Num_heads, Head_dim) --> (Batch_size, Num_heads, Seq_len, Head_dim)
        q, k, v = [t.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2) 
                  for t in qkv]
        
        # Efficient attention calculation (Flash Attention - I)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0
        )
        
        # (Batch_size, Num_heads, Seq_len, Head_dim) --> (Batch_size, Seq_len, Num_heads, Head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        #(Batch_size, Seq_len, Num_heads, Head_dim) --> (Batch_size, Seq_len, Emb_dim)
        attn_output = attn_output.view(batch_size, -1, self.emb_dim)

        return self.out(attn_output)

class FeedForward(nn.Module):

    """Feed-forward neural network."""

    def __init__(self, emb_dim: int = 768, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.GELU(),  
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class LayerNormalizer(nn.Module):

    """Layer normalization layer."""

    def __init__(self, emb_dim:int = 768, eps:float = 1e-6):
        
        super.__init__()
        self.alpha = nn.Parameter(torch.ones(emb_dim))
        self.bias =  nn.Parameter(torch.ones(emb_dim))
        self.eps = eps

    def forward(self, x:torch.Tensor= None):

        # X_Dim : (Batch_size, Seq_len, Emb_dim)
        mean = x.mean(dim = -1, keep_dim = True)
        std = x.std(dim = -1, keep_dim = True)
        return self.alpha * ((x-mean)/ (std +self.eps)) + self.bias

class EncoderBlock(nn.Module):

    """Transformer encoder block."""

    def __init__(self, emb_dim: int = 768, num_heads: int = 12, 
                 ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(emb_dim, num_heads, dropout)
        self.ffn = FeedForward(emb_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Pre-norm architecture (more modern than original paper)
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x
    
class TransformerClassifier(nn.Module):
    
    """Classifier layer that uses the CLS token embedding."""
    
    def __init__(self, num_classes: int = 0,emb_dim: int = 768 ):
        super().__init__()
        self.ln = nn.Linear(emb_dim, num_classes)

    def forward(self,x):
        cls_token = x[:,0,:]
        return self.ln(cls_token)

class CustomBert(nn.Module):

    """Transformer Encoder based on the paper 'Attention is All You Need'."""

    def __init__(self, num_blocks: int = 6, num_heads: int = 12, 
                 vocab_size: int = 30522, emb_dim: int = 768, 
                 ff_dim: int = 2048, max_seq_len: int = 512, 
                 dropout: float = 0.1, num_classes: int = 0):
        super().__init__()
        self.embedding = Embedding(vocab_size, emb_dim)
        self.pos_encoding = PositionalEncoding(emb_dim, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(emb_dim, num_heads, ff_dim, dropout)
            for _ in range(num_blocks)
        ])
        # (Batch_size, Seq_len, Emb_dim) --> (Batch_size, Class_size)
        self.classifier = TransformerClassifier(num_classes, emb_dim )

        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, mask, target):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x,mask)
        classification_output = self.classifier(x)
        
        return {"logits" :self.norm(x), 
                "cls_token_logits" :classification_output,
                "loss" : CrossEntropyLoss()(classification_output, target)} 
    


    
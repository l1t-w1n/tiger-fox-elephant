import torch 
import torch.nn as nn
import torch.nn.functional as F

from attention import SelfAttention 

class CLIP_Embedding(nn.Module):
    def __init__(self, vocabulary_size: int = 49408, embedding_dim: int = 768, sequence_length: int = 77):
        super(CLIP_Embedding, self).__init__()
        
        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim
        )
        
        # positional_embeddings - will be learned during training as usual weights
        self.positional_embeddings = nn.Parameter(
            torch.zeros((sequence_length, embedding_dim))
        )
        
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        # text_tokens = (batch_size, sequence_length)
        
        # (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dim)
        text_tokens = self.token_embeddingz(text_tokens)
        
        # (batch_size, sequence_length, embedding_dim) + (sequence_length, embedding_dim) -> (batch_size, sequence_length, embedding_dim)
        text_tokens = text_tokens + self.positional_embeddings
        return text_tokens

class CLIP_Layer(nn.Module):
    def __init__(self, num_heads: int = 12, embedding_dim: int = 768):
        super(CLIP_Layer, self).__init__()
        
        self.layer_normalization_1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.attention = SelfAttention(
            num_heads=num_heads,
            embedding_dim=embedding_dim
        )
        
        self.layer_normalization_2 = nn.LayerNorm(normalized_shape=embedding_dim)
        
        self.mlp_1 = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim * 4
        )
        
        self.mlp_2 = nn.Linear(
            in_features=embedding_dim * 4,
            out_features=embedding_dim
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (batch_size, sequence_length, embedding_dim)
        
        residual = x
        
        x = self.layer_normalizxation_1(x)
        
        x = self.attention(x, causal_mask=True)
        
        x += residual
        
        residual = x
        
        x = self.layer_normalization_2(x)
        
        x = self.mlp_1(x)
        
        x = x * torch.sigmoid(1.702 * x)   # QuickGELU activation function
        
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.linear_2(x)
        
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residual

        return x


class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        
        # embedding exists for convert text data representation into high-dimensional vector space
        self.embedding = CLIP_Embedding(
            vocabulary_size=49408,
            embedding_dim=768,
            sequence_length=77
        )
        
        self.layers = nn.ModuleList([
            CLIP_Layer(
                num_heads=12,
                embedding_dim=768
            )
            for _ in range(12)
        ])
        
        self.layer_normalization = nn.LayerNorm(normalized_shape=768)
    
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        # text_tokens = (batch_size, sequence_length)
        
        text_tokens = text_tokens.type(torch.long)
        
        # (batch, sequence_length) -> (batch, sequence_length, embedding_dim)
        x = self.embedding(text_tokens)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.layer_normalization(x)
        return x 
import torch 
import torch.nn as nn 
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int, input_bias: bool = True, output_bias: bool = True):
        super(SelfAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.embedding_dim = embedding_dim
        
        self.output_bias = output_bias
        self.input_bias = input_bias
        
        self.Q_W = nn.Linear(
            in_features=embedding_dim, 
            out_features=embedding_dim, 
            bias=input_bias
        )
        
        self.K_W = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=input_bias
        )
        
        self.V_W = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=input_bias
        )
        
        self.O_W = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=output_bias
        )
        
    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        # x = (batch_size, sequential_length, embedding_dim) == (batch_size, height * width, channels)
        # causal_mask - use for causal attention mask (to avoid looking into the future)
        
        input_shape = x.shape 
        batch_size, sequence_length, embedding_dim = x.shape 
        
        intemidiate_dim = (batch_size, sequence_length, self.num_heads, self.head_dim)
        
        q = self.Q_W(x)
        k = self.K_W(x)
        v = self.V_W(x)
        
        # (batch_size, sequence_length, embedding_dim) -> (batch_size, sequence_length, num_heads, head_dim) -> (batch_size, num_heads, sequence_length, head_dim) 
        q = q.view(*intemidiate_dim).transpose(1, 2)
        k = k.view(*intemidiate_dim).transpose(1, 2)
        v = v.view(*intemidiate_dim).transpose(1, 2)
        
        # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = torch.matmul(q, k.transpose(-1, -2))
        
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(diagonal=1)
            weight.masked_fill__(mask, -float('inf'))
        
        # Divide by d_k (Dim / H). 
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = weight / torch.sqrt(self.head_dim)
        
        # Softmax - assigns a probability to each token which i will interpret as a attention weight between other tokens
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1)
        
        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        x = torch.matmul(weight, v)
        
        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        x = x.transpose(1, 2).contiguous()
        
        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        x = x.view(*input_shape)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.O_W(x)
        
        # (Batch_Size, Seq_Len, Dim)
        return x 
        
        
class CrossAttention(nn.Module): 
    def __init__(self, num_heads: int, embedding_dim: int, cross_embedding_dim: int, input_bias: bool = True, output_bias: bool = True):
        super(CrossAttention, self).__init__()
        
        self.number_of_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.embedding_dim = embedding_dim
        self.cross_embedding_dim = cross_embedding_dim
        
        self.input_bias = input_bias 
        self.output_bias = output_bias 
        
        self.Q_W = nn.Linear(
            in_fgeatures=embedding_dim,
            out_features=embedding_dim,
            bias=input_bias
        )
        
        self.K_W = nn.Linear(
            in_features=cross_embedding_dim,
            out_features=embedding_dim,
            bias=input_bias
        )
        
        self.V_W = nn.Linear(
            in_features=cross_embedding_dim,
            out_features=embedding_dim,
            bias=input_bias
        )
        
        self.O_W = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            bias=output_bias
        )
        
    def forward(self, image: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        # image = (batch_size, sequential_length, embedding_dim) == (batch_size, height * width, channels)
        # prompt = (batch_size, prompt_length, cross_embedding_dim)
        
        input_shape = image.shape 
        batch_size, sequence_length, embedding_dim = image.shape
        intermidiate_shape = (batch_size, sequence_length, self.number_of_heads, self.head_dim)
        
        q = self.Q_W(image)
        k = self.K_W(prompt)
        v = self.V_W(prompt)
        
        q = q.view(*intermidiate_shape).transpose(1, 2)
        k = k.view(*intermidiate_shape).transpose(1, 2)
        v = v.view(*intermidiate_shape).transpose(1, 2)
        
        weight = torch.matmul(q, k.transpose(-1, -2))
        weight = weight / torch.sqrt(self.head_dim)
        
        weight = F.softmax(weight, dim=-1)
        
        x = torch.matmul(weight, v)
        x = x.trasponse(1, 2).contiguous()
        x = x.view(*intermidiate_shape)
        x = self.O_W(x)
        
        return x
        
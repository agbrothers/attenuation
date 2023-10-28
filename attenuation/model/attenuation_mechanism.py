import math
import torch
import torch.nn as nn



def normal_init(module, mean=0.0, std=0.02):
    nn.init.normal_(module.weight, mean=mean, std=std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class MultiHeadAttention(nn.Module):

    def __init__(
        self, 
        query_dim, 
        context_dim, 
        hidden_dim, 
        num_heads, 
        drop_attn=0, 
        drop_out=0
    ):
        super().__init__()

        self.Q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.KV = nn.Linear(context_dim, 2*hidden_dim, bias = False)
        self.out = nn.Linear(hidden_dim, context_dim, bias=False)
        self.scale = 1 / math.sqrt(hidden_dim)
        self.drop_attn = nn.Dropout(drop_attn)
        self.drop_out = nn.Dropout(drop_out)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_dim = hidden_dim // num_heads
        self.attn_weights = None

        normal_init(self.Q, mean=0.0, std=0.02)
        normal_init(self.KV, mean=0.0, std=0.02)
        normal_init(self.out, mean=0.0, std=0.02)

    def forward(self, query, context, mask=None):
        ## Cross Attention if query != context
        ## Self Attention  if query == context
        ## query, context ~ [ batch, seq_len, hidden_dim ]

        ## GET INPUT DIMENSIONS
        b = query.shape[0] # batch size
        h = self.num_heads
        a = self.attn_dim
        q_len = query.shape[1]
        c_len = context.shape[1]

        ## PROJECT INPUTS
        q   = self.Q(query)
        k,v = self.KV(context).chunk(2, dim=-1)

        ## SPLIT ATTENTION HEADS
        q = q.view(b, q_len, h, a)
        k = k.view(b, c_len, h, a)
        v = v.view(b, c_len, h, a)

        ## APPLY MASK(s) TO KEYS PER HEAD HERE
        if mask is not None:
            if mask.dtype != bool: mask = mask.bool()
            mask = mask.unsqueeze(1)

        ## COMPUTE ATTENTION WEIGHTS
        attn_weights = torch.einsum("bqha,bkha->bqkh", (q, k))
        if mask is not None: attn_weights = self.apply_mask(attn_weights, mask, -1e5)
        attn_weights = torch.softmax(attn_weights * self.scale, dim=2)
        attn_weights = self.drop_attn(attn_weights)
        self.attn_weights = attn_weights

        ## APPLY ATTENTION WEIGHTS TO VALUES
        attn = torch.einsum("bqkh,bvha->bqha", (attn_weights, v)).contiguous() 
        attn = attn.view_as(query)
        output = self.drop_out(self.out(attn))
        return output

    def apply_mask(self, x, mask, fill_value=-1e5):
        ## ASSUMPTIONS
        ## • True values in mask are replaced with fill_value
        ## • False values in mask are left as is

        ## BOOLEAN MASK
        x = x * (mask==False) 
        ## FILL VALUE
        return x + (mask * fill_value)



class AttenuationMechanism(nn.Module):

    def __init__(
        self,
        query_dim,
        context_dim,
        hidden_dim,
        num_heads,
        dropout,
        num_embeddings=12,
    ):
        super().__init__()
        self.device = torch.device(
            type = 'cuda' if torch.cuda.is_available() else 'cpu',
            index=0 if torch.cuda.is_available() else None
        )
        self.attn = MultiHeadAttention(query_dim, context_dim, hidden_dim, num_heads, dropout)
        self.function_embedder = nn.Embedding(num_embeddings=12, embedding_dim=query_dim)
        self.to(self.device)

    def forward(self, vector_sets, aggregation_functions):
        ## input_set: Tensor - (batch, vectors, hidden_dim)
        ## function : int - integer mapping to an embedding representing an aggregation function
        function_embeddings = self.function_embedder(aggregation_functions.long())
        return self.attn(query=function_embeddings, context=vector_sets)


if __name__ == '__main__':

    model = AttenuationMechanism(
        query_dim=32,
        context_dim=3,
        hidden_dim=32,
        num_heads=1,
        dropout=0,
        num_embeddings=4,
    )

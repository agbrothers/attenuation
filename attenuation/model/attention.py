import math
import torch
import torch.nn as nn

from attenuation.model.initializers import normal_init


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads, dropout_w=0, dropout_e=0):
        super().__init__()

        self.Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.KV = nn.Linear(hidden_dim, 2*hidden_dim, bias = False)
        # self.K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_dim = hidden_dim // num_heads
        self.scale = 1 / math.sqrt(hidden_dim)
        self.dropout_w = nn.Dropout(dropout_w)
        self.dropout_e = nn.Dropout(dropout_e)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        normal_init(self.Q)
        # normal_init(self.K)
        # normal_init(self.V)
        normal_init(self.KV)
        normal_init(self.out)

    def forward(self, query, context, mask=None):

        ## PROJECT INPUTS
        q = self.Q(query)
        # k = self.K(context)
        # v = self.V(context)
        k,v = self.KV(context).chunk(2, dim=-1)

        ## SPLIT ATTENTION HEADS
        b = query.size(0) # Assume [batch, seq_len, hidden]
        q = q.view(b, -1, self.num_heads, self.attn_dim)
        k = k.view(b, -1, self.num_heads, self.attn_dim)
        v = v.view(b, -1, self.num_heads, self.attn_dim)

        ## COMPUTE ATTENTION WEIGHTS
        dot_product = torch.einsum("bqha,bkha->bqkh", (q, k))
        dot_product = self.apply_mask(dot_product, mask)
        w = torch.softmax(dot_product * self.scale, dim=2)
        w = self.dropout_w(w)

        ## APPLY ATTENTION WEIGHTS TO VALUES
        e = torch.einsum("bqvh,bvha->bqha", (w, v)).contiguous() 
        e = e.view_as(query) 
        return self.dropout_e(self.out(e))

    def apply_mask(self, x, mask):
        if mask is None:
            return x
        if len(mask.shape) == 2: 
            mask = mask.unsqueeze(2)
        return x + mask

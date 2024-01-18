import math
import torch
import torch.nn as nn

from attenuation.model.initializers import normal_init


def generalized_mean(x, p):
    n = len(x)
    return (1/n * torch.sum(torch.pow(x, p))).pow(1/p)

def geometric_mean(x):
    n = len(x)
    return torch.prod(x) ** (1/n)

def harmonic_mean(x):
    n = len(x)
    return n / torch.sum(1/x)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class GeMAttention(nn.Module):

    def __init__(
        self, 
        hidden_dim, 
        num_heads, 
        dropout_w=0,
        dropout_e=0,
        shift=5,
        p_min=1e-4,
        p_max=1e+4,
        v_min=1e-10,
    ):
        super().__init__()
        self.Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn_dim = hidden_dim // num_heads
        self.scale = 1 / math.sqrt(hidden_dim)
        self.dropout_w = nn.Dropout(dropout_w)
        self.dropout_e = nn.Dropout(dropout_e)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        ## GeM ATTENTION PARAMETERS
        self.shift = nn.Parameter(torch.ones((hidden_dim,)) * shift)
        self.p = nn.Parameter(torch.normal(mean=1, std=0.02, size=(hidden_dim,)))
        self.p_max = p_max
        self.p_min = p_min
        self.v_min = v_min
        ## TODO: TRY LEARNING log_p instead of p
        # self.log_p = nn.Parameter(torch.rand((hidden_dim,))) 
        # self.p = nn.Parameter(torch.rand((hidden_dim,)))

        normal_init(self.Q)
        normal_init(self.K)
        normal_init(self.V)
        normal_init(self.out) 
        #xavier_normal_

    def forward(self, query, context, mask=None):

        ## PROJECT INPUTS
        q = self.Q(query)
        k = self.K(context)
        v = self.V(context)
        
        ## SPLIT ATTENTION HEADS
        b = query.size(0) # Assume [batch, seq_len, hidden]
        q = q.view(b, -1, self.num_heads, self.attn_dim)
        k = k.view(b, -1, self.num_heads, self.attn_dim)

        ## COMPUTE ATTENTION WEIGHTS
        dot_product = torch.einsum("bqha,bkha->bqkh", (q, k))
        dot_product = self.apply_mask(dot_product, mask)
        w = torch.softmax(dot_product * self.scale, dim=2)
        ## NOTE: No w dropout, the torch implementation breaks the sum(w)=1 assumption

        ## CLAMP AND SHIFT p,v TO PREVENT GeM DISCONTINUITIES
        # p = (self.p.sign()+0.5).sign() * torch.clamp(self.p.abs(), min=self.p_min, max=self.p_max)
        # p = (self.p.sign()+0.5).sign() * torch.clamp(symexp(self.p).abs(), min=self.p_min, max=self.p_max)
        p = (self.p.sign()+0.5).sign() * torch.clamp(self.p.abs(), min=self.p_min, max=self.p_max)
        v = torch.clamp(torch.abs(v + self.shift), min=self.v_min)
        
        ## RAISE v TO p VIA THE LOG-SUM-EXP TRICK 
        z = p * torch.log(v)
        z_max = z.max(dim=1)[0].unsqueeze(1) 
        # z_max = z.min(dim=1)[0].unsqueeze(1) 
        v = torch.exp(z - z_max) 

        ## APPLY ATTENTION WEIGHTS TO VALUES
        v = v.view(b, -1, self.num_heads, self.attn_dim)
        mean = torch.einsum("bqvh,bvha->bqha", (w, v)).contiguous() 
        mean = mean.view_as(query) 

        ## RAISE mean TO 1/p IN LOG SCALE AND SHIFT BACK
        e = torch.exp((z_max + torch.log(mean)) / p)
        e = e - self.shift
        return self.dropout_e(self.out(e))

    def apply_mask(self, x, mask):
        if mask is None:
            return x
        if len(mask.shape) == 2: 
            mask = mask.unsqueeze(2)
        return x + mask


if __name__ == "__main__":

    b = 32   # batch
    s = 128  # sequence len
    d = 512  # hidden dim
    h = 8    # heads

    batch = torch.rand((b, s, d)) * 2 - 1
    gem = GeMAttention(hidden_dim=d, num_heads=h)
    emb = gem(batch, batch)
    batch = batch + emb
    
    assert not torch.any(torch.isnan(emb)), "Nans in GeM embeddings"

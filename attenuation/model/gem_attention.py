import math
import torch
import torch.nn as nn

from attenuation.experiments.dataset import generate_test, generate_means


def identity_init(module, bias=0):
    nn.init.eye_(module.weight)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)

def normal_init(module, bias=0):
    std = 1/math.sqrt(module.weight.shape[1])
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x))
    # return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x):
    return torch.sign(x) * torch.exp(torch.abs(x))
    # return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def donut(x, min=1e-2, max=1e+6):
    ## PREVENT x FROM TAKING VALUES WITHIN min OF 0 OR GREATER THAN max
    return torch.sign(x + min/2) * torch.clamp(torch.abs(x), min=min, max=max)

def geometric_mean(x):
    n = len(x)
    return torch.prod(x) ** (1/n)

def harmonic_mean(x):
    n = len(x)
    return n / torch.sum(1/x)

def generalized_mean(x, p):
    n = len(x)
    eps = 1e-10
    return (torch.sum(torch.pow(-x-1, p)) / n).pow(1/p)
    # return torch.exp(torch.log((torch.sum(torch.exp(p * torch.log(-x))) / n)) / p)
    # return symexp(symlog((torch.sum(symexp(p * symlog(x+eps))) / n) ) / p)


class GeMAttention(nn.Module):

    def __init__(
        self, 
        query_dim, 
        context_dim, 
        hidden_dim, 
        num_heads, 
        drop_attn=0, 
        drop_out=0,
    ):
        super().__init__()

        self.Q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.K = nn.Linear(context_dim, hidden_dim, bias=False)
        self.V = nn.Linear(context_dim, hidden_dim, bias=True)
        self.out = nn.Linear(hidden_dim, context_dim, bias=True)
        self.scale = 1 / math.sqrt(hidden_dim)
        self.drop_attn = nn.Dropout(drop_attn)
        self.drop_out = nn.Dropout(drop_out)
        self.log_p = nn.Parameter(torch.rand((context_dim,))) ## UNIQUE AGGEGATION PER FEATURE
        # self.p = nn.Parameter(torch.rand((context_dim,))) ## UNIQUE AGGEGATION PER FEATURE
        # self.p = nn.Parameter(torch.Tensor((1, 0, -1, 1e+3, -1e+3)))
        self.eps = 1e-10

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_dim = hidden_dim // num_heads
        self.attn_weights = None

        # identity_init(self.Q)
        # identity_init(self.K)
        # identity_init(self.V)
        # identity_init(self.out)
        normal_init(self.Q)
        normal_init(self.K)
        normal_init(self.V)
        normal_init(self.out)
        with torch.no_grad(): self.V.bias[:] = 5
        with torch.no_grad(): self.out.bias[:] = -5


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
        q = self.Q(query)
        k = self.K(context)
        v = self.V(context)
        
        ## POWER SCALING FOR GENERALIZED MEAN
        # p = donut(self.p)   ## PREVENT P FROM BEING EXACTLY 0
        p = donut(torch.exp(self.log_p))   ## PREVENT P FROM BEING EXACTLY 0
        ## IF p < 1 -> Take abs value of v??
        v = torch.abs(v) + self.eps        ## ENFORCE v in positive real numbers
        # v = torch.clamp(v, min=self.eps) ## ENFORCE v in positive real numbers
        # v = torch.relu(v) + self.eps     ## ENFORCE v in positive real numbers
        
        ## LOG SUM EXP OVERFLOW TRICK 
        ## v = torch.pow(v, p)
        ## v = torch.exp(p * torch.log(v)) ## RAISE v TO p
        z = p * torch.log(v)  ## COMPUTE POWER ALONG CONTEXT DIMENSION
        z_max = z.max(dim=1)[0].unsqueeze(1) ## TAKE MAX ALONG CONTEXT DIMENSION
        v = torch.exp(z - z_max) ## RAISE v TO p

        ## TODO: When p is small, why does z get small for that feature but NOT v after?
        ## something happening with the max subtraction exponentiation?

        ## SPLIT ATTENTION HEADS
        q = q.view(b, q_len, h, a)
        k = k.view(b, c_len, h, a)
        v = v.view(b, c_len, h, a)

        ## COMPUTE ATTENTION WEIGHTS
        attn_weights = torch.einsum("bqha,bkha->bqkh", (q, k))
        if mask is not None: attn_weights = self.apply_mask(attn_weights, mask)
        attn_weights = torch.softmax(attn_weights * self.scale, dim=2)
        ## TODO: Make not to remove dropout attention
        ##       It violates the property of weights summing to 1 
        ##       which is essential for preventing explosion
        # attn_weights = self.drop_attn(attn_weights)

        ## APPLY ATTENTION WEIGHTS TO VALUES
        attn = torch.einsum("bqvh,bvha->bqha", (attn_weights, v)).contiguous() 
        attn = attn.view_as(query) 
        ## POWER SCALING FOR GENERALIZED MEAN
        ## mean = torch.pow(attn, 1/p)
        ## mean = torch.exp(torch.log(attn) / p)
        # logsumexp = z_max + torch.log(attn)
        # mean = torch.exp(logsumexp / p)
        mean = torch.exp((z_max + torch.log(attn)) / p)
        
        output = self.drop_out(self.out(mean))
        assert not torch.any(torch.isnan(output)), "Nans in GeM output!"
        # assert not torch.any(output > 1e+3), "GeM output too large!"
        return output

    def apply_mask(self, x, mask):
        if len(mask.shape) == 2: 
            mask = mask.unsqueeze(2)
        return x + mask


class AttenuationMechanism(nn.Module):

    def __init__(
        self,
        query_dim,
        context_dim,
        hidden_dim,
        num_heads,
        dropout,
        num_embeddings=3,
    ):
        super().__init__()
        self.device = torch.device(
            type = 'cuda' if torch.cuda.is_available() else 'cpu',
            index=0 if torch.cuda.is_available() else None
        )
        ## TODO: SET UP EACH HEAD TO COMPUTE A DIFFERENT AGGREGATION:
        ## TODO: FIGURE OUT HOW TO SELECT THE AGGREGATION? OR HAVE THE OUTPUT BE ALL AGGREGTAIONS?

        ## SET ATTN KEY/QUERY/VALUE MATRICES TO THE IDENTITY
        self.attn = GeMAttention(query_dim, context_dim, hidden_dim, num_heads, dropout)
        self.function_embedder = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=query_dim)

        ## SET INITIAL WEIGHTS MANUALLY:
        with torch.no_grad():
            ## 0. MAX -> query 
            self.function_embedder.weight[0,:] =  0.0
            ## 1. MIN -> query 
            self.function_embedder.weight[1,:] =  0.0
            ## 2. AVG -> query 
            self.function_embedder.weight[2,:] =  0.0
            ## 3. SUM?

            ## 4. MAX NORM? -> One head? Use the query matrix to pull out the norm

        ## INITIALIZE THE WEIGHTS RANDOMLY
        # identity_init(self.Q)
        # identity_init(self.K)
        # identity_init(self.V)
        # identity_init(self.out)

        self.to(self.device)

    def forward(self, vector_sets, aggregation_functions):
        ## input_set: Tensor - (batch, vectors, hidden_dim)
        ## function : int - integer mapping to an embedding representing an aggregation function
        function_embeddings = self.function_embedder(aggregation_functions.long())
        return self.attn(query=function_embeddings, context=vector_sets)


if __name__ == '__main__':

    model = AttenuationMechanism(
        query_dim=5,
        context_dim=5,
        hidden_dim=5,
        num_heads=1,
        dropout=0,
        num_embeddings=4,
    )
    features, functions, targets = generate_means(device=model.device)
    preds = model(features, functions).squeeze(1)
    loss = torch.mean(torch.square(preds - targets))
    print(loss)

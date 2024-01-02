import math
import torch
import torch.nn as nn

from attenuation.experiments.test_batches import generate_test, generate_means


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

def void(x, epsilon=1e-6):
    ## PREVENT x FROM TAKING VALUES WITHIN epsilon OF 0
    return torch.sign(x + epsilon/2) * torch.clamp(torch.abs(x), min=epsilon)


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads, drop_attn=0, drop_out=0):
        super().__init__()

        self.Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.KV = nn.Linear(hidden_dim, 2*hidden_dim, bias = False)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = 1 / math.sqrt(hidden_dim)
        self.drop_attn = nn.Dropout(drop_attn)
        self.drop_out = nn.Dropout(drop_out)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_dim = hidden_dim // num_heads
        self.attn_weights = None

        normal_init(self.Q)
        normal_init(self.KV)
        normal_init(self.out)

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

        ## COMPUTE ATTENTION WEIGHTS
        attn_weights = torch.einsum("bqha,bkha->bqkh", (q, k))
        if mask is not None: attn_weights = self.apply_mask(attn_weights, mask)
        attn_weights = torch.softmax(attn_weights * self.scale, dim=2)
        # attn_weights = self.drop_attn(attn_weights)
        self.attn_weights = attn_weights

        ## APPLY ATTENTION WEIGHTS TO VALUES
        attn = torch.einsum("bqvh,bvha->bqha", (attn_weights, v)).contiguous() 
        attn = attn.view_as(query)
        output = self.drop_out(self.out(attn))
        return output

    def apply_mask(self, x, mask):
        if len(mask.shape) == 2: 
            mask = mask.unsqueeze(2)
        return x + mask

    def get_weights(self):
        return self.attn_weights


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
        self.attn = MultiHeadAttention(query_dim, context_dim, hidden_dim, num_heads, dropout)
        self.function_embedder = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=query_dim)

        ## SET INITIAL WEIGHTS MANUALLY:
        with torch.no_grad():
            ## 0. MAX -> query 
            self.function_embedder.weight[0,:] =  1e+3
            ## 1. MIN -> query 
            self.function_embedder.weight[1,:] = -1e+3
            ## 2. AVG -> query 
            self.function_embedder.weight[2,:] =   0.0
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
        query_dim=3,
        context_dim=3,
        hidden_dim=3,
        num_heads=3,
        dropout=0,
        num_embeddings=4,
    )
    features, functions, targets = generate_means(device=model.device)
    preds = model(features, functions).squeeze(1)
    loss = torch.mean(torch.square(preds - targets))
    print(loss)

    features, functions, targets = generate_test(device=model.device)
    preds = model(features, functions).squeeze(1)
    loss = torch.mean(torch.square(preds - targets))
    print(loss)

    
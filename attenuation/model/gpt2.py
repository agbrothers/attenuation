import math
import torch
import torch.nn as nn
from torch import Tensor
from copy import deepcopy

from attenuation.model.gem_attention import GeMAttention, normal_init
from attenuation.model.attention import MultiHeadAttention
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, filter_size)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(filter_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)

        normal_init(self.linear1)
        normal_init(self.linear2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class DecoderLayer(nn.Module):

    def __init__(
            self, 
            query_dim,
            context_dim,
            hidden_dim, 
            num_heads, 
            feedforward_size, 
            dropout,
        ):
        super().__init__()
        self.hidden_size = hidden_dim
        # self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout, dropout)
        self.attn = GeMAttention(query_dim=query_dim, context_dim=context_dim, hidden_dim=hidden_dim, num_heads=num_heads, drop_attn=dropout, drop_out=dropout)
        self.ff = FeedForward(hidden_dim, feedforward_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn_weights = None

    def forward(self, query, context=None, mask=None):  
        if context is None: context = query.clone()
        
        ## MULTIHEADED ATTENTION + SKIP CONNECTION
        skip = query.clone()
        query = self.norm1(query)
        context = self.norm1(context)
        residual = self.attn(query, context, mask)
        x = skip + residual

        ## FEED FORWARD NETWORK + SKIP CONNECTION   
        skip = x.clone()
        residual = self.ff(self.norm2(x))
        return skip + residual


class Decoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        nn.Module.__init__(self)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(encoder_layer.hidden_size)
        self.layers = nn.ModuleList([deepcopy(encoder_layer) for _ in range(num_layers)])
    
    def forward(self, context, mask) -> Tensor:
        for layer in self.layers:
            context = layer(context, mask=mask)
        return self.norm(context)


class GPT2(nn.Module):

    def __init__(
        self,
        query_dim=768,
        context_dim=768,
        hidden_dim=768,
        num_heads=12,
        num_layers=12,
        num_vocab=0,
        dropout=0,
        context_len=1024,
    ):
        nn.Module.__init__(self)
        self.device = torch.device(
            type = 'cuda' if torch.cuda.is_available() else 'cpu',
            index=0 if torch.cuda.is_available() else None
        )

        ## PARSE MODEL PARAMETERS
        self._query_dim = query_dim
        self._hidden_dim = hidden_dim
        self._context_dim = context_dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._feedforward_size = 4*hidden_dim

        ## INITIALIE LAYERS
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        self.embedder = nn.Embedding(num_embeddings=num_vocab, embedding_dim=hidden_dim)
        transformer_layer = DecoderLayer(query_dim, context_dim, hidden_dim, num_heads, 4*hidden_dim, dropout)
        self.decoder = Decoder(transformer_layer, num_layers)
        self.output = nn.Linear(hidden_dim, num_vocab)
        self.to(self.device)

    def forward(self, tokens, mask=None):
        ## CAUSAL MASK
        if mask is None:
            ## Generate a square causal mask for the sequence. The masked positions 
            ## are filled with float('-inf'). Unmasked positions are filled with float(0.0).
            mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(1), tokens.device)
        
        ## LOOKUP TOKEN EMBEDDINGS
        embeddings = self.embedder(tokens)
        ## ADD POSITIONAL ENCODINGS
        embeddings = self.pos_encoder(embeddings)
        ## STEP MODEL
        h = self.decoder(embeddings, mask)
        ## RETURN LOGITS
        return self.output(h)

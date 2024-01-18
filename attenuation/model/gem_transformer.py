import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.activation as A
from torch import Tensor
from torch.nn import (
    Transformer, 
    TransformerEncoderLayer, 
    TransformerEncoder, 
    TransformerDecoderLayer, 
    TransformerDecoder, 
    LayerNorm, 
    MultiheadAttention
)
from typing import Optional, Tuple
# from torchvision.models.vision_transformer import VisionTransformer

from attenuation.model.functional import gem_multi_head_attention_forward


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class EncoderLayerGeM(TransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = F.relu, layer_norm_eps: float = 0.00001, batch_first: bool = False, norm_first: bool = False, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias, device, dtype)
        ## OVERWRITE SELF ATTENTION CLASS
        self.self_attn = AttentionGeM(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)


class DecoderLayerGeM(TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = F.relu, layer_norm_eps: float = 0.00001, batch_first: bool = False, norm_first: bool = False, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias, device, dtype)
        ## OVERWRITE ATTENTION CLASSES
        self.self_attn = AttentionGeM(d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias)
        self.multihead_attn = AttentionGeM(d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias)


class AttentionGeM(MultiheadAttention):
    
    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None, shift=5) -> None:
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)
        self.p = nn.Parameter(torch.normal(mean=1, std=0.02, size=(embed_dim,)))
        # self.p = nn.Parameter(torch.normal(mean=0, std=1, size=(embed_dim,)))
        self.shift = nn.Parameter(torch.ones((embed_dim,)) * shift) ## SHIFT VALUES PRIOR TO GENERALIZED MEAN

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and float masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Set ``need_weights=False`` to use the optimized ``scaled_dot_product_attention``
            and achieve the best performance for MHA.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
            If both attn_mask and key_padding_mask are supplied, their types should match.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)
        is_causal: If specified, applies a causal mask as attention mask.
            Default: ``False``.
            Warning:
            ``is_causal`` provides a hint that ``attn_mask`` is the
            causal mask. Providing incorrect hints can result in
            incorrect execution, including forward and backward
            compatibility.

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """

        why_not_fast_path = ''
        if ((attn_mask is not None and torch.is_floating_point(attn_mask))
           or (key_padding_mask is not None) and torch.is_floating_point(key_padding_mask)):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )


        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif A._is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(A._check_arg_device(x) for x in tensor_args):
                why_not_fast_path = ("some Tensor argument's device is neither one of "
                                     f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}")
            elif torch.is_grad_enabled() and any(A._arg_requires_grad(x) for x in tensor_args):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        self.in_proj_weight,
                        self.in_proj_bias,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = gem_multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads, self.p, self.shift,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        else:
            attn_output, attn_output_weights = gem_multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads, self.p, self.shift,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 batch_first=True,
                 norm_first=True,
                 activation="gelu",
                 gem=False,
        ):
        super(Seq2SeqTransformer, self).__init__()
        if gem:
            ## GeM TRANSFORMER
            encoder_norm = LayerNorm(emb_size)
            decoder_norm = LayerNorm(emb_size)    
            encoder_layer = EncoderLayerGeM(emb_size, nhead, dim_feedforward, dropout, activation, batch_first=batch_first, norm_first=norm_first)
            decoder_layer = DecoderLayerGeM(emb_size, nhead, dim_feedforward, dropout, activation, batch_first=batch_first, norm_first=norm_first)
            gem_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
            gem_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
            self.transformer = Transformer(
                d_model=emb_size, 
                nhead=nhead, 
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                activation=activation,
                custom_encoder=gem_encoder,
                custom_decoder=gem_decoder,
                batch_first=batch_first,
                norm_first=norm_first,        
            )        
        else:
            ## DEFAULT TRANSFORMER
            self.transformer = Transformer(
                d_model=emb_size, 
                nhead=nhead, 
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                activation=activation,
                batch_first=batch_first,
                norm_first=norm_first,
            )

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


if __name__ == "__main__":


    d_model = 512
    num_encoder_layers = 6
    num_decoder_layers = 6
    nhead = 8
    dropout = 0.1
    dim_feedforward = 2048
    activation = "gelu"
    seq_len = 128
    bsz = 32

    ## DEFAULT TRANSFORMER
    t = Transformer(
        d_model=d_model, 
        nhead=nhead, 
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers, 
        dim_feedforward=dim_feedforward, 
        dropout=dropout, 
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )

    src = torch.rand((bsz, seq_len, d_model))
    tgt = torch.rand((bsz, seq_len, d_model))
    output = t(src=src, tgt=tgt)

    ## GEM TRANSFORMER

    ## GeM ENCODER
    encoder_layer = EncoderLayerGeM(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True, norm_first=True)
    encoder_norm = LayerNorm(d_model)
    gem_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    ## GeM DECODER
    decoder_layer = DecoderLayerGeM(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True, norm_first=True)
    decoder_norm = LayerNorm(d_model)    
    gem_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

    gem = Transformer(
        d_model=d_model, 
        nhead=nhead, 
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers, 
        dim_feedforward=dim_feedforward, 
        dropout=dropout, 
        activation="gelu",
        custom_encoder=gem_encoder,
        custom_decoder=gem_decoder,
        batch_first=True,
        norm_first=True,        
    )

    src = torch.rand((bsz, seq_len, d_model))
    tgt = torch.rand((bsz, seq_len, d_model))
    output = gem(src=src, tgt=tgt)


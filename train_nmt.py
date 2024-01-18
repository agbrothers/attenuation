import os
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from timeit import default_timer as timer
from typing import Iterable, List
from tqdm import tqdm
import numpy as np
import random 

from attenuation.model.gem_transformer import Seq2SeqTransformer
from attenuation.model.viz import plot, boxplot, Animation, update_boxplot

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# We need to modify the URLs for the dataset since the links to the original dataset are broken
# Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Place-holders
token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = Multi30k(root='/home/brothag1/code/', split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = Multi30k(root='/home/brothag1/code/', split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # for src, tgt in tqdm(train_dataloader):
    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = Multi30k(root='/home/brothag1/code/', split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")




if __name__ == "__main__":

    # d_model = 512
    # num_encoder_layers = 6
    # num_decoder_layers = 6
    # nhead = 8
    # dropout = 0.1
    # dim_feedforward = 2048
    # activation = "gelu"
    # seq_len = 128
    # bsz = 32

    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    NUM_EPOCHS = 32 #64
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 4 * EMB_SIZE
    BATCH_SIZE = 512
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    LR = 1e-2 #1e-2
    GEM = True

    # NUM_EPOCHS = 32
    # SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    # TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    # EMB_SIZE = 768 # 512
    # NHEAD = 12 # 8
    # FFN_HID_DIM = 4 * EMB_SIZE
    # BATCH_SIZE = 384 # 128
    # NUM_ENCODER_LAYERS = 12 # 6
    # NUM_DECODER_LAYERS = 12 # 6
    # LR = 1e-4
    # GEM = True

    ## MODEL PATH
    experiment_name = f"WMT16_Transformer_lr-2{'_GeM' if GEM else ''}"
    MODEL_PATH = f"attenuation/experiments/results/{experiment_name}/best_model.pth"
    if not os.path.exists(os.path.dirname(MODEL_PATH)):            
        os.makedirs(os.path.dirname(MODEL_PATH))


    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                    NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM,
                                    batch_first=False, gem=GEM)

    transformer.to(DEVICE)
    # transformer = nn.DataParallel(transformer, dim=1)
    # transformer.to(DEVICE)    
    if os.path.exists(MODEL_PATH):
        transformer.load_state_dict(torch.load(MODEL_PATH))
    else:
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    
    ## SET UP RENDERER
    anim_p = Animation(
        title=f"{experiment_name} Learned P", 
        filepath=f"attenuation/experiments/results/{experiment_name}/p.mp4", 
        update_func=update_boxplot, 
        ylim=(-5, 5),
    )

    min_val_loss = torch.inf
    train_history, val_history = [], []
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer)

        ## SAVE BEST MODEL WEIGHTS
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(transformer.state_dict(), MODEL_PATH)

        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        train_history.append(train_loss)
        val_history.append(val_loss)

        ## UPDATE ANIMATION
        if GEM:
            ## PLOT LEARNED P PARAMETERS BY LAYER
            p_list = [layer.self_attn.p.to("cpu").detach() for layer in transformer.transformer.encoder.layers]
            labels = [f"enc_layer_{i}" for i in range(NUM_ENCODER_LAYERS)]
            p_list += [layer.self_attn.p.to("cpu").detach() for layer in transformer.transformer.decoder.layers]
            labels += [f"dec_sa_layer_{i}" for i in range(NUM_DECODER_LAYERS)]
            p_list += [layer.multihead_attn.p.to("cpu").detach() for layer in transformer.transformer.decoder.layers]
            labels += [f"dec_ca_layer_{i}" for i in range(NUM_DECODER_LAYERS)]        
            anim_p(data=p_list, labels=labels)
    
    ## RELOAD BEST MODEL
    transformer.load_state_dict(torch.load(MODEL_PATH))
    print(translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu ."))
    print(f"Best Val Loss: {min_val_loss}, Best Val PPL: {math.exp(min_val_loss)}, ")
    
    ## PLOT LOSS CURVES
    plot(train_history, val_history, title=f"{experiment_name} Validation Loss", filepath=f"attenuation/experiments/results/{experiment_name}/loss.png", step_size=1)

    if GEM:
        ## PLOT LEARNED P PARAMETERS BY LAYER
        p_list = [layer.self_attn.p.to("cpu").detach() for layer in transformer.transformer.encoder.layers]
        labels = [f"enc_layer_{i}" for i in range(NUM_ENCODER_LAYERS)]
        p_list += [layer.self_attn.p.to("cpu").detach() for layer in transformer.transformer.decoder.layers]
        labels += [f"dec_sa_layer_{i}" for i in range(NUM_DECODER_LAYERS)]
        p_list += [layer.multihead_attn.p.to("cpu").detach() for layer in transformer.transformer.decoder.layers]
        labels += [f"dec_ca_layer_{i}" for i in range(NUM_DECODER_LAYERS)]
        boxplot(p_list, title=f"{experiment_name} Learned P", filepath=f"attenuation/experiments/results/{experiment_name}/p.png", labels=labels)

        ## PLOT LEARNED SHIFT PARAMETERS BY LAYER
        shift_list = [layer.self_attn.shift.to("cpu").detach() for layer in transformer.transformer.encoder.layers]
        shift_list += [layer.self_attn.shift.to("cpu").detach() for layer in transformer.transformer.decoder.layers]
        shift_list += [layer.multihead_attn.shift.to("cpu").detach() for layer in transformer.transformer.decoder.layers]
        boxplot(shift_list, title=f"{experiment_name} Learned Shift", filepath=f"attenuation/experiments/results/{experiment_name}/shift.png", labels=labels)


    # from torcheval.metrics.functional.text import bleu_score
    # candidates = ["the squirrel is eating the nut"]
    # references = [["a squirrel is eating a nut", "the squirrel is eating a tasty nut"]]
    # bleu_score(candidates, references, n_gram=4)

    # candidates = ["the squirrel is eating the nut", "the cat is on the mat"]
    # references = [["a squirrel is eating a nut", "the squirrel is eating a tasty nut"], ["there is a cat on the mat", "a cat is on the mat"]]
    # bleu_score(candidates, references, n_gram=4)


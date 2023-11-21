import os
import time
import math
from typing import Tuple
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
from torch.utils.data import dataset
from torchtext.datasets import WikiText2, WikiText103
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
# from transformers import GPT2Config
# print(GPT2Config())


from attenuation.model.gpt2 import GPT2


# bptt = 4
bptt = 32

def train(model: nn.Module, data, optimizer, criterion, ntokens, seq_len) -> None:
    model.train()  # turn on train mode
    epoch_loss = 0.

    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = data.shape[-1]

    for i in tqdm(range(0, num_batches - 1, seq_len), desc='Training: '):
        optimizer.zero_grad()

        batch, targets = get_batch(data, i, seq_len)
        preds = model(batch)
        preds_flat = preds.view(-1, ntokens)
        loss = criterion(preds_flat, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.12)
        optimizer.step()

        epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches

def evaluate(model: nn.Module, data: Tensor, seq_len) -> float:
    model.eval()  # turn on evaluation mode
    epoch_loss = 0.

    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = data.shape[-1]
    
    with torch.no_grad():
        for i in tqdm(range(0, num_batches - 1, seq_len),  desc='Evaluating: '):
            batch, targets = get_batch(data, i, seq_len)
            preds = model(batch)
            preds_flat = preds.view(-1, ntokens)
            loss = criterion(preds_flat, targets)
            # total_loss += loss.item()
            # total_loss += seq_len * criterion(output_flat, targets).item()
            epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: Tensor, bsz: int, device="cpu") -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    num_batches = data.size(0) // bsz
    data = data[:num_batches * bsz]
    data = data.view(bsz, num_batches).contiguous()
    return data.to(device)

    
def get_batch(source: Tensor, i: int, seq_len) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(seq_len, source.size(1) - 1 - i)
    data = source[..., i:i+seq_len]
    target = source[..., i+1:i+1+seq_len].reshape(-1)
    return data, target


def plot_loss(train_history, val_history, filepath):
    ## PLOT LOSS
    x = torch.arange(len(train_history))
    fig = plt.figure(figsize=(4, 4), dpi=400)
    ax = fig.add_subplot(111)
    ax.plot(x, train_history, alpha=0.7, lw=0.4, label='Training Loss')
    ax.plot(x, val_history, alpha=0.7, lw=0.4, label='Validation Loss')
    ax.legend(loc=1, prop={'size': 4})
    ax.tick_params(axis='both', which='major', labelsize=4)
    ax.tick_params(axis='both', which='minor', labelsize=4)
    plt.title("GeM GPT-2 WikiText2", fontsize=6, fontweight="bold")
    plt.xlabel("Epoch", fontsize=5)
    plt.ylabel("Cross Entropy Loss", fontsize=5)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return



if __name__ == "__main__":

    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])


    # ``train_iter`` was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    ## INITIALIZE MODEL
    ntokens = len(vocab)  # size of vocabulary
    # model = GPT2(
    #     query_dim=8,
    #     context_dim=8,
    #     hidden_dim=8,
    #     num_heads=1,
    #     num_layers=4,
    #     num_vocab=ntokens,
    #     dropout=0.1,
    #     context_len=1024,
    # )
    model = GPT2(
        query_dim=768,
        context_dim=768,
        hidden_dim=768,
        num_heads=12,
        num_layers=12,
        num_vocab=ntokens,
        dropout=0.1,
        context_len=1024,
    )

    ## MULTI-GPU
    device = model.device
    model = nn.DataParallel(model)
    model.to(device)

    epochs = 100
    seq_len = bptt
    # batch_size = 1
    batch_size = 512
    eval_batch_size = 32
    train_data = batchify(train_data, batch_size, device) 
    val_data = batchify(val_data, eval_batch_size, device)
    test_data = batchify(test_data, eval_batch_size, device)


    ## TRAIN MODEL 
    lr = 1e-2 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float('inf')
    with TemporaryDirectory() as tempdir:
        best_model_params_path = "results/best_val_gem_gpt2.pth"
        train_history, val_history = [], []
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train_loss = train(model, train_data, optimizer, criterion, ntokens, seq_len)
            val_loss = evaluate(model, val_data, seq_len)
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s |  train loss {train_loss:5.5f} | '
                f'valid loss {val_loss:5.5f} | val ppl {val_ppl:8.2f} |')
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

            scheduler.step()
            
            ## PLOT LOSS CURVES
            train_history.append(train_loss)
            val_history.append(val_loss)
            plot_loss(train_history, val_history, "results/loss.png")

    ## COMPUTE LOSS ON TEST DATASET
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states
    test_loss = evaluate(model, test_data, seq_len)
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print(f'| End of training |test loss {test_loss:5.4f} | '
        f'test ppl {test_ppl:8.4f}')
    print('=' * 89)




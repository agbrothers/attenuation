import os
import time
import math
import json
import random
import numpy as np
from typing import Tuple
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
# from transformers import GPT2Config
# print(GPT2Config())

from attenuation.model.gpt2 import GPT2
from attenuation.datasets import DATASETS
from attenuation.logger import Logger


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

def evaluate(model: nn.Module, data: Tensor, ntokens, seq_len, criterion) -> float:
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
            epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches


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


def plot(train_history, val_history, filepath, step_size=1):
    ## PLOT LOSS
    x = torch.arange(len(train_history)) * step_size
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
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    plt.savefig(filepath)
    plt.close()
    return


def run_experiment(model_config, training_config, trial:str, log_path="attenuation/experiments/results/log.csv",):

    ## SET SEED FOR REPRODUCIBILITY
    seed = training_config["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    ## INITIALIZE LOGGER
    logger = Logger(
        path=log_path,
        model_config=model_config,
        training_config=training_config,
    )

    ## LOAD DATA
    train_data, val_data, test_data, num_vocab = DATASETS[training_config["dataset_name"]]()
    model_config.update({"num_vocab": num_vocab})

    ## INITIALIZE MODEL
    model = GPT2(**model_config)

    ## MULTI-GPU
    device = model.device
    model = nn.DataParallel(model)
    model.to(device)
    if model_config.get("model_path") is not None:
        model.load_state_dict(torch.load(model_config["model_path"]))

    ## RESHAPE DATA INTO BATCHES
    train_data = batchify(train_data, training_config["batch_size"], device) 
    val_data = batchify(val_data, training_config["eval_batch_size"], device)
    test_data = batchify(test_data, training_config["eval_batch_size"], device)

    ## INITIALIZE LOSS AND OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    ## TRAIN MODEL 
    best_val_loss = float('inf')
    with TemporaryDirectory() as tempdir:
        ## CREATE OUTPUT DIR
        experiment_name = logger.key.replace("_train","")
        best_model_params_path = f"attenuation/experiments/results/{experiment_name}/best_{model_config['model_name']}_{logger.run}.pth"
        if not os.path.exists(os.path.dirname(best_model_params_path)):            
            os.makedirs(os.path.dirname(best_model_params_path))
        
        ## TRAINING EPOCH LOOP
        train_loss_history, val_loss_history, train_ppl_history, val_ppl_history = [], [], [], []
        for epoch in range(1, training_config["epochs"] + 1):
            epoch_start_time = time.time()
            train_loss = train(model, train_data, optimizer, criterion, num_vocab, training_config["seq_len"])
            val_loss = evaluate(model, val_data, num_vocab, training_config["seq_len"], criterion)
            train_ppl = math.exp(train_loss)
            val_ppl = math.exp(val_loss)
            scheduler.step()
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| {trial} | end of epoch {epoch:3d} | time: {elapsed:5.2f}s |  train loss {train_loss:5.5f} | '
                f'valid loss {val_loss:5.5f} | val ppl {val_ppl:8.2f} |')
            print('-' * 89)

            ## SAVE OFF BEST MODEL WEIGHTS
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

            ## PLOT LOSS CURVES
            train_loss_history.append(train_loss)
            train_ppl_history.append(train_ppl)
            val_loss_history.append(val_loss)
            val_ppl_history.append(val_ppl)

    ## COMPUTE LOSS ON TEST DATASET
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states
    test_loss = evaluate(model, test_data, num_vocab, training_config["seq_len"], criterion)
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print(f'| {trial} | End of training |test loss {test_loss:5.4f} | '
        f'test ppl {test_ppl:8.4f}')
    print('=' * 89)
    
    ## RECORD RESULTS
    print(f"LOGGING {experiment_name}")
    step_size = train_data.shape[-1] // training_config["seq_len"]
    logger.log(train_loss_history, val_loss_history, [test_loss])
    print("PLOTTING LOSS")
    plot(train_loss_history, val_loss_history, f"attenuation/experiments/results/{experiment_name}/loss_{logger.run}.png", step_size)
    print("PLOTTING PPL\n")
    plot(train_ppl_history[3:], val_ppl_history[3:], f"attenuation/experiments/results/{experiment_name}/ppl_{logger.run}.png", step_size)




if __name__ == "__main__":

    configs = json.load(open("attenuation/experiments/baseline_configs.json"))

    for i,v in enumerate(configs.values()):
        model_config = v["model_config"]
        training_config = v["training_config"]
        model_config["attn_type"] = "GeM" if "GeM" in model_config["model_name"] else None
        run_experiment(model_config, training_config, trial=f"{i+1}/{len(configs)}")

    # # ## SPECIFY MODEL HYPERPARAMETERS
    # model_config = {
    #     # "model_name": "GPT2_GeM_c32",
    #     # "model_name": "GPT2_GeM_c32_s5",
    #     # "model_name": "GPT2_GeM_c512",
    #     # "model_name": "GPT2_GeM_c512_s5",
    #     "model_name": "GPT2_GeM_c32_h768",
    #     # "model_name": "GPT2_c32",
    #     # "model_name": "GPT2_c32_h768",
    #     # "model_name": "GPT2_c512",
    #     # "model_name": "GPT2_c32_h768",
    #     "model_path": None, # "attenuation/results/old/weights/best_val_gem_gpt2_0.pth"
    #     "context_len": 32, #512,
    #     "hidden_dim": 768,
    #     "num_layers": 12,
    #     "num_heads": 768, #12, #768,
    #     "dropout": 0.1,
    # }
    # # model_config = {
    # #     # "model_name": "GPT2_L_GeM_c32_s5",
    # #     "model_name": "GPT2_L_c32",
    # #     "model_path": None, # "attenuation/results/old/weights/best_val_gem_gpt2_0.pth"
    # #     "context_len": 32, #512,
    # #     "hidden_dim": 1280,
    # #     "num_layers": 36,
    # #     "num_heads": 16, #768,
    # #     "dropout": 0.1,
    # # }
    # model_config["attn_type"] = "GeM" if "GeM" in model_config["model_name"] else None

    # ## SPECIFY TRAINING PARAMETERS
    # training_config = {
    #     "seed": 1,
    #     "lr": 1e-2,
    #     "epochs": 100,
    #     "batch_size": 1024, #2048, #1024, #128 # 512 # 1
    #     "eval_batch_size": 512,
    #     "dataset_name": "WikiText2", 
    #     "seq_len": model_config["context_len"],
    #     "loss_func": "CrossEntropy",
    # }    
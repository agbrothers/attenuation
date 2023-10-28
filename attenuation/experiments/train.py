import os
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR

from attenuation.experiments.dataset import generate_batch
from attenuation.model.attenuation_mechanism import AttenuationMechanism

def train(
    model, 
    num_features,
    dataset_size=128000,
    epochs=1024, 
    batch_size=512, 
    learning_rate=1e-4,
):

    ## PARAMETERS
    batches_per_epoch = dataset_size // batch_size
    train_history = []
    test_history = []

    ## OPTIMIZER
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
    sched = OneCycleLR(
        opt,
        max_lr=learning_rate,
        steps_per_epoch=batches_per_epoch,
        epochs=epochs,
        final_div_factor=1e5,
    )

    for epoch in range(0, epochs):

        train_loss = 0.0
        test_loss  = 0.0

        ## TRAINING LOOP
        for _ in trange(batches_per_epoch):
            
            ## GENERATE TRAINING BATCH ON THE FLY (CHEAP)
            num_items = torch.randint(low=1, high=20)
            features, functions, targets = generate_batch(num_items=num_items, num_features=num_features, device=model.device)

            ## STEP MODEL
            opt.zero_grad()
            preds = model(features, functions)
            
            ## BACKPROPAGATION
            loss = torch.mean(torch.square(preds - targets))            
            loss.backward()
            opt.step()
            sched.step()

            ## UPDATE LOSS HISTORY
            train_loss += loss.item()
        
        train_history.append(train_loss / dataset_size)

        ## TEST LOOP
        for _ in trange(1):
            
            ## GENERATE TEST BATCH ON THE FLY (CHEAP)
            features, functions, targets = generate_batch(batch_size=batch_size, num_features=num_features, device=model.device)

            ## STEP MODEL
            preds = model(features, functions)

            ## UPDATE LOSS HISTORY
            loss = torch.mean(torch.square(preds - targets))            
            test_loss += loss.item()
        
        test_history.append(test_loss / (dataset_size // 10))


        print(f"+-- Epoch: {epoch+1}/{epochs} -------------------")
        print(f"| Training loss:   {train_history[-1]}")
        print(f"| Validation loss: {test_history[-1]}")
        print(f"+-----------------------------------")
        plot_loss(train_history, test_history, "loss.png")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), "model.pth")

    return train_history, test_history


def plot_loss(train_history, test_history, filepath):
    ## PLOT LOSS
    x = torch.arange(len(train_history))
    fig = plt.figure(figsize=(4, 4), dpi=400)
    ax = fig.add_subplot(111)
    ax.plot(x, train_history)
    ax.plot(x, test_history)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return



if __name__ == '__main__':

    model = AttenuationMechanism(
        query_dim=32,
        context_dim=3,
        hidden_dim=32,
        num_heads=1,
        dropout=0,
        num_embeddings=4,
    )

    train(
        model, 
        num_features=3, 
        dataset_size=65536, 
        epochs=1_000, 
        batch_size=8192
    )

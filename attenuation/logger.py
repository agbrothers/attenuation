import os
import csv
import pandas as pd
from copy import copy
from datetime import datetime

FIELDS = [
    "key",
    "dataset_name",
    "model_name",
    "type",
    "run",
    "timestamp",
    "seed",
    "epochs",
    "batch_size",
    "lr",
    "hidden_dim",
    "context_len",
    "num_layers",
    "num_heads",
    "dropout",
    "loss_func",
]


class Logger:

    def __init__(self, path, model_config, training_config):
        ## PARSE ARGUMENTS
        self.path = path
        self.model_config = model_config
        self.training_config = training_config

        ## CONSTRUCT MODEL KEY
        model = model_config["model_name"]
        dataset = training_config["dataset_name"]
        key = self.key = f"{model}_{dataset}_train"
        
        row = [""] * len(FIELDS)
        for field, value in training_config.items():
            if field in FIELDS: row[FIELDS.index(field)] = str(value)
        for field, value in model_config.items():
            if field in FIELDS: row[FIELDS.index(field)] = str(value)

        ## LOG MODEL CONFIG TO JSON

        ## CREATE LOG IF IT DOESN'T EXIST
        if not os.path.exists(os.path.dirname(path)):            
            os.makedirs(os.path.dirname(path))
        if not os.path.exists(path):
            with open(path,'a') as log:
                log.write(",".join(FIELDS)+",\n")

        ## DETERMINE RUN AND TIMESTAMP
        log = pd.read_csv(path)
        runs = log.loc[log['key'] == key]["run"].values
        self.run = runs[-1] + 1 if len(runs) else 0
        timestamp = datetime.now().strftime("%m/%d/%Y_%H:%M:%S")

        ## ADD METADATA TO NEW ROW
        row[FIELDS.index("key")] = key
        row[FIELDS.index("run")] = str(self.run)
        row[FIELDS.index("timestamp")] = timestamp

        self.train_row = row
        self.train_row[FIELDS.index("type")] = "train"
        self.val_row = copy(row)
        self.val_row[FIELDS.index("key")] = key.replace("train", "val")
        self.val_row[FIELDS.index("type")] = "val"
        self.test_row = copy(row)
        self.test_row[FIELDS.index("key")] = key.replace("train", "test")
        self.test_row[FIELDS.index("type")] = "test"


    def log(self, train_loss, val_loss=None, test_loss=None):
        ## LOG NEW ROW
        with open(self.path,'a') as log:
            log.write(",".join(self.train_row + [str(i) for i in train_loss])+",\n")

            if val_loss:

                log.write(",".join(self.val_row + [str(i) for i in val_loss])+",\n")

            if test_loss:
                log.write(",".join(self.test_row + [str(i) for i in test_loss])+",\n")

        return

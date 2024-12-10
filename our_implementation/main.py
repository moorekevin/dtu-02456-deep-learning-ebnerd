#!/usr/bin/env python
# coding: utf-8

# # Simplified Jupyter notebook

# In[ ]:


import sys
import os

# Get the current directory
current_dir = os.getcwd()

# Append the relative path to the utils folder
sys.path.append(os.path.join(current_dir, "utils"))

import utils.model
from importlib import reload

reload(utils.model)
from utils.model import (
    HParams,
    Path,
    prepare_training_data,
    torch,
    NRMSModel,
    train_and_evaluate,
    evaluate_model,
)


# In[ ]:


# Setting hyperparameters
hparams = HParams()
hparams.data_fraction = 0.1
hparams.batch_size = 32

# Preprocessing and Loading Data
# Data loading
PATH = Path("~/Git Repositories/ebnerd-benchmark/data").expanduser()
DATASPLIT = "ebnerd_small"

print("Loading data from ", PATH)
train_loader, val_loader, word_embeddings = prepare_training_data(
    hparams, PATH, DATASPLIT
)

# Initialize and train model
print("Training model with ", hparams)
device = torch.device("cunda" if torch.cuda.is_available() else "cpu")
model = NRMSModel(hparams, word_embeddings)

model = train_and_evaluate(
    device,
    model,
    train_loader,
    val_loader,
    num_epochs=hparams.epochs,
    learning_rate=1e-3,
    patience=3,
)

# Evaluate model
metrics = evaluate_model(model, val_loader, device)
print("\nValidation Metrics:")
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:.4f}")


#!/usr/bin/env python
# coding: utf-8

# # PyTorch Implementation of the NRMS Model for `EBNeRD` (RecSys'24 Challenge)
# ### Course: `02456 Deep Learning` (Fall 2024)  
# **Institution:** Technical University of Denmark (DTU)  
# **Authors:** Kevin Moore (s204462) and Nico Tananow (s[insert number])
# 
# ### Acknowledgments  
# 1. Special thanks to **Johannes Kruse** for his [TensorFlow implementation of the NRMS Model](https://github.com/ebanalyse/ebnerd-benchmark), which greatly supported the development of this PyTorch implementation for the EBNeRD project.  
# 
# 
# 2. Our implementation is based on the NRMS model described in the paper **["Neural News Recommendation with Multi-Head Self-Attention"](https://aclanthology.org/D19-1671/)** by Wu et al. (2019).

# # Implementation

# ## 1. Importing Dependencies
# Import all necessary libraries and modules, including `utils.model` and `utils.helper` for the NRMS model, data preparation, training, and evaluation.

# In[1]:


import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warnings in transformers
from pathlib import Path
# Get the current directory
current_dir = os.getcwd()
root_dir = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(root_dir, "src")
# Append the relative path to the utils folder and ebrec src
sys.path.append(os.path.join(current_dir, "utils"))
sys.path.append(src_dir)
from importlib import reload

import torch
import utils.model
reload(utils.model)
from utils.model import (
    NRMSModel
)

import utils.helper
reload(utils.helper)
from utils.helper import (
    HParams,
    load_articles_and_embeddings,
    prepare_training_data,
    prepare_test_data,
    train_model,
    evaluate_model,
)


# ## 2. Setting Hyperparameters
# Initialize the hyperparameters for the model

# In[2]:


# Setting hyperparameters
hparams = HParams()
hparams.data_fraction = 0.01
hparams.batch_size = 32


# ### 3. Loading Data
# 1. Loading and creating **embeddings** for articles
# 2. Loading **training** data and splitting it to training/validation (from `{datasplit}/train`)
# 3. Loading **testing** data for final evaluation (from `{datasplit}/validation`)

# In[ ]:


PATH = Path(os.path.join(current_dir, "data"))
DATASPLIT = "ebnerd_large"
print("Loading data from ", PATH, "with datasplit:", DATASPLIT)

# Loading articles and embeddings
article_mapping, word_embeddings = load_articles_and_embeddings(hparams, PATH)

# Training Data
train_loader, val_loader = prepare_training_data(
    hparams, PATH, DATASPLIT, article_mapping
)

# Test Data for final evaluation
test_loader = prepare_test_data(
    hparams, PATH, DATASPLIT, article_mapping
)


# ### 4. Training the Model
# - Initialize the NRMS model with preloaded hyperparameters and embeddings. 
# - Train the model using the training and validation datasets. 
# - Early stopping is applied with a patience parameter of 3 to prevent overfitting.

# In[4]:


print("Training model with ", hparams)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NRMSModel(hparams, word_embeddings)

model = train_model(
    device,
    model,
    train_loader,
    val_loader,
    hparams,
    patience=3,
)


# ### 5. Evaluating the Model
# - Evaluate the trained NRMS model on the testing dataset. 
# - Print out performance metrics

# In[5]:


# Evaluate model
metrics = evaluate_model(model, test_loader, device)
print("\nValidation Metrics:")
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:.4f}")


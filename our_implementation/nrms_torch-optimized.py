#!/usr/bin/env python
# coding: utf-8

# # NRMS Model (PyTorch Version)
# 
# This notebook demonstrates how to build, train, and evaluate a Neural News Recommendation Model (NRMS) using PyTorch instead of TensorFlow. We will still attempt to use `ebrec` utilities for data loading and evaluation where possible.
# 
# ## Overview
# 
# We will:
# 1.  Setup: Import necessary libraries and define hyperparameters.
# 2.  Define NRMS Model Components: Implement custom layers and the NRMS model architecture.
# 3.  Data Loading and Preparation: Load and preprocess the dataset.
# 4.  Article Embeddings: Generate embeddings for articles using a pre-trained transformer model.
# 5.  Batch and Shape Data: Create PyTorch datasets and dataloaders.
# 6.  Training the Model: Train the NRMS model.
# 7.  Evaluation on Test Set: Evaluate the trained model.

# ## Imports
import sys
import os

# Add the parent directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
sys.path.append(root_dir)
sys.path.append(data_dir)
print(sys.path)
# In[18]:


import datetime
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score

from ebrec.utils._behaviors import ebnerd_from_path, create_binary_labels_column, sampling_strategy_wu2019
from ebrec.utils._articles import convert_text2encoding_with_transformers, create_article_id_to_value_mapping
from ebrec.utils._polars import concat_str_columns
from ebrec.utils._constants import (
	DEFAULT_USER_COL, DEFAULT_IMPRESSION_ID_COL, DEFAULT_IMPRESSION_TIMESTAMP_COL,
	DEFAULT_HISTORY_ARTICLE_ID_COL, DEFAULT_CLICKED_ARTICLES_COL, DEFAULT_INVIEW_ARTICLES_COL
)

# Set random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


# ## Hyperparameters

# In[ ]:


class HParams:
	title_size = 30
	head_num = 20
	head_dim = 20
	attention_hidden_dim = 200
	dropout = 0.2
	batch_size = 32
	verbose = False
	data_fraction = 1 # Fraction of data to use
	sampling_nratio = 4 # For every positive sample ( a click ), we sample X negative samples
	history_size = 20 # History of each users interactions will be limited to the most recent X articles
	transformer_model_name = "facebookai/xlm-roberta-base"

	def __str__(self):
		return (
			f"\n title_size: {self.title_size}"
			f"\n head_num: {self.head_num}"
			f"\n head_dim: {self.head_dim}"
			f"\n attention_hidden_dim: {self.attention_hidden_dim}"
			f"\n dropout: {self.dropout}"
			f"\n batch_size: {self.batch_size}"
			f"\n verbose: {self.verbose}"
			f"\n data_fraction: {self.data_fraction}"
			f"\n sampling_nratio: {self.sampling_nratio}"
			f"\n history_size: {self.history_size}"
			f"\n transformer_model_name: {self.transformer_model_name}"
		)


# ## Defining Model

# In[20]:


class SelfAttention(nn.Module):
	def __init__(self,hparams, verbose=False):
		super().__init__()
		self.head_num = hparams.head_num
		self.head_dim = hparams.head_dim
		self.output_dim = self.head_num * self.head_dim
		self.WQ = self.WK = self.WV = None
		self.dropout = nn.Dropout(hparams.dropout)
		self.verbose = verbose

	def forward(self, Q_seq, K_seq, V_seq):
		# Lazy initialization of the weights
		if self.WQ is None:
			embedding_dim = Q_seq.size(-1)
			self.WQ = nn.Linear(embedding_dim, self.output_dim)
			self.WK = nn.Linear(embedding_dim, self.output_dim)
			self.WV = nn.Linear(embedding_dim, self.output_dim)
		
		Q = self.WQ(Q_seq)
		K = self.WK(K_seq)
		V = self.WV(V_seq)

		N, L, _ = Q.size()
		Q = Q.view(N, L, self.head_num, self.head_dim).transpose(1, 2)
		K = K.view(N, L, self.head_num, self.head_dim).transpose(1, 2)
		V = V.view(N, L, self.head_num, self.head_dim).transpose(1, 2)

		if self.verbose:
			print(f"Q shape: {Q.shape}")
			print(f"K shape: {K.shape}")
			print(f"V shape: {V.shape}")

		scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
		attn = torch.softmax(scores, dim=-1)
		attn = self.dropout(attn)

		output = torch.matmul(attn, V)
		output = output.transpose(1, 2).contiguous().view(N, L, self.output_dim)

		if self.verbose:
			print(f"Attention shape: {attn.shape}")
			print(f"Output shape: {output.shape}")

		return output

class AttLayer(nn.Module):
	def __init__(self, hparams, verbose=False):
		super().__init__()
		self.W = nn.Linear(hparams.head_num * hparams.head_dim, hparams.attention_hidden_dim)
		self.q = nn.Linear(hparams.attention_hidden_dim, 1, bias=False)
		self.dropout = nn.Dropout(hparams.dropout)
		self.verbose = verbose

	def forward(self, x):
		attn = torch.tanh(self.W(x))
		attn = self.q(attn).squeeze(-1)
		attn = torch.softmax(attn, dim=1).unsqueeze(-1)

		if self.verbose:
			print(f"Attention weights shape: {attn.shape}")
			print(f"Input shape: {x.shape}")

		output = torch.sum(x * attn, dim=1)
		output = self.dropout(output)

		if self.verbose:
			print(f"Output shape: {output.shape}")

		return output
 
class NRMSModel(nn.Module):
	def __init__(self, hparams, word_embeddings):
		super().__init__()
		self.embedding = nn.Embedding.from_pretrained(
			torch.FloatTensor(word_embeddings), freeze=False
			)
		self.dropout = nn.Dropout(hparams.dropout)

		# News Encoder
		self.news_self_att = SelfAttention(hparams, verbose=hparams.verbose)
		self.news_att = AttLayer(hparams, verbose=hparams.verbose)

		# User Encoder
		self.user_self_att = SelfAttention(hparams, verbose=hparams.verbose)
		self.user_att = AttLayer(hparams, verbose=hparams.verbose)

	def encode_news(self, news_input):
		x = self.embedding(news_input)
		x = self.dropout(x)
		x = self.news_self_att(x, x, x)
		x = self.news_att(x)
		return x

	def encode_user(self, history_input):
		N, H, L = history_input.size()
		history_input = history_input.view(N * H, L)
		news_vectors = self.encode_news(history_input)
		news_vectors = news_vectors.view(N, H, -1)
		user_vector = self.user_self_att(news_vectors, news_vectors, news_vectors)
		user_vector = self.user_att(user_vector)
		return user_vector

	def forward(self, his_input, pred_input):
		user_vector = self.encode_user(his_input)
		N, M, L = pred_input.size()
		pred_input = pred_input.view(N * M, L)
		news_vectors = self.encode_news(pred_input)
		news_vectors = news_vectors.view(N, M, -1)
		scores = torch.bmm(news_vectors, user_vector.unsqueeze(2)).squeeze(-1)
		return scores


# ## Helper functions

# ### Loading Data Helper Function

# In[21]:


class NRMSDataset(Dataset):
	def __init__(self, df, article_mapping, title_size, history_column, candidate_column, verbose=False):
		"""
		Args:
			df (pl.DataFrame): DataFrame containing raw history and candidate article IDs.
			article_mapping (dict): Mapping of article IDs to tokenized representations.
			title_size (int): Maximum size of title tokens for padding/truncation.
			history_column (str): Column containing user history.
			candidate_column (str): Column containing candidate articles.
			verbose (bool): If True, prints debug information.
		"""
		self.history_raw = df[history_column].to_list()
		self.candidates_raw = df[candidate_column].to_list()
		self.labels = df["labels"].to_list()
		self.article_mapping = article_mapping
		self.title_size = title_size
		self.verbose = verbose

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		# Convert article IDs to tokenized representations
		history_tokens = [
			self.article_mapping.get(aid, [0] * self.title_size) for aid in self.history_raw[idx]
		]
		candidate_tokens = [
			self.article_mapping.get(aid, [0] * self.title_size) for aid in self.candidates_raw[idx]
		]
		
		# Convert to PyTorch tensors
		his_ids = torch.tensor(history_tokens, dtype=torch.long)
		pred_ids = torch.tensor(candidate_tokens, dtype=torch.long)
		y = torch.tensor(self.labels[idx], dtype=torch.float32)

		if self.verbose:
			print(f"History Tokens: {his_ids.shape}")
			print(f"Candidate Tokens: {pred_ids.shape}")
			print(f"Label: {y.shape}")

		return his_ids, pred_ids, y

def nrms_collate_fn(batch):
	histories, candidates, labels = zip(*batch)
	max_candidates = max([cand.size(0) for cand in candidates])
	
	padded_candidates = []
	candidate_masks = []
	for cand in candidates:
		num_cands = cand.size(0)
		if num_cands < max_candidates:
			pad_size = max_candidates - num_cands
			padded_cand = torch.cat([cand, torch.zeros(pad_size, cand.size(1), dtype=torch.long)])
			mask = torch.cat([torch.ones(num_cands, dtype=torch.bool), torch.zeros(pad_size, dtype=torch.bool)])
		else:
			padded_cand = cand[:max_candidates]
			mask = torch.ones(max_candidates, dtype=torch.bool)
		padded_candidates.append(padded_cand)
		candidate_masks.append(mask)
	
	padded_candidates = torch.stack(padded_candidates)
	candidate_masks = torch.stack(candidate_masks)
	histories = torch.stack(histories)
	
	padded_labels = []
	for label in labels:
		num_cands = label.size(0)
		if num_cands < max_candidates:
			pad_size = max_candidates - num_cands
			padded_label = torch.cat([label, torch.zeros(pad_size, dtype=torch.float32)])
		else:
			padded_label = label[:max_candidates]
		padded_labels.append(padded_label)
	padded_labels = torch.stack(padded_labels)
	
	return {
		'history': histories,
		'candidates': padded_candidates,
		'labels': padded_labels,
		'candidate_masks': candidate_masks
	}

def create_dataloader(df, article_mapping, title_size, batch_size, history_column, candidate_column, shuffle=False):
	dataset = NRMSDataset(
		df, article_mapping, title_size, history_column, candidate_column
	)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		collate_fn=nrms_collate_fn,
		num_workers=0,  # Adjust based on your system
	)

def prepare_df_for_training(df):
	"""
	Validate and preprocess DataFrame to ensure required columns exist.
	"""
	required_columns = [DEFAULT_HISTORY_ARTICLE_ID_COL, DEFAULT_INVIEW_ARTICLES_COL, "labels"]
	for col in required_columns:
		if col not in df.columns:
			raise ValueError(f"Missing required column: {col}")
	return df


# ### Training Helper Function
# 1. Optimizer: Adam
# 2. Loss Function: Cross Entropy 

# In[22]:


def train_and_evaluate(device, model, train_loader, val_loader, num_epochs, learning_rate=1e-3, patience=3):
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	best_val_loss = float('inf')
	best_val_auc = 0
	patience_counter = 0

	for epoch in range(num_epochs):
		# Training
		model.train()
		total_loss = 0
		with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
			for batch in train_loader:
				his_input = batch['history'].to(device)
				pred_input = batch['candidates'].to(device)
				labels = batch['labels'].to(device)
				masks = batch['candidate_masks'].to(device)

				optimizer.zero_grad()
				scores = model(his_input, pred_input)
				scores = scores * masks
				loss = criterion(scores, labels)

				loss.backward()
				optimizer.step()
				total_loss += loss.item()
				pbar.update(1)

		avg_train_loss = total_loss / len(train_loader)

		# Validation
		model.eval()
		val_loss = 0
		all_scores = []
		all_labels = []
		with torch.no_grad(), tqdm(total=len(val_loader), desc="Validation", unit="batch") as pbar:
			for batch in val_loader:
				his_input = batch['history'].to(device)
				pred_input = batch['candidates'].to(device)
				labels = batch['labels']
				masks = batch['candidate_masks']

				scores = model(his_input, pred_input)
				scores = scores * masks
				loss = criterion(scores, labels.to(device))
				val_loss += loss.item()

				valid_scores = scores[masks.bool()].cpu().numpy()
				valid_labels = labels[masks.bool()].numpy()
				all_scores.extend(valid_scores)
				all_labels.extend(valid_labels)
				pbar.update(1)

		avg_val_loss = val_loss / len(val_loader)
		val_auc = roc_auc_score(all_labels, all_scores)
		auc_improvement = val_auc - best_val_auc if epoch > 0 else val_auc
		print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}, Improvement from Previous Epoch: {auc_improvement:.4f}")
		best_val_auc = max(best_val_auc, val_auc)
		
		checkpoint_dir = "checkpoints"
		os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists
		checkpoint_path = f"checkpoints/nrms_checkpoint_{epoch+1}.pth"
		torch.save({
			'epoch': epoch+1, 
			'model_state_dict': model.state_dict(), 
			'optimizer_state_dict': optimizer.state_dict(), 
			'loss': avg_val_loss, 
			'auc': val_auc}, checkpoint_path)
		print(f"Checkpoint saved to: {checkpoint_path}")


		# Early stopping
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			patience_counter = 0
		else:
			patience_counter += 1
			if patience_counter >= patience:
				print("Early stopping triggered")
				break

	return model


# ## Training

# In[23]:


# Setting hyperparameters
hparams=HParams()
print("Hyperparameters:", hparams)


# ### Preprocessing and Loading Data

# In[24]:


# Data loading
PATH = Path("./data").resolve()
DATASPLIT = "ebnerd_small"

# Load and process training data
df_train = (
	ebnerd_from_path(
		PATH.joinpath(DATASPLIT, "train"),
		history_size=hparams.history_size,
		padding=0,
	)
	.pipe(
		sampling_strategy_wu2019,
		npratio=hparams.sampling_nratio,
		with_replacement=True,
		seed=seed,
	)
	.pipe(create_binary_labels_column)
	.sample(fraction=hparams.data_fraction)
)

# Split into train/validation
dt_split = df_train[DEFAULT_IMPRESSION_TIMESTAMP_COL].max() - datetime.timedelta(days=1)
df_train_split = df_train.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL) < dt_split)
df_validation = df_train.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL) >= dt_split)

# Load articles and prepare embeddings
df_articles = pl.read_parquet(PATH.joinpath("articles.parquet"))
transformer_model = AutoModel.from_pretrained(hparams.transformer_model_name)
transformer_tokenizer = AutoTokenizer.from_pretrained(hparams.transformer_model_name)
word_embeddings = transformer_model.get_input_embeddings().weight.detach().numpy()

# Prepare article embeddings
df_articles, cat_col = concat_str_columns(df_articles, columns=["subtitle", "title"])
df_articles, token_col_title = convert_text2encoding_with_transformers(
	df_articles, 
	transformer_tokenizer, 
	cat_col, 
	max_length=hparams.title_size
)
article_mapping = create_article_id_to_value_mapping(
	df=df_articles, 
	value_col=token_col_title
)


# ### Running Training

# In[25]:


# Validate DataFrames
df_train_split = prepare_df_for_training(df_train_split)
df_validation = prepare_df_for_training(df_validation)

# Create DataLoaders
train_loader = create_dataloader(
	df_train_split, article_mapping, hparams.title_size,
	batch_size=hparams.batch_size,
	history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
	candidate_column=DEFAULT_INVIEW_ARTICLES_COL,
	shuffle=True
)

val_loader = create_dataloader(
	df_validation, article_mapping, hparams.title_size,
	batch_size=hparams.batch_size,
	history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
	candidate_column=DEFAULT_INVIEW_ARTICLES_COL,
	shuffle=False
)

# Initialize and train model
print("Training model with ", hparams)
device = torch.device('cunda' if torch.cuda.is_available() else 'cpu')
model = NRMSModel(hparams, word_embeddings)

EPOCHS = 10

model = train_and_evaluate(
  device,
	model, 
	train_loader, 
	val_loader, 
	num_epochs=EPOCHS, 
	learning_rate=1e-3, 
	patience=3
)


# ## Evaluation

# In[26]:


def evaluate_model(model, dataloader, device):
	model.eval()
	all_scores = []
	all_labels = []
	with torch.no_grad():
		for batch in dataloader:
			his_input = batch['history'].to(device)
			pred_input = batch['candidates'].to(device)
			labels = batch['labels']
			masks = batch['candidate_masks']

			scores = model(his_input, pred_input)
			valid_scores = scores[masks.bool()].cpu().numpy()
			valid_labels = labels[masks.bool()].numpy()
			all_scores.extend(valid_scores)
			all_labels.extend(valid_labels)

	all_scores = np.array(all_scores)
	all_labels = np.array(all_labels)

	metrics = {
		'auc': roc_auc_score(all_labels, all_scores),
	}
	return metrics

# Evaluate model
metrics = evaluate_model(model, val_loader, device)
print("\nValidation Metrics:")
for metric_name, value in metrics.items():
	print(f"{metric_name}: {value:.4f}")


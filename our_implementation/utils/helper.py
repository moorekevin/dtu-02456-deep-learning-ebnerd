import datetime
import os
from tqdm import tqdm
import json
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score

from ebrec.utils._behaviors import (
    ebnerd_from_path,
    create_binary_labels_column,
    sampling_strategy_wu2019,
)
from ebrec.utils._articles import (
    convert_text2encoding_with_transformers,
)
from ebrec.utils._polars import concat_str_columns
from ebrec.utils._constants import (
    DEFAULT_USER_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_ARTICLE_PUBLISHED_TIMESTAMP_COL,
    DEFAULT_IS_BEYOND_ACCURACY_COL,
    DEFAULT_LABELS_COL
)

# Set random seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
]


class HParams:
    def __init__(
        self,
        title_size=30,
        head_num=20,
        head_dim=20,
        attention_hidden_dim=200,
        dropout=0.2,
        batch_size=32,
        verbose=False,
        # Fraction of data to use
        data_fraction=1,
        # For every positive sample ( a click ), we sample X negative samples
        sampling_nratio=4,
        # History of each users interactions will be limited to the most recent X articles
        history_size=20,
        epochs=1,
        lr=1e-3,
        datasplit="ebnerd_small",
        transformer_model_name="facebookai/xlm-roberta-base"
    ):
        self.title_size = title_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.verbose = verbose
        self.data_fraction = data_fraction
        self.sampling_nratio = sampling_nratio
        self.history_size = history_size
        self.epochs = epochs
        self.lr = lr
        self.datasplit = datasplit
        self.transformer_model_name = transformer_model_name

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
            f"\n epochs: {self.epochs}"
            f"\n learning_rate: {self.lr}"
            f"\n transformer_model_name: {self.transformer_model_name}"
        )

# Helper functions

# Loading Data Helper Function


def nrms_collate_fn(batch):
    histories, histories_time, candidates, candidates_time, labels = zip(
        *batch)
    max_candidates = max([cand.size(0) for cand in candidates])

    padded_candidates = []
    padded_candidates_time = []
    candidate_masks = []
    for cand, cand_time in zip(candidates, candidates_time):
        num_cands = cand.size(0)
        if num_cands < max_candidates:
            pad_size = max_candidates - num_cands
            padded_cand = torch.cat(
                [cand, torch.zeros(pad_size, cand.size(1), dtype=torch.long)]
            )

            padded_time = torch.cat(
                [cand_time, torch.zeros(pad_size, dtype=torch.float32)]
            )

            mask = torch.cat(
                [
                    torch.ones(num_cands, dtype=torch.bool),
                    torch.zeros(pad_size, dtype=torch.bool),
                ]
            )
        else:
            padded_cand = cand[:max_candidates]
            padded_time = cand_time[:max_candidates]
            mask = torch.ones(max_candidates, dtype=torch.bool)
        padded_candidates.append(padded_cand)
        padded_candidates_time.append(padded_time)
        candidate_masks.append(mask)

    padded_candidates = torch.stack(padded_candidates)
    padded_candidates_time = torch.stack(padded_candidates_time)
    candidate_masks = torch.stack(candidate_masks)
    histories = torch.stack(histories)
    histories_time = torch.stack(histories_time)

    padded_labels = []
    for label in labels:
        num_cands = label.size(0)
        if num_cands < max_candidates:
            pad_size = max_candidates - num_cands
            padded_label = torch.cat(
                [label, torch.zeros(pad_size, dtype=torch.long)]
            )
        else:
            padded_label = label[:max_candidates]
        padded_labels.append(padded_label)
    padded_labels = torch.stack(padded_labels)

    return {
        "history": histories,
        "history_time": histories_time,
        "candidates": padded_candidates,
        "candidates_time": padded_candidates_time,
        "labels": padded_labels,
        "candidate_masks": candidate_masks,
    }


class NRMSDataset(Dataset):
    def __init__(
        self,
        df,
        article_mapping,
        title_size,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        candidate_column=DEFAULT_INVIEW_ARTICLES_COL,
        verbose=False,
    ):
        self.history_raw = df[history_column].to_list()
        self.candidates_raw = df[candidate_column].to_list()
        self.labels = df["labels"].to_list()
        self.article_mapping = article_mapping
        self.title_size = title_size
        self.verbose = verbose
        # Day before first day of data collection (april 26th)
        self.reference_time = datetime.datetime(2023, 4, 26)

        self.days_since_reference = [
            self.datetime_to_days_since_reference(article["timepublished"])
            for article in article_mapping.values()
        ]

        # Step 2: Compute mean and standard deviation
        self.mean_days = np.mean(self.days_since_reference)
        self.std_days = np.std(self.days_since_reference)

    def datetime_to_days_since_reference(self, dt):
        """Convert datetime to days since the reference time."""
        return (dt - self.reference_time).total_seconds() / 86400  # Convert to days

    def standardize_time(self, dt):
        """Standardize time based on precomputed mean and std."""
        days_since_ref = self.datetime_to_days_since_reference(dt)
        return (days_since_ref - self.mean_days) / self.std_days

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert article IDs to tokenized representations
        history_tokens = [
            self.article_mapping.get(aid, {"tokens": [
                                     0] * self.title_size, "timepublished": self.reference_time})["tokens"]
            for aid in self.history_raw[idx]
        ]

        history_timepublished = [
            self.standardize_time(
                self.article_mapping.get(aid, {"tokens": [0] * self.title_size, "timepublished": self.reference_time})[
                    "timepublished"]
            )
            for aid in self.history_raw[idx]
        ]

        candidate_tokens = [
            self.article_mapping.get(aid, {"tokens": [
                                     0] * self.title_size, "timepublished": self.reference_time})["tokens"]
            for aid in self.candidates_raw[idx]
        ]

        candidate_timepublished = [
            self.standardize_time(
                self.article_mapping.get(aid, {"tokens": [0] * self.title_size, "timepublished": self.reference_time})[
                    "timepublished"]
            )
            for aid in self.candidates_raw[idx]
        ]

        # Convert to PyTorch tensors
        his_ids = torch.tensor(history_tokens, dtype=torch.long)
        his_time = torch.tensor(history_timepublished, dtype=torch.float32)
        pred_ids = torch.tensor(candidate_tokens, dtype=torch.long)
        pred_time = torch.tensor(candidate_timepublished, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.verbose:
            print(f"History Tokens: {his_ids.shape}")
            print(f"Candidate Tokens: {pred_ids.shape}")
            print(f"Label: {y.shape}")

        return his_ids, his_time, pred_ids, pred_time, y

    def get_dataloader(self, batch_size, num_workers=2):
        print(f"Using {num_workers} workers for dataloading")
        return DataLoader(
            self,
            batch_size,
            collate_fn=nrms_collate_fn,
            num_workers=num_workers,
        )


def validate_df_col(df):
    """
    Validate and preprocess DataFrame to ensure required columns exist.
    """
    required_columns = [
        DEFAULT_HISTORY_ARTICLE_ID_COL,
        DEFAULT_INVIEW_ARTICLES_COL,
        "labels",
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df


def load_articles_and_embeddings(hparams, PATH):
    # Load articles
    df_articles = pl.read_parquet(PATH.joinpath("articles.parquet"))
    # -> Concatenate title and subtitle
    df_articles, cat_col = concat_str_columns(
        df_articles, columns=["subtitle", "title"])

    # Load Tokenizer
    transformer_tokenizer = AutoTokenizer.from_pretrained(
        hparams.transformer_model_name)
    # # -> Convert text to tokenized representation and add it to df_articles
    df_articles, token_col_title = convert_text2encoding_with_transformers(
        df_articles, transformer_tokenizer, cat_col, max_length=hparams.title_size
    )
    # # -> Create article mapping
    df_mapping = df_articles.select(
        DEFAULT_ARTICLE_ID_COL, token_col_title, DEFAULT_ARTICLE_PUBLISHED_TIMESTAMP_COL)
    article_mapping = {
        article_id: {
            "tokens": token,
            "timepublished": time_published
        }
        for article_id, token, time_published in zip(
            df_mapping[DEFAULT_ARTICLE_ID_COL],
            df_mapping[token_col_title],
            df_mapping[DEFAULT_ARTICLE_PUBLISHED_TIMESTAMP_COL]
        )
    }
    # Load Transformer Model and get word embeddings
    transformer_model = AutoModel.from_pretrained(
        hparams.transformer_model_name)
    word_embeddings = transformer_model.get_input_embeddings().weight.detach().numpy()

    return article_mapping, word_embeddings


# ### Training Helper Function
# 1. Optimizer: Adam
# 2. Loss Function: Cross Entropy
def prepare_training_data(hparams, PATH, DATASPLIT, article_mapping):
    # Load and sample training data
    df = (
        ebnerd_from_path(
            PATH.joinpath(DATASPLIT, "train"),
            history_size=hparams.history_size,
            padding=0,
        )
        # pl.concat(
        #     [
        #         ebnerd_from_path(
        #             PATH.joinpath(DATASPLIT, "train"),
        #             history_size=hparams.history_size,
        #             padding=0,
        #         ),
        #         ebnerd_from_path(
        #             PATH.joinpath(DATASPLIT, "validation"),
        #             history_size=hparams.history_size,
        #             padding=0,
        #         ),
        #     ]
        # )
        .sample(fraction=hparams.data_fraction)
        .select(COLUMNS)
        .pipe(
            sampling_strategy_wu2019,
            npratio=hparams.sampling_nratio,
            shuffle=True,
            with_replacement=True,
            seed=SEED,
        )
        .pipe(create_binary_labels_column)
    )

    # Split into train/validation (last day is validation)
    last_day = df[DEFAULT_IMPRESSION_TIMESTAMP_COL].max(
    ) - datetime.timedelta(days=1)
    df_train = df.filter(
        pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL) < last_day)
    df_validation = df.filter(
        pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL) >= last_day)

    # # -> Validate DataFrames
    df_train = validate_df_col(df_train)
    df_validation = validate_df_col(df_validation)

    # Create DataLoaders
    # # -> Training
    train_dataset = NRMSDataset(
        df_train, article_mapping, hparams.title_size
    )
    train_loader = train_dataset.get_dataloader(
        hparams.batch_size)

    # # -> Validation
    val_dataset = NRMSDataset(
        df_validation, article_mapping, hparams.title_size
    )
    val_loader = val_dataset.get_dataloader(hparams.batch_size)

    print(
        f" -> Train samples: {df_train.height}\n -> Validation samples: {df_validation.height}")

    return train_loader, val_loader


def save_hparams(device, hparams):
    # Create a unique directory for checkpoints based on the current ISO date-time
    timestamp = datetime.datetime.now().isoformat(
        timespec="seconds").replace(":", "-")
    checkpoint_dir = os.path.join("checkpoints", timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)

    def hparams_to_dict(hparams_class):
        return {key: value for key, value in vars(hparams_class).items()}

    hparams_dict = hparams_to_dict(hparams)

    # Save to JSON
    info_path = os.path.join(checkpoint_dir, "info.json")
    info_data = {
        "Training Timestamp": timestamp,
        "Device": str(device),
        "Hyperparameters": hparams_dict,
    }
    with open(info_path, "w") as json_file:
        json.dump(info_data, json_file, indent=4)  # Pretty-print with indent=4

    print(f"Training information saved to: {checkpoint_dir}")
    return checkpoint_dir


def save_checkpoint(checkpoint_dir, epoch, model, optimizer, avg_val_loss, val_auc):
    checkpoint_path = f"{checkpoint_dir}/nrms_checkpoint_{epoch+1}.pth"
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_val_loss,
            "auc": val_auc,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to: {checkpoint_path}")


def train_model(
    device, model, train_loader, val_loader, hparams, patience=3, checkpoint_dir=None
):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=hparams.lr)

    best_val_loss = float("inf")
    best_val_auc = 0
    patience_counter = 0

    for epoch in range(hparams.epochs):
        # Training
        model.train()
        total_loss = 0
        with tqdm(
            total=len(train_loader), desc=f"Epoch {epoch+1}/{hparams.epochs}", unit="batch", dynamic_ncols=True
        ) as pbar:
            for batch in train_loader:
                his_input = batch["history"].to(device)
                his_time = batch["history_time"].to(device)
                pred_input = batch["candidates"].to(device)
                pred_time = batch["candidates_time"].to(device)
                labels = batch["labels"].to(device)
                masks = batch["candidate_masks"].to(device)

                optimizer.zero_grad()
                scores = model(his_input, his_time, pred_input, pred_time)
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
        with (
            torch.no_grad(),
            tqdm(total=len(val_loader), desc="Validation", unit="batch",  dynamic_ncols=True) as pbar,
        ):
            for batch in val_loader:
                his_input = batch["history"].to(device)
                his_time = batch["history_time"].to(device)
                pred_input = batch["candidates"].to(device)
                pred_time = batch["candidates_time"].to(device)
                labels = batch["labels"].to(device)
                masks = batch["candidate_masks"].to(device)

                scores = model(his_input, his_time, pred_input, pred_time)
                scores = scores * masks
                loss = criterion(scores, labels.to(device))
                val_loss += loss.item()

                valid_scores = scores[masks.bool()].detach().cpu().numpy()
                valid_labels = labels.to(
                    device)[masks.bool()].detach().cpu().numpy()
                all_scores.extend(valid_scores)
                all_labels.extend(valid_labels)
                pbar.update(1)

        avg_val_loss = val_loss / len(val_loader)
        val_auc = roc_auc_score(all_labels, all_scores)
        auc_improvement = val_auc - best_val_auc if epoch > 0 else val_auc
        print(
            f"Epoch {epoch+1}/{hparams.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}, Improvement from Previous Epoch: {auc_improvement:.4f}"
        )
        best_val_auc = max(best_val_auc, val_auc)

        save_checkpoint(checkpoint_dir, epoch, model,
                        optimizer, avg_val_loss, val_auc)

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


def evaluate_model(model, dataloader, device):
    model.eval()
    all_scores = []
    all_labels = []
    with (
            torch.no_grad(),
            tqdm(total=len(dataloader), desc="Testing", unit="batch", dynamic_ncols=True) as pbar
    ):
        for batch in dataloader:
            his_input = batch["history"].to(device)
            his_time = batch["history_time"].to(device)
            pred_input = batch["candidates"].to(device)
            pred_time = batch["candidates_time"].to(device)
            labels = batch["labels"].to(device)
            masks = batch["candidate_masks"].to(device)

            scores = model(his_input, his_time, pred_input, pred_time)
            valid_scores = scores[masks.bool()].detach().cpu().numpy()
            valid_labels = labels.to(
                device)[masks.bool()].detach().cpu().numpy()
            all_scores.extend(valid_scores)
            all_labels.extend(valid_labels)
            pbar.update(1)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    metrics = {
        'auc': roc_auc_score(all_labels, all_scores),
    }
    return metrics


def predict_scores(model, dataloader, device):
    model.eval()
    all_scores = []
    all_labels = []
    with (
            torch.no_grad(),
            tqdm(total=len(dataloader), desc="Testing", unit="batch", dynamic_ncols=True) as pbar
    ):
        for batch in dataloader:
            his_input = batch["history"].to(device)
            his_time = batch["history_time"].to(device)
            pred_input = batch["candidates"].to(device)
            pred_time = batch["candidates_time"].to(device)
            labels = batch["labels"].to(device)
            masks = batch["candidate_masks"].to(device)

            # Model predictions (logits to probabilities)
            scores = model(his_input, his_time, pred_input, pred_time)
            scores = torch.sigmoid(scores)  # Convert logits to probabilities

            # Group valid scores and labels by user
            for score_row, label_row, mask_row in zip(
                scores.cpu().numpy(), labels.cpu().numpy(), masks.cpu().numpy()
            ):
                valid_scores = score_row[mask_row.astype(bool)]
                valid_labels = label_row[mask_row.astype(bool)]
                all_scores.append(valid_scores.tolist())
                all_labels.append(valid_labels.tolist())

            pbar.update(1)

    return all_scores, all_labels

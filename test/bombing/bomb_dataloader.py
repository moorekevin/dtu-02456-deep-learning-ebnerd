from pathlib import Path
import polars as pl
import numpy as np

from ebrec.models.newsrec.dataloader import (
    NRMSDataLoader,
)
from ebrec.utils._behaviors import create_user_id_to_int_mapping
from ebrec.utils._articles import create_article_id_to_value_mapping

from ebrec.utils._python import time_it
from tqdm import tqdm

from ebrec.utils._behaviors import create_binary_labels_column
from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_CATEGORY_COL,
    DEFAULT_USER_COL,
)

from torch.utils.data import DataLoader

N_ITERATIONS = 300
BATCH_SIZE = 100
TOKEN_COL = "tokens"
N_SAMPLES = "n"

# LOAD DATA:
PATH_DATA = Path("test/data")
df_articles = (
    pl.scan_parquet(PATH_DATA.joinpath("ebnerd", "articles.parquet"))
    .select(pl.col(DEFAULT_ARTICLE_ID_COL, DEFAULT_CATEGORY_COL))
    .with_columns(pl.Series(TOKEN_COL, np.random.randint(0, 20, (1, 10))))
    .collect()
)
df_history = (
    pl.scan_parquet(PATH_DATA.joinpath("ebnerd", "history.parquet"))
    .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
    .with_columns(pl.col(DEFAULT_HISTORY_ARTICLE_ID_COL).list.tail(3))
)
df_behaviors = (
    pl.scan_parquet(PATH_DATA.joinpath("ebnerd", "behaviors.parquet"))
    .select(DEFAULT_USER_COL, DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL)
    .with_columns(pl.col(DEFAULT_INVIEW_ARTICLES_COL).list.len().alias(N_SAMPLES))
    .join(df_history, on=DEFAULT_USER_COL, how="left")
    .collect()
    .pipe(create_binary_labels_column)
)
# => MAPPINGS:
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=TOKEN_COL
)
user_mapping = create_user_id_to_int_mapping(df=df_behaviors)
# => NPRATIO IMPRESSION - SAME LENGTHS:
df_behaviors_train = df_behaviors.filter(pl.col(N_SAMPLES) == pl.col(N_SAMPLES).min())
# => FOR TEST-DATALOADER
label_lengths = df_behaviors[DEFAULT_INVIEW_ARTICLES_COL].list.len().to_list()


def iter_dataloader(dataloader, name: str, iterations: int):
    for _ in tqdm(range(iterations), desc=name):
        for _ in dataloader:
            pass


# ===
@time_it(True)
def bomb_NRMSDataLoader():
    dataloader = NRMSDataLoader(
        behaviors=df_behaviors_train,
        article_dict=article_mapping,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        unknown_representation="zeros",
        eval_mode=False,
        batch_size=BATCH_SIZE,
    )
    iter_dataloader(dataloader, "NRMS-train", iterations=N_ITERATIONS)

    dataloader = NRMSDataLoader(
        behaviors=df_behaviors,
        article_dict=article_mapping,
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        unknown_representation="zeros",
        eval_mode=True,
        batch_size=BATCH_SIZE,
    )
    iter_dataloader(dataloader, "NRMS-test", iterations=N_ITERATIONS)


if __name__ == "__main__":
    bomb_NRMSDataLoader()
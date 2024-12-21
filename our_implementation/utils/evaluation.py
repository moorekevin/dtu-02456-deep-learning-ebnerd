from ebrec.utils._constants import (DEFAULT_IMPRESSION_ID_COL, DEFAULT_LABELS_COL,
                                    DEFAULT_INVIEW_ARTICLES_COL, DEFAULT_CLICKED_ARTICLES_COL, DEFAULT_IS_BEYOND_ACCURACY_COL)
from ebrec.utils._python import rank_predictions_by_score, write_submission_file
from ebrec.utils._behaviors import (
    ebnerd_from_path,
    create_binary_labels_column,
)
from utils.helper import COLUMNS, NRMSDataset

import polars as pl


def prepare_validation_data(hparams, PATH, DATASPLIT, article_mapping, validation_data_fraction=1):
    df_validation = (
        ebnerd_from_path(
            PATH.joinpath(DATASPLIT, "validation"),
            history_size=hparams.history_size,
            padding=0,
        )
        .select(COLUMNS)
        .pipe(create_binary_labels_column)
        .sample(fraction=validation_data_fraction)
    )

    val_dataset = NRMSDataset(
        df_validation, article_mapping, hparams.title_size)
    val_loader = val_dataset.get_dataloader(
        hparams.batch_size)

    print(f" -> Validation samples: {df_validation.height}")
    return val_loader


def prepare_test_data(hparams, PATH, article_mapping, test_data_fraction=1):
    df_test = (
        ebnerd_from_path(
            PATH,
            history_size=hparams.history_size,
            padding=0,
        )
        .sample(fraction=test_data_fraction)
        .with_columns(
            pl.col(DEFAULT_INVIEW_ARTICLES_COL)
            .list.first()
            .alias(DEFAULT_CLICKED_ARTICLES_COL)
        )
        .select(COLUMNS + [DEFAULT_IS_BEYOND_ACCURACY_COL])
        .with_columns(
            pl.col(DEFAULT_INVIEW_ARTICLES_COL)
            .list.eval(pl.element() * 0)
            .alias(DEFAULT_LABELS_COL)
        )
    )

    df_test_wo_beyond = df_test.filter(~pl.col(DEFAULT_IS_BEYOND_ACCURACY_COL))
    df_test_w_beyond = df_test.filter(pl.col(DEFAULT_IS_BEYOND_ACCURACY_COL))

    # Create test Dataset and Dataloader
    dataset_wo_beyond = NRMSDataset(
        df_test_wo_beyond, article_mapping, hparams.title_size
    )
    loader_wo_beyond = dataset_wo_beyond.get_dataloader(
        hparams.batch_size)

    dataset_w_beyond = NRMSDataset(
        df_test_w_beyond, article_mapping, hparams.title_size
    )
    loader_w_beyond = dataset_w_beyond.get_dataloader(
        hparams.batch_size)

    print(
        f" -> Testing samples: {df_test.height}")
    return df_test_wo_beyond, df_test_w_beyond, loader_wo_beyond, loader_w_beyond


def save_ranked_scores(df_test, scores, evaluation_dir, eval_name):
    """
    Save ranked scores for given test data and scores.

    Args:
                    df_test (pl.DataFrame): Test DataFrame containing the impression ID column.
                    scores (list): List of scores to be ranked.
                    checkpoint_dir (str): Path to the checkpoint directory.
                    eval_name (str): Name for the evaluation file (e.g., "wo_ba", "w_beyond").
    """
    # Rank predictions by score
    ranked_scores = [list(rank_predictions_by_score(score))
                     for score in scores]

    # Add ranked scores to the test DataFrame
    df_pred_test = df_test.select(DEFAULT_IMPRESSION_ID_COL, DEFAULT_LABELS_COL).with_columns(
        pl.Series("scores", scores),
        pl.Series("ranked_scores", ranked_scores)
    )

    # Write to a parquet file
    df_pred_test.write_parquet(
        evaluation_dir.joinpath(f"pred_{eval_name}.parquet")
    )

    return df_pred_test

import os
from copy import deepcopy

import pandas as pd
from loguru import logger


def target_statistics(data: pd.DataFrame, target_label: str):
    target = data[target_label]
    if target.nunique() == 2:  # binary target -> classification
        ratio = (target.sum() / len(target.index)).round(2)
        logger.info(
            f'\nSummary statistics for binary target variable {target_label}:\n'
            f'Positive class makes up {target.sum()} samples out of {len(target.index)}, i.e. {ratio*100}%.'
        )
        return 'classification', target  # stratify w.r.t. target classes
    else:  # continous target -> regression
        logger.info(
            f'\nSummary statistics for continuous target variable {target_label}:\n'
            f'{target.describe(percentiles=[]).round(2)}'
        )
        return 'regression', None  # do not stratify for regression task


def save_tables(out_dir, experiment_name, tables) -> None:
    """Save tables to excel file"""
    file_path = os.path.join(out_dir, f'{experiment_name}.xlsx')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tables.to_excel(file_path, index=False)


def split_data(data: pd.DataFrame, metadata: list, hue: str, remove_mdata: bool = True):
    """Split data into data to analyse and hue data"""
    to_analyse = deepcopy(data)
    hue_df = to_analyse[hue]
    if remove_mdata:
        to_analyse = to_analyse.drop(metadata, axis=1, errors='ignore')
    suffix = 'no_mdata' if remove_mdata else 'with_mdata'
    return to_analyse, hue_df, suffix

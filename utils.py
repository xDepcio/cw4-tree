from typing import List

import numpy as np
import pandas as pd


def load_data_frame_u(
    path: str,
    skiprows: int = 0,
    cut_cols: List[str] = [],
) -> pd.DataFrame:
    data = pd.read_csv(
        path, skiprows=skiprows, header=None, names=None, index_col=False
    )

    for col in cut_cols:
        data = data.drop(col, axis=1)
    return data


def split_train_data_u(
    data: pd.DataFrame, train_size: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for row in data.iloc:
        if np.random.rand() < train_size:
            train_data = pd.concat([train_data, pd.DataFrame([row])])
        else:
            test_data = pd.concat([test_data, pd.DataFrame([row])])

    return train_data, test_data


def calculate_entropy_u(data: pd.DataFrame):
    class_col = data.iloc[:, 0]
    entropy = 0
    for unique_class_val in np.unique(class_col):
        probability = np.sum(class_col == unique_class_val) / len(class_col)
        entropy -= probability * np.log2(probability)
    return entropy

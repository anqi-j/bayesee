import numpy as np
import pandas as pd
from bayesee.validation import *


def independent_bin(df, bin_columns, n_bins):
    df_binned = df.copy()

    if is_number(n_bins):
        n_bins = np.ones((len(bin_columns),), dtype=int) * n_bins

    for index_bin, bin_column in enumerate(bin_columns):
        n_bin = n_bins[index_bin]
        df_binned[f"bin_{bin_column}"] = pd.qcut(
            df[bin_column], n_bin, labels=range(1, n_bin + 1)
        )

    return df_binned


def recursive_bin(df, bin_columns, n_bins, level_counter=1):
    if len(bin_columns) == 1:
        return df

    df_binned = df.copy()
    current_feature = bin_columns[0]
    next_features = bin_columns[1:]
    current_n_bin = n_bins[0]
    next_n_bins = n_bins[1:]

    current_bin_name = f"level{level_counter}_bin_{current_feature}"
    next_bin_name = f"level{level_counter+1}_bin_{next_features[0]}"

    if current_bin_name not in df.columns:
        df_binned[current_bin_name] = pd.qcut(
            df[current_feature],
            current_n_bin,
            labels=range(1, 1 + current_n_bin),
        )

    unique_bins = df_binned[current_bin_name].unique()
    bin_counter = 1

    df_binned[next_bin_name] = None
    for index_bin in unique_bins:
        bool_array_bin = df_binned[current_bin_name] == index_bin

        df_binned.loc[bool_array_bin, next_bin_name] = pd.qcut(
            df.loc[bool_array_bin, next_features[0]],
            next_n_bins[0],
            labels=range(bin_counter, bin_counter + next_n_bins[0]),
        )

        bin_counter += next_n_bins[0]

    level_counter += 1

    df_binned = recursive_bin(df_binned, next_features, next_n_bins, level_counter)

    return df_binned

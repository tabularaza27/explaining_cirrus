import pandas as pd
import numpy as np


def create_filter_string(filters):
    # add brackets
    filters = ["({})".format(f) for f in filters]
    filter_str = " & ".join(filters)
    return filter_str


def filter_df(df, filters):
    """

    Args:
        df (pd.DataFrame): the complete dataframe
        filters (list): list of filter expressions. e.g. ["clm == 1", "instrument_flag == 3"]

    Returns:
        pd.DataFrame: filtered dataframe
    """
    filter_str = create_filter_string(filters)
    f_df = df.query(filter_str)

    return f_df


def oh_encoding(df, column_names, drop_original=True):
    """create one hot encoding for given columns

    Args:
        df (pd.DataFrame):
        column_names (list):
        drop_original (bool): if True, drop original column

    Returns:
        pd.DataFrame: dataframe with one hot encoding for given column. i.e. on column per feature value

    """
    for column_name in column_names:
        oh_df = pd.get_dummies(df[column_name], prefix=column_name)
        df = pd.concat([df, oh_df], axis=1)
        if drop_original:
            df = df.drop(column_name, axis=1)

    return df


def kickout_outliers(df, predictand, iqr_multiple=1.5):
    """returns dataframe removed from outliers

    Args:
        df:
        predictand (str): predictand where outlier calculation is based on
        iqr_multiple (float): inter quartile range multiple. values outside iqr * multiple are considered outliers

    Returns:

    """
    q25 = df[predictand].quantile(0.25)
    q75 = df[predictand].quantile(0.75)

    iqr = q75 - q25
    cut_off = iqr * iqr_multiple
    lower, upper = q25 - cut_off, q75 + cut_off

    cutoff_idx = ~((df[predictand] < lower) | (df[predictand] > upper))
    n_outliers = cutoff_idx[~cutoff_idx].count()
    outlier_cleaned_df = df[cutoff_idx]

    print("Original df: {} datapoints".format(df[df.columns[0]].count()))
    print("kickout {} outliers".format(n_outliers))
    print("New df: {} datapoints".format(outlier_cleaned_df[outlier_cleaned_df.columns[0]].count()))

    return outlier_cleaned_df


def log_transform(df, column_names, zero_handling="add_constant", drop_original=False):
    """log transforms given columns of dataframe

    Args:
        df (pd.DataFrame):
        column_names (list):
        zero_handling (str): strategy to handle zero values, either `drop` or `add_constant`
        drop_original (bool): if True, drop original column

    Returns:
        pd.DataFrame
    """
    assert zero_handling in ["add_constant", "drop"]

    for col in column_names:

        if df.query("{} == 0".format(col))[col].count() > 0:
            print("{} contains zero values")
            if zero_handling == "add_constant":
                df["{}_log".format(col)] = (df[col] + 1e-25).transform(np.log)
            elif zero_handling == "drop":
                df = df.query("{} > 0".format(col))
                df["{}_log".format(col)] = (df[col]).transform(np.log)

        elif df.query("{} < 0".format(col))[col].count() > 0:
            raise ValueError("{} contains negative values values")

        else:
            print("{} contains positive values only".format(col))
            df["{}_log".format(col)] = (df[col]).transform(np.log)

        if drop_original:
            df.drop(col, axis=1, inplace=True)

    return df

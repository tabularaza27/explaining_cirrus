import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

CAT_VARS = ["season","lat_region","IC_CIR","clm_v2",'nightday_flag','land_water_mask','instrument_flag']
LOG_TRANS_VARS = ['DU',"SO4", 'DU001','DU002','DU003','DU004','DU005']
BASE_PREDICTORS = [ 't', 'w', 'u', 'v', 'rh_ice','SO4','season','lat_region','dz_top',"IC_CIR",] # DU, clm_v2

# other predictor variables: 'DU001','DU002','DU003','DU004','DU005','DU','clm_v2', 'nightday_flag','land_water_mask','instrument_flag'

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


def log_transform(df, column_names, zero_handling="add_constant", drop_original=True):
    """log transforms given columns of dataframe

    Args:
        df (pd.DataFrame):
        column_names (list):
        zero_handling (str): strategy to handle zero values, either `drop`, `add_constant`, `error`
        drop_original (bool): if True, drop original column

    Returns:
        pd.DataFrame
    """
    assert zero_handling in ["add_constant", "drop", "error"]

    for col in column_names:

        if df.query("{} == 0".format(col))[col].count() > 0:
            print("{} contains zero values".format(col))
            if zero_handling == "add_constant":
                df["{}_log".format(col)] = (df[col] + 1e-25).transform(np.log)
            elif zero_handling == "drop":
                df = df.query("{} > 0".format(col))
                df["{}_log".format(col)] = (df[col]).transform(np.log)
            elif zero_handling == "error":
                raise ValueError("zero values not allowed for this variable")

        elif df.query("{} < 0".format(col))[col].count() > 0:
            raise ValueError("{} contains negative values values")

        else:
            print("{} contains positive values only".format(col))
            df["{}_log".format(col)] = (df[col]).transform(np.log)

        if drop_original:
            df.drop(col, axis=1, inplace=True)

    return df


def select_columns(df, predictors, predictand, add_grid_cell=True):
    sel = predictors + BASE_PREDICTORS
    sel.append(predictand)
    if add_grid_cell:
        sel.append("grid_cell")
    df = df[sel]

    return df


def run_preprocessing_steps(df, preproc_steps, predictand):
    # outliers
    if preproc_steps["kickout_outliers"]:
        df = kickout_outliers(df, predictand)

    # log transforms
    if preproc_steps["x_log_trans"]:
        x_log_vars = [var for var in df.columns if var in LOG_TRANS_VARS]
        df = log_transform(df, x_log_vars, zero_handling="add_constant")

    if preproc_steps["y_log_trans"]:
        df = log_transform(df, [predictand], zero_handling="error")

    # oh encoding
    if preproc_steps["oh_encoding"]:
        oh_vars = [var for var in df.columns if var in CAT_VARS]
        df = oh_encoding(df, oh_vars)

    return df


def split_train_test(df, predictand, random_state, test_size=0.2):
    X = df.drop(predictand, 1)
    y = df[predictand]

    ### split train / test data
    unique_gridcell = X["grid_cell"].unique()
    train_cells, test_cells = train_test_split(unique_gridcell, test_size=test_size, random_state=random_state)
    X_train, X_test = X[X.grid_cell.isin(train_cells)], X[X.grid_cell.isin(test_cells)]
    y_train, y_test = y[y.index.isin(X_train.index)], y[y.index.isin(X_test.index)]
    X_train.drop("grid_cell", inplace=True, axis=1)
    X_test.drop("grid_cell", inplace=True, axis=1)

    return X_train, X_test, y_train, y_test


def create_dataset(df, filters, predictors, predictand, preproc_steps, random_state=123):
    # filter dataset on conditions
    df = filter_df(df, filters)
    # select columns
    df = select_columns(df, predictors, predictand)
    # pre processing steps #
    df = run_preprocessing_steps(df, preproc_steps, predictand)

    # split train / test data set
    if preproc_steps["y_log_trans"]:
        predictand = "{}_log".format(predictand)
    X_train, X_test, y_train, y_test = split_train_test(df, predictand, random_state)

    return X_train, X_test, y_train, y_test
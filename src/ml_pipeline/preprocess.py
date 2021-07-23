import pandas as pd


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


def oh_encoding(df, column_name):
    """create one hot encoding for given column

    Args:
        df (pd.DataFrame):
        column_name (str):

    Returns:
        pd.DataFrame: dataframe with one hot encoding for given column. i.e. on column per feature value

    """
    oh_df = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, oh_df], axis=1)
    df = df.drop(column_name, axis=1)

    return df

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

from src.ml_pipeline.ml_preprocess import create_filter_string


### general ###

def filter_temporal_df(df: pd.DataFrame, filters: list):
    """Filters trajectory dataframe on conditions at trajectory endpoints

    filters are applied only on timestep==0, i.e. where I have satellite observations
    i.e. filter on trajectory ids

    Args:
        df (pd.DataFrame): the complete dataframe
        filters (list): list of filter expressions. e.g. ["liquid_origin == 1", "instrument_flag == 3"]

    Returns:
        pd.DataFrame: filtered dataframe
    """
    filters.append("timestep==0")
    filter_str = create_filter_string(filters)
    filtered_trajectory_ids = df.query(filter_str).trajectory_id.unique()
    filtered_df = df[df.trajectory_id.isin(filtered_trajectory_ids)].reset_index(drop=True)

    return filtered_df


### deep imbalanced regression helpers ###

def get_lds_kernel_window(kernel, ks, sigma):
    """todo"""
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

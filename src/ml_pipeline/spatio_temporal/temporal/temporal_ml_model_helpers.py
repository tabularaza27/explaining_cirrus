import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.special import exp10

from src.ml_pipeline.ml_preprocess import create_filter_string


### general ###

def filter_temporal_df(df: pd.DataFrame, filters: list, drop_nan_rows: bool = True):
    """Filters trajectory dataframe on conditions at trajectory endpoints

    filters are applied only on timestep==0, i.e. where I have satellite observations
    i.e. filter on trajectory ids

    Args:
        df (pd.DataFrame): the complete dataframe
        filters (list): list of filter expressions. e.g. ["liquid_origin == 1", "instrument_flag == 3"]
        drop_nan_rows (bool): if True, dropping trajectories with nan value in any variable at trajectory end point

    Returns:
        pd.DataFrame: filtered dataframe
    """
    filter_str = create_filter_string(filters)
    start_point_df = df.query("timestep==0")
    filtered_trajectory_ids = start_point_df.query(filter_str).trajectory_id.unique() # trajectories that will be kept

    if drop_nan_rows:
        nan_df = df.query("timestep==0")[df.query("timestep==0").isna().any(axis=1)] # checks for nan in all columns
        nan_counts = nan_df.isna().sum()
        print("nan counts per variable:")
        print(nan_counts[nan_counts > 0])
        nan_trajectory_id = nan_df.trajectory_id.unique() # trajectories that will be kicked out
        df = df[~df.trajectory_id.isin(nan_trajectory_id)].reset_index(drop=True)
        print("dropped trajectories with nan variables at timestep==0")

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


### plotting / logging ###

def create_ditribution_figures(predictand: str, y: np.ndarray, y_hat: np.ndarray,
                               log_scale=False) -> plt.Figure:
    """create distribution (true vs. predicted) and residual plot for one predictand

    Args:
        predictand:
        y:
        y_hat:
        log_scale:

    Returns:

    """
    residuals = y - y_hat

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    # predictand vs. ground truth distributions
    axs[0, 0].hist([y_hat, y], bins=100, alpha=0.5, density=True)
    axs[0, 0].set_xlabel("{}".format(predictand))
    axs[0, 0].set_ylabel("density")
    axs[0, 0].legend(["predicted", "ground_truth"])
    axs[0, 0].set_title(f"{predictand} distribution")

    # residuals
    axs[1, 0].hist(residuals, bins=100, density=True)
    axs[1, 0].set_xlabel("residuals")
    axs[1, 0].set_ylabel("density")
    axs[1, 0].set_title("residuals")

    if log_scale:
        # inverse log10 transform transform back to original scale
        y_org = exp10(y)
        y_hat_org = exp10(y_hat)
        residuals_org = y_org - y_hat_org

        # get percentile for zooming in on axis
        percentile = 99
        y_org_percentile = np.percentile(y_org, percentile)
        residuals_org_percentile = np.percentile(residuals_org, percentile)
        residuals_org_low_percentile = np.percentile(residuals_org, 100 - percentile)

        # predictand vs. ground truth (original scale)
        axs[0, 1].hist([y_hat_org, y_org], bins=1000, alpha=0.5, density=True)
        axs[0, 1].set_xlabel(predictand)
        axs[0, 1].set_ylabel("density")
        axs[0, 1].legend(["predicted", "ground_truth"])
        axs[0, 1].set_title(f"{predictand} distribution (original scale)")
        axs[0, 1].set_xlim([0, y_org_percentile])

        # residuals (original scale)
        axs[1, 1].hist(residuals_org, bins=1000, density=True)
        axs[1, 1].set_xlabel("residuals")
        axs[1, 1].set_ylabel("density")
        axs[1, 1].set_title("residuals (original scale)")
        axs[1, 1].set_xlim([residuals_org_low_percentile, residuals_org_percentile])

    plt.tight_layout()
    plt.show()

    return fig

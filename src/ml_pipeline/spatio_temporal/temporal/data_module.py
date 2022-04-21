import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from scipy.ndimage import convolve1d

import torch
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl

from src.ml_pipeline.ml_preprocess import split_train_val_test, log_transform
from src.ml_pipeline.ml_preprocess import CAT_VARS
from src.ml_pipeline.spatio_temporal.temporal.temporal_ml_model_helpers import *


class BacktrajDataset(Dataset):
    def __init__(self,
                 X_seq: np.ndarray,
                 X_static: np.ndarray,
                 y: np.ndarray,
                 coords: np.ndarray,
                 reweight: str = 'none',
                 reweight_bin_width: int = 10,
                 lds: bool = False,
                 lds_kernel: str = "gaussian",
                 lds_ks: int = 5,
                 lds_sigma: int = 2
                 ):
        """

        Args:
            X_seq: n_samples x n_timesteps x n_sequential_features
            X_static: n_samples x x_static_features
            y: n_samples x n_target_variables
            coords: n_samples x 4 → (time, lev, lat, lon)
            reweight:
            reweight_bin_width:
            lds:
            lds_kernel:
            lds_ks:
            lds_sigma:
        """

        # validity check
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        self.X_seq = torch.tensor(X_seq).float()
        self.X_static = torch.tensor(X_static).float()
        self.y = torch.tensor(y).float()
        self.coords = torch.tensor(coords).float()
        self.weights = self._prepare_weights()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index, :, :], self.X_static[index], self.y[index], self.coords[index]

    def _prepare_weights(self):

        if self.reweight == "none":
            return None

        decimals = int(np.log10(self.bin_width))
        y_rounded = np.round(self.y.cpu().numpy(), decimals)

        bins, bins_frequency = np.unique(y_rounded, return_counts=True)

        bin_dict = {bin: frequency for bin, frequency in zip(bins, bins_frequency)}

        # frequency per bin
        if self.reweight == 'sqrt_inv':
            bin_dict = {k: np.sqrt(v) for k, v in bin_dict.items()}
        elif self.reweight == 'inverse':
            bin_dict = {k: np.clip(v, 5, 1000) for k, v in bin_dict.items()}  # clip weights for inverse re-weight
        num_per_label = np.vectorize(bin_dict.get)(y_rounded)

        if self.lds:
            lds_kernel_window = get_lds_kernel_window(self.lds_kernel, self.lds_ks, self.lds_sigma)
            print(f'Using LDS: [{self.lds_kernel.upper()}] ({self.lds_ks}/{self.lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in bin_dict.items()]), weights=lds_kernel_window, mode='constant')
            smoothed_value_dict = {key: val for key, val in zip(bin_dict.keys(), smoothed_value)}
            num_per_label = np.vectorize(smoothed_value_dict.get)(y_rounded)

        # weights: inverse of occurance frequency
        weights = 1 / num_per_label

        scaling = weights.shape[0] / np.sum(weights)

        weights *= scaling

        return weights


class BacktrajDataModule(pl.LightningDataModule):
    '''
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading
      and processing work in one place.
    '''

    def __init__(self,
                 traj_df: pd.DataFrame,
                 data_filters: list = ["cloud_cover>0.9"],
                 sequential_features: list = ["p", "GPH", "T", "Q", "U", "V", "OMEGA", "o3", "RH_ice"],
                 static_features: list = ['land_water_mask', 'season', 'nightday_flag'],
                 predictands: list = ["iwc"],
                 log_transform_predictands: list = ["iwc", "icnc_5um", "icnc_100um"],
                 sequential_scaler: TransformerMixin = StandardScaler(),
                 static_scaler: TransformerMixin = StandardScaler(),
                 batch_size: int = 128,
                 train_size: float = 0.8,
                 num_workers: int = 0,
                 regional_feature_resolution=10,
                 reweight: str = 'none',
                 reweight_bin_width: int = 10,
                 lds: bool = False,
                 lds_kernel: str = "gaussian",
                 lds_ks: int = 5,
                 lds_sigma: int = 2):
        """

        Args:
            traj_df:
            data_filters:
            sequential_features:
            static_features:
            predictands:
            sequential_scaler:
            batch_size:
            train_size:
            num_workers:
            regional_feature_resolution: in degrees. if None, no regional feature is used
            reweight:
            reweight_bin_width:
            lds:
            lds_kernel:
            lds_ks:
            lds_sigma:
        """
        super().__init__()

        ### init df, features and predictands ###

        self.traj_df = traj_df
        self.data_filters = data_filters
        self.predictands = predictands
        self.sequential_features = sequential_features
        self.static_features = static_features
        self.static_df = pd.DataFrame()

        ### init placeholder variables for train/val/test splits ###

        # train
        self.df_train = None  # pd.DataFrame
        self.train_traj_ids = None  # np.ndarray
        self.X_train_sequential = None  # np.ndarray
        self.X_train_static = None  # np.ndarray
        self.y_train = None  # np.ndarray
        self.coords_train = None  # np.ndarray

        # val
        self.df_val = None
        self.val_traj_ids = None
        self.X_val_sequential = None
        self.X_val_static = None
        self.y_val = None
        self.coords_val = None

        # test
        self.df_test = None
        self.test_traj_ids = None
        self.X_test_sequential = None
        self.X_test_static = None
        self.y_test = None
        self.coords_test = None

        ### hparams ###

        # model level

        self.batch_size = batch_size
        self.train_size = train_size
        self.num_workers = num_workers

        # data preproc
        self.sequential_scaler = sequential_scaler
        self.static_scaler = static_scaler
        self.log_transform_predictands = log_transform_predictands
        self.regional_feature_resolution = regional_feature_resolution

        # deep imbalanced regression
        self.reweight = reweight
        self.reweight_bin_width = reweight_bin_width
        self.lds = lds
        self.lds_kernel = lds_kernel
        self.lds_ks = lds_ks
        self.lds_sigma = lds_sigma

        ### helpers ###

        self.coords = ["std_time", "lev", "lat", "lon"]
        self.columns = None  # todo delete ?
        self.preprocessing = None  # todo delete ?

        # todo preproc for static features
        # todo add data filters
        # todo train/val/test loaders
        # todo init variables
        # todo save coords
        # todo save y_preds if multitask, i.e. is it working with current setup
        #

    def prepare_data(self):
        """
        see info here: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html?highlight=prepare_data#prepare-data

        todo double check when this is called, do I need if/else based on stage → I think no for now
        """
        if "grid_cell" not in self.traj_df.columns:
            self.traj_df["grid_cell"] = self.traj_df["date"].astype("str") + self.traj_df["lat"].astype("str") + \
                                        self.traj_df["lon"].astype("str")

        # filter dataframe
        self.traj_df = filter_temporal_df(self.traj_df, self.data_filters)

        # set dtypes for static features
        self.traj_df[self.static_features] = self.traj_df[
            self.static_features].convert_dtypes()  # converts to best possible datatypes

        # add lat/lon region feature
        if self.regional_feature_resolution is not None:
            self.traj_df["lat_region"] = (np.round(
                self.traj_df.lat * (1 / self.regional_feature_resolution)) * self.regional_feature_resolution).astype(
                'int')
            self.traj_df["lon_region"] = (np.round(
                self.traj_df.lon * (1 / self.regional_feature_resolution)) * self.regional_feature_resolution).astype(
                'int')
            self.sequential_features.append("lat_region")
            self.sequential_features.append("lon_region")

        # one hot encoding of categorical variables (both sequential and static)
        oh_features = [col for col in self.static_features + self.sequential_features if col in CAT_VARS]
        for feature in oh_features:
            oh_df = pd.get_dummies(self.traj_df[feature], prefix=feature)

            for col in oh_df.columns:
                # add oh_encoded_features to feature lists and df
                if feature in self.sequential_features:
                    self.sequential_features.append(col)
                    self.sequential_features.remove(feature)
                else:
                    self.static_features.append(col)
                    self.static_features.remove(feature)

                # add oh encoded features to df
                self.traj_df[col] = oh_df[col]

            # remove original feature fro feature lists
            if feature in self.sequential_features:
                self.sequential_features.remove(feature)
            else:
                self.static_features.remove(feature)

        # todo remove I don't use it
        # create static df, i.e. data that is only available for timestep==0, i.e. predictands & static_features
        self.static_df = self.traj_df.query("timestep==0")[
            self.predictands + self.static_features + ["grid_cell", "trajectory_id"]]

        # todo kickout outliers on log transformed y data

    def setup(self, stage=None):
        '''
        '''

        if stage == 'fit' and self.X_train_sequential is not None:
            return
        if stage == 'test' and self.X_test_sequential is not None:
            return
        if stage is None and self.X_train_sequential is not None and self.X_test_sequential is not None:
            return

        ### create train/val/test split ###

        # only for selecting trajectory ids using function from xgboost preproc
        # get current global seed → it is set via pl seed_everything() in outer loop
        current_seed = np.random.get_state()[1][0]
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(df=self.traj_df.query("timestep==0"),
                                                                              predictand=self.predictands,
                                                                              random_state=current_seed,
                                                                              train_size=self.train_size)
        # save traj ids
        self.train_traj_ids = X_train.trajectory_id.unique()
        self.val_traj_ids = X_val.trajectory_id.unique()
        self.test_traj_ids = X_test.trajectory_id.unique()

        # create train/val/test dfs
        self.df_train = self.traj_df[self.traj_df.trajectory_id.isin(self.train_traj_ids)]
        self.df_val = self.traj_df[self.traj_df.trajectory_id.isin(self.val_traj_ids)]
        self.df_test = self.traj_df[self.traj_df.trajectory_id.isin(self.test_traj_ids)]

        # save coords (time, lev, lat, lon)
        self.coords_train = self.df_train.query("timestep==0")[self.coords].values
        self.coords_val = self.df_val.query("timestep==0")[self.coords].values
        self.coords_test = self.df_test.query("timestep==0")[self.coords].values

        # sequential features
        self.cont_sequential_features_list = [var for var in self.sequential_features if
                                              not any(map(var.startswith, CAT_VARS))]  # select only cont. features
        self.categorical_sequential_feature_list = [var for var in self.sequential_features if
                                                    any(map(var.startswith, CAT_VARS))]  # select only cat. features

        # static features
        self.cont_static_features_list = [var for var in self.static_features if not any(map(var.startswith, CAT_VARS))]
        self.categorical_static_features_list = [var for var in self.static_features if
                                                 any(map(var.startswith, CAT_VARS))]

        # init scalers for sequential and static features
        self.sequential_scaler.fit(self.df_train[self.scaling_sequential_features])
        self.static_scaler.fit(self.df_train.query("timestep==0")[self.cont_static_features_list])

        # create scaled np.ndarrays
        self.X_train_sequential = self.scale_and_create_x(self.df_train, "sequential")
        self.X_train_static = self.scale_and_create_x(self.df_train, "static")

        self.X_val_sequential = self.scale_and_create_x(self.df_val, "sequential")
        self.X_val_static = self.scale_and_create_x(self.df_val, "static")

        self.X_test_sequential = self.scale_and_create_x(self.df_test, "sequential")
        self.X_test_static = self.scale_and_create_x(self.df_test, "static")

        # create train/val/test arrays
        self.y_train = self.transform_and_create_y(self.df_train)
        self.y_val = self.transform_and_create_y(self.df_val)
        self.y_test = self.transform_and_create_y(self.df_test)

    def train_dataloader(self):
        train_dataset = BacktrajDataset(X_seq=self.X_train_sequential,
                                        X_static=self.X_train_static,
                                        y=self.y_train,
                                        coords=self.coords_train,
                                        reweight=self.reweight,
                                        reweight_bin_width=self.reweight_bin_width,
                                        lds=self.lds,
                                        lds_kernel=self.lds_kernel,
                                        lds_ks=self.lds_ks,
                                        lds_sigma=self.lds_sigma)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers)

        return train_loader

    def val_dataloader(self):
        val_dataset = BacktrajDataset(
            X_seq=self.X_val_sequential,
            X_static=self.X_val_static,
            y=self.y_val,
            coords=self.coords_val,
            reweight=self.reweight,
            reweight_bin_width=self.reweight_bin_width,
            lds=self.lds,
            lds_kernel=self.lds_kernel,
            lds_ks=self.lds_ks,
            lds_sigma=self.lds_sigma
        )
        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers)

        return val_loader

    def test_dataloader(self):
        test_dataset = BacktrajDataset(
            X_seq=self.X_test_sequential,
            X_static=self.X_test_static,
            y=self.y_test,
            coords=self.coords_test,
            reweight=self.reweight,
            reweight_bin_width=self.reweight_bin_width,
            lds=self.lds,
            lds_kernel=self.lds_kernel,
            lds_ks=self.lds_ks,
            lds_sigma=self.lds_sigma
        )
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers)

        return test_loader

    def scale_and_create_x(self, df, var_type):
        """scales continuous variables, concats with categorical data

        Args:
            df (pd.DataFrame): train/val/test dataframe
            var_type (str): sequential | static

        Returns: np.ndarray; sequential: (n_samples, n_timesteps, n_features); static: (n_samples, n_features)
        """
        if var_type == "sequential":
            X_cont = self.sequential_scaler.transform(df[self.cont_sequential_features_list]).reshape(
                int(df.shape[0] / 61), 61,
                len(self.cont_sequential_features_list))  # n_samples, # n_timesteps, # n_features
            X_cat = df[[self.categorical_sequential_feature_list]].values.reshape(int(df.shape[0] / 61), 61,
                                                                                  len(self.categorical_sequential_feature_list))

            X = np.concatenate((X_cont, X_cat), axis=2)
            X = np.flip(X, axis=1).copy()  # flip time so that last index is timestep 0, i.e end of trajectory

        elif var_type == "static":
            X_cont = self.static_scaler.transform(
                df.query("timestep==0")[self.cont_static_features_list])  # n_samples, # n_features
            X_cat = df.query("timestep==0")[[self.categorical_static_feature_list]].values

            X = np.concatenate((X_cont, X_cat), axis=1)

        else:
            raise ValueError("var_type needs to be sequential or static, is: {}".format(var_type))

        return X

    def transform_and_create_y(self, df):
        """applies given transformations on predictands and returns np.ndarray ready for train/val/test

        Args:
            df:

        Returns: np.ndarray

        """
        predictand_df = df.query("timestep==0")[self.predictands]
        # log transform predictands
        if len(self.log_transform_predictands) > 0:
            predictand_df = log_transform(df=self.predictand_df,
                                          column_names=self.log_transform_predictands,
                                          zero_handling="add_constant",
                                          drop_original=False)

            predictand_column_names = [p + "_log" if p in self.log_transform_predictands else p for p in self.predictands]

            predictand_df = predictand_df[predictand_column_names]

        y = predictand_df.values

        return y
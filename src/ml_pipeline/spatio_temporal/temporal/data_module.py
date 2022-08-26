import os
import pickle
from typing import Union

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
from src.preprocess.helpers.common_helpers import pd_dtime_to_std_seconds, std_seconds_to_pd_dtime

from src.preprocess.helpers.constants import *
from src.scaffolding.scaffolding import get_data_product_dir


class BacktrajDataset(Dataset):
    def __init__(self,
                 X_seq: np.ndarray,
                 X_static: np.ndarray,
                 y: np.ndarray,
                 coords: np.ndarray,
                 reweight: str = 'none',
                 multiple_predictand_reweight_type: str = "individual",
                 reweight_lead_predictand_idx: int = 0,
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
            reweight: 'none', 'inverse', 'sqrt_inv'
            multiple_predictand_reweight_type: "individual", "lead_predictand"
            reweight_lead_predictand_idx:  index of predictand that is used for calculating deep imbalanced regression weights
                                               only used if multiple_predictand_reweight_type=="lead_predicatne" and reweight!="none"
            reweight_bin_width:
            lds:
            lds_kernel:
            lds_ks:
            lds_sigma:
        """
        self.X_seq = torch.tensor(X_seq).float()
        self.X_static = torch.tensor(X_static).float()
        self.y = torch.tensor(y).float()
        self.coords = torch.tensor(coords).float()
        self.n_predictands = y.shape[1]

        if reweight == "none":
            self.weights = None
        elif self.n_predictands == 1:
            # only one predictor, no multi task learning
            self.weights = BacktrajDataset.prepare_weights(y, reweight=reweight, lds=lds, lds_kernel=lds_kernel,
                                                           lds_ks=lds_ks, lds_sigma=lds_sigma)
        elif multiple_predictand_reweight_type == "individual":
            # calculate weights for each predictand individually
            # weights has same shape as y → n_samples x n_predictors
            self.weights = np.array(
                [BacktrajDataset.prepare_weights(y_i, reweight=reweight, lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks,
                                                 lds_sigma=lds_sigma) for y_i in y.T]).T
        elif multiple_predictand_reweight_type == "lead_predictand":
            # calculate weights based on lead predictand and use same weights for every predictand
            lead_predictand_weights = BacktrajDataset.prepare_weights(y[:, reweight_lead_predictand_idx],
                                                                      reweight=reweight, lds=lds, lds_kernel=lds_kernel,
                                                                      lds_ks=lds_ks, lds_sigma=lds_sigma)
            lead_predictand_weights = np.expand_dims(lead_predictand_weights,
                                                     1)  # expand dimensionality → shape: n_samples x 1
            self.weights = np.repeat(lead_predictand_weights, repeats=self.n_predictands,
                                     axis=1)  # "copy" weights to use for each predictor → shape: n_samples x n_predictors
        else:
            raise ValueError(
                'multiple_predictand_reweight_type needs to be in ["individual", "lead_predictand"], is: {}'.format(
                    multiple_predictand_reweight_type))

    def __len__(self):
        return self.X_seq.shape[0]

    def __getitem__(self, index):
        # retrieve weights, if None, return array with ones
        weight = self.weights[index].astype('float32') if self.weights is not None else np.ones(shape=self.n_predictands).astype('float32')
        return self.X_seq[index, :, :], self.X_static[index], self.y[index], weight, self.coords[index]

    @staticmethod
    def prepare_weights(y: np.ndarray,
                        reweight: str,
                        bin_width: int = 10,
                        lds: bool = False,
                        lds_kernel: str = 'gaussian',
                        lds_ks: int = 5,
                        lds_sigma: int = 2):
        """calculate weights for one predictand

        Args:
            y: shape: (n_samples)
            reweight: 'none', 'inverse', 'sqrt_inv'
            bin_width:
            lds:
            lds_kernel:
            lds_ks:
            lds_sigma:

        Returns:

        """
        # validity check
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert reweight != 'none' if lds else True, \
            "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

        if reweight == "none":
            return None

        decimals = int(np.log10(bin_width))
        y_rounded = np.round(y, decimals)

        bins, bins_frequency = np.unique(y_rounded, return_counts=True)

        bin_dict = {bin: frequency for bin, frequency in zip(bins, bins_frequency)}

        # frequency per bin
        if reweight == 'sqrt_inv':
            bin_dict = {k: np.sqrt(v) for k, v in bin_dict.items()}
        elif reweight == 'inverse':
            bin_dict = {k: np.clip(v, 5, 1000) for k, v in bin_dict.items()}  # clip weights for inverse re-weight
        num_per_label = np.vectorize(bin_dict.get)(y_rounded)

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
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
                 traj_df: Union[pd.DataFrame, None],
                 preloaded_dataset_id: Union[str, None] = None,
                 inference_only: bool = False,
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
                 multiple_predictand_reweight_type: str = "individual",
                 reweight_lead_predictand: str = "iwc",
                 reweight_bin_width: int = 10,
                 lds: bool = False,
                 lds_kernel: str = "gaussian",
                 lds_ks: int = 5,
                 lds_sigma: int = 2):
        """

        Args:
            traj_df:
            preloaded_dataset_id: id of preprocesses dataset, loads npy arrays from disk instead of preprocessing in _prepare_data and setup()
            inference_only: If True and preloaded_dataset_id is given, only load test data
            data_filters:
            sequential_features:
            static_features:
            predictands:
            sequential_scaler:
            batch_size:
            train_size:
            num_workers:
            regional_feature_resolution: in degrees. if None, no regional feature is used
            reweight: 'none', 'inverse', 'sqrt_inv'
            multiple_predictand_reweight_type: "individual", "lead_predictand"
            reweight_lead_predictand: predictand that is used for calculating deep imbalanced regression weights
                                               only used if multiple_predictand_reweight_type=="lead_predicatne" and reweight!="none"
            reweight_bin_width:
            lds:
            lds_kernel:
            lds_ks:
            lds_sigma:

            todo implement loading only specific features when loading preloaded dataset
        """
        super().__init__()

        # todo assertions
        assert not (type(traj_df) == type(preloaded_dataset_id) == type(
            None)), "either traj_df or preload_data_filters can be None, pass either datafram or id"
        assert (type(traj_df) == type(None)) or (type(preloaded_dataset_id) == type(None)), "Pass either dataframe or " \
                                                                                            "preload dataset id, " \
                                                                                            "now passed both "
        ### init df, features and predictands ###

        self.traj_df = traj_df
        self.preloaded_dataset_id = preloaded_dataset_id
        self.inference_only = inference_only
        self.data_filters = data_filters
        self.predictands = predictands
        self.sequential_features = sequential_features
        self.static_features = static_features

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
        self.log_transform_predictands = [p for p in log_transform_predictands if p in predictands] # only transform predictands that are active
        self.regional_feature_resolution = regional_feature_resolution

        # deep imbalanced regression
        self.reweight = reweight
        self.reweight_bin_width = reweight_bin_width
        self.lds = lds
        self.lds_kernel = lds_kernel
        self.lds_ks = lds_ks
        self.lds_sigma = lds_sigma

        # for dir on multi task regression
        self.multiple_predictand_reweight_type = multiple_predictand_reweight_type
        self.reweight_lead_predictand = reweight_lead_predictand

        # get index for lead predicatand for use in BacktrajDataset
        if (self.multiple_predictand_reweight_type == "lead_predictand") and (self.reweight!="none") and (len(self.predictands)>1):
            self.reweight_lead_predictand_idx = self.predictands.index(self.reweight_lead_predictand)
        else:
            self.reweight_lead_predictand_idx = None

        ### helpers ###

        self.coords = ["std_time", "lev", "lat", "lon"]

        # feature name lists
        # sequential features
        self.cont_sequential_features_list = []
        self.categorical_sequential_feature_list = []
        # static features
        self.cont_static_features_list = []
        self.categorical_static_features_list = []

        ### call prepare data routine ###

        self._prepare_data()

        # todo save y_preds if multitask, i.e. is it working with current setup
        #

    def _prepare_data(self):
        """
        call this at the end of init, cause info about e.g. oh encoded features is needed for ml model initialization


        there is also a hook prepare_data() in pytorch lightning that is called before requesting the dataloaders
        see info here: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html?highlight=prepare_data#prepare-data
        """

        # preprocessing steps on dataframe, not necessary if preloaded dataset it used
        if self.traj_df is not None:
            # add grid_cell column if it doesnt exist yet
            if "grid_cell" not in self.traj_df.columns:
                self.traj_df["grid_cell"] = self.traj_df["date"].astype("str") + self.traj_df["lat"].astype("str") + \
                                            self.traj_df["lon"].astype("str")

            # create std time column (time of observation), necessary to create torch tensor, i.e. needs to be numeric
            if "std_time" not in self.traj_df.columns:
                self.traj_df["std_time"] = pd_dtime_to_std_seconds(self.traj_df["time"])

            # filter dataframe
            self.traj_df = filter_temporal_df(self.traj_df, self.data_filters, drop_nan_rows=True)

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
                    else:
                        self.static_features.append(col)

                    # add oh encoded features to df
                    self.traj_df[col] = oh_df[col]

                # remove original feature from feature lists
                if feature in self.sequential_features:
                    self.sequential_features.remove(feature)
                else:
                    self.static_features.remove(feature)

        # sequential features
        self.cont_sequential_features_list = [var for var in self.sequential_features if
                                              not any(
                                                  map(var.startswith, CAT_VARS))]  # select only cont. features
        self.categorical_sequential_feature_list = [var for var in self.sequential_features if
                                                    any(map(var.startswith,
                                                            CAT_VARS))]  # select only cat. features

        # static features
        self.cont_static_features_list = [var for var in self.static_features if
                                          not any(map(var.startswith, CAT_VARS))]
        self.categorical_static_features_list = [var for var in self.static_features if
                                                 any(map(var.startswith, CAT_VARS))]

        # todo kickout outliers on log transformed y data

    def _load_preprocessed_data(self, dataset_id):

        # make dynamic if I will use different config ids in the future
        config_id = "larger_domain_high_res"
        ml_dir = get_data_product_dir(config_id, ML_DATA_DIR)
        dataset_dir = os.path.join(ml_dir, dataset_id)


        # arrays to load from disk
        arr_to_load = ['X_test_sequential',
                       'X_test_static',
                       'X_train_sequential',
                       'X_train_static',
                       'X_val_sequential',
                       'X_val_static',
                       'y_test',
                       'y_train',
                       'y_val',
                       'coords_train',
                       'coords_val',
                       'coords_test']

        if self.inference_only:
            # load test data only
            arr_to_load = [arr for arr in arr_to_load if "test" in arr]

        for arr_name in arr_to_load:
            filename = os.path.join(dataset_dir, "{}.npy".format(arr_name.lower()))
            arr_vals = np.load(filename)
            setattr(self, arr_name, arr_vals)
            print("loaded", arr_name)
        # scalers to load from disk
        scalers_to_load = ["sequential_scaler", "static_scaler"]

        for scaler_name in scalers_to_load:
            fname = os.path.join(dataset_dir, "{}.pkl".format(scaler_name.lower()))

            with open(fname, "rb") as f:
                scaler = pickle.load(f)
                setattr(self, scaler_name, scaler)
                print("loaded", scaler_name)

    def setup(self, stage=None):
        '''preprocessing

        - split train/val/test
        - scale continuous features
        - concate oh encoded features with cont. features
        - create static feature arrays
        - create prediction arrays and log scale if required
        '''

        if stage == 'fit' and self.X_train_sequential is not None:
            return
        if stage == 'test' and self.X_test_sequential is not None:
            return
        if stage is None and self.X_train_sequential is not None and self.X_test_sequential is not None:
            return

        # preprocessing steps on dataframe, not necessary if preloaded dataset it used
        if self.traj_df is not None:

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

            # init scalers for sequential and static features
            self.sequential_scaler.fit(self.df_train[self.cont_sequential_features_list])
            if len(self.cont_static_features_list) > 0:
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
        else:
            # load preprocessed arrays
            self._load_preprocessed_data(dataset_id=self.preloaded_dataset_id)

    def train_dataloader(self):
        train_dataset = BacktrajDataset(X_seq=self.X_train_sequential,
                                        X_static=self.X_train_static,
                                        y=self.y_train,
                                        coords=self.coords_train,
                                        reweight=self.reweight,
                                        multiple_predictand_reweight_type=self.multiple_predictand_reweight_type,
                                        reweight_lead_predictand_idx=self.reweight_lead_predictand_idx,
                                        reweight_bin_width=self.reweight_bin_width,
                                        lds=self.lds,
                                        lds_kernel=self.lds_kernel,
                                        lds_ks=self.lds_ks,
                                        lds_sigma=self.lds_sigma)
        self.train_dataset = train_dataset
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
            reweight="none",
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
            reweight="none",
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
            X_cat = df[self.categorical_sequential_feature_list].values.reshape(int(df.shape[0] / 61), 61,
                                                                                len(self.categorical_sequential_feature_list))

            X = np.concatenate((X_cont, X_cat), axis=2)
            X = np.flip(X, axis=1).copy()  # flip time so that last index is timestep 0, i.e end of trajectory

        elif var_type == "static":
            if len(self.cont_static_features_list) > 0:
                X_cont = self.static_scaler.transform(
                    df.query("timestep==0")[self.cont_static_features_list])  # n_samples, # n_features
                X_cat = df.query("timestep==0")[self.categorical_static_features_list].values

                X = np.concatenate((X_cont, X_cat), axis=1)
            else:
                # not cont static features
                X = df.query("timestep==0")[self.categorical_static_features_list].values

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
            predictand_df = log_transform(df=predictand_df,
                                          column_names=self.log_transform_predictands,
                                          zero_handling="add_constant",
                                          drop_original=False)

            predictand_column_names = [p + "_log" if p in self.log_transform_predictands else p for p in
                                       self.predictands]

            predictand_df = predictand_df[predictand_column_names]

        y = predictand_df.values

        return y

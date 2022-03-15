import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

import torch
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl

from src.ml_pipeline.ml_preprocess import split_train_val_test


class BacktrajDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index, :, :], self.y[index])


class BacktrajDataModule(pl.LightningDataModule):
    '''
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading
      and processing work in one place.
    '''

    def __init__(self, traj_df, scaler=StandardScaler(), batch_size=128, num_workers=0,
                 features=["p", "GPH", "T", "Q", "U", "V", "OMEGA", "o3", "RH_ice"]):
        super().__init__()

        self.traj_df = traj_df
        self.scaler = scaler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.features = features
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.X_test = None
        self.columns = None
        self.preprocessing = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_traj_ids = None
        self.val_traj_ids = None
        self.test_traj_ids = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        '''
        Data is resampled to hourly intervals.
        Both 'np.nan' and '?' are converted to 'np.nan'
        'Date' and 'Time' columns are merged into 'dt' index
        '''

        if stage == 'fit' and self.X_train is not None:
            return
        if stage == 'test' and self.X_test is not None:
            return
        if stage is None and self.X_train is not None and self.X_test is not None:
            return

        self.traj_df["grid_cell"] = self.traj_df["date"].astype("str") + self.traj_df["lat"].astype("str") + \
                                    self.traj_df["lon"].astype("str")

        # filter for observations with high dardar cloud cover
        tids = self.traj_df.query("(timestep==0) & (cloud_cover>0.9)").trajectory_id.unique()
        self.traj_df = self.traj_df[self.traj_df.trajectory_id.isin(tids)]

        # only for selecting trajectory ids using function from xgboost preproc
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(self.traj_df.query("timestep==0"), "iwc",
                                                                              random_state=1, train_size=0.8)

        # save traj ids
        self.train_traj_ids = X_train.trajectory_id.unique()
        self.val_traj_ids = X_val.trajectory_id.unique()
        self.test_traj_ids = X_test.trajectory_id.unique()

        # create train/val/test dfs
        self.df_train = self.traj_df[self.traj_df.trajectory_id.isin(self.train_traj_ids)]
        self.df_val = self.traj_df[self.traj_df.trajectory_id.isin(self.val_traj_ids)]
        self.df_test = self.traj_df[self.traj_df.trajectory_id.isin(self.test_traj_ids)]

        preprocessing = StandardScaler()
        preprocessing.fit(self.df_train[self.features])

        # if stage == 'fit' or stage is None:
        self.X_train = preprocessing.transform(self.df_train[self.features]).reshape(int(self.df_train.shape[0] / 61),
                                                                                     61,
                                                                                     len(self.features))  # n_samples, n_timesteps, n_features
        self.X_train = np.flip(self.X_train, axis=1).copy()  # flip time so that last index is timestep 0, i.e end of
        #             self.y_train = self.y_train.values.reshape((-1, 1))
        self.X_val = preprocessing.transform(self.df_val[self.features]).reshape(int(self.df_val.shape[0] / 61), 61,
                                                                                 len(self.features))  # n_samples, n_timesteps, n_features
        self.X_val = np.flip(self.X_val, axis=1).copy()
        #             self.y_val = self.y_val.values.reshape((-1, 1))

        # if stage == 'test' or stage is None:
        self.X_test = preprocessing.transform(self.df_test[self.features]).reshape(int(self.df_test.shape[0] / 61), 61,
                                                                                   len(self.features))  # n_samples, n_timesteps, n_features
        self.X_test = np.flip(self.X_test, axis=1).copy()
        #             self.y_test = self.y_test.values.reshape((-1, 1))

        # create train/val/test arrays
        #         print(self.df_train.shape)
        #         self.X_train = self.df_train[self.features].values.reshape(int(self.df_train.shape[0]/61), 61, len(self.features)) # n_samples, n_timesteps, n_features
        self.y_train = self.df_train.query("timestep==0")["iwc"].values
        self.y_train = np.log10(self.y_train)

        #         self.X_val = self.df_val[self.features].values.reshape(int(self.df_val.shape[0]/61), 61, len(self.features)) # n_samples, n_timesteps, n_features
        self.y_val = self.df_val.query("timestep==0")["iwc"].values
        self.y_val = np.log10(self.y_val)

        #         self.X_test = self.df_test[self.features].values.reshape(int(self.df_test.shape[0]/61), 61, len(self.features)) # n_samples, n_timesteps, n_features
        self.y_test = self.df_test.query("timestep==0")["iwc"].values
        self.y_test = np.log10(self.y_test)

    def train_dataloader(self):
        train_dataset = BacktrajDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers)

        return train_loader

    def val_dataloader(self):
        val_dataset = BacktrajDataset(self.X_val, self.y_val)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers)

        return val_loader

    def test_dataloader(self):
        test_dataset = BacktrajDataset(self.X_test, self.y_test)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers)

        return test_loader

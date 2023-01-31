# Understanding cirrus clouds using explainable machine learning

This repo contains the code for replicating the research conducted in "Understanding cirrus clouds using explainable machine learning" by Jeggle et al.
If you have any question please reach out to me (kai.jeggle@env.ethz.ch)

## Structure

├── src  
    ├── config_files    
    ├── ml_pipeline  
    ├── preprocess 
    ├── scaffolding  

The code to train the machine learning models and apply the XAI methods is located in `src/ml_pipeline`. The other directories contain the code to create the co-located dataset of DARDAR-Nice, ERA5, and MERRA-2. 
To reproduce the results in the paper only the scripts in `src/ml_pipeline` are relevant. The co-located datasets can be downloaded here: 

## How to reproduce the results

Download the co-located datasets from: 

In the paper a XGBoost model is trained on an instantaneous data set and a LSTM+Attention network is trained on a temporal dataset.  
To train the models and the XAI pipeline run the code in the Notebooks `src/ml_pipeline/instantaneous/InstantaneousModel.ipynb` & `src/ml_pipeline/temporal/TemporalModel.ipynb`.
Further details are outlined as comments in the notebooks and the pyhton scripts that contain the program logic.

## Dependencies

bokeh                     2.4.2            py39h06a4308_1  
cartopy                   0.20.2           py39hc85cdae_3    conda-forge
cudatoolkit               11.3.1               h2bc3f7f_2  
dask                      2022.3.0           pyhd8ed1ab_0    conda-forge
dask-core                 2022.3.0           pyhd8ed1ab_0    conda-forge
datashader                0.13.0             pyh6c4a22f_0    conda-forge
holoviews                 1.14.8             pyhd8ed1ab_0    conda-forge
hvplot                    0.7.3                    pypi_0    pypi
matplotlib                3.5.0            py39h06a4308_0  
numba                     0.55.1           py39h56b8d98_0    conda-forge
numpy                     1.21.2           py39h20f2e39_0  
numpy-base                1.21.2           py39h79a1101_0  
nvidia-ml-py3             7.352.0                  pypi_0    pypi
pandas                    1.3.5            py39h8c16a72_0  
pytorch                   1.10.2          py3.9_cuda11.3_cudnn8.2.0_0    pytorch
pytorch-lightning         1.5.9              pyhd8ed1ab_0    conda-forge
pytorch-mutex             1.0                        cuda    pytorch
scikit-image              0.19.3                   pypi_0    pypi
scikit-learn              1.0.2            py39h51133e4_1  
scipy                     1.7.3            py39hc147768_0  
seaborn                   0.11.2             pyhd3eb1b0_0  
shapely                   1.8.0            py39ha65c37e_5    conda-forge
tensorboard               2.8.0              pyhd8ed1ab_1    conda-forge
tensorboard-data-server   0.6.0            py39h3da14fd_0    conda-forge
tensorboard-plugin-wit    1.8.1              pyhd8ed1ab_0    conda-forge
shap                      0.39.0           py39hde0f152_0    conda-forge
statsmodels               0.12.2                   pypi_0    pypi
xgboost                   1.4.0            py39hf3d152e_0    conda-forge

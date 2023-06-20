# Understanding cirrus clouds using explainable machine learning

This repo contains the code for replicating the research conducted in "Understanding cirrus clouds using explainable machine learning" by Jeggle et al.
If you have any question please reach out to me (kai.jeggle@env.ethz.ch)

## Structure

├── config_files    
├── ml_pipeline  
├── preprocess  
├── scaffolding  

All the code to train the machine learning models and apply the XAI methods is located in `src/ml_pipeline`. The other directories contain the code and config files to create the co-located dataset of DARDAR-Nice, ERA5, and MERRA-2. 
To reproduce the results in the paper only the scripts in `src/ml_pipeline` are relevant. The co-located datasets can be downloaded [Zenodo](https://zenodo.org/record/7965381)

## How to reproduce the results

**Download the co-located datasets from  [Zenodo](https://zenodo.org/record/7965381)**

In the paper a XGBoost model is trained on an instantaneous data set and a LSTM+Attention network is trained on a temporal dataset.  

To train the models and the XAI pipeline run the code in the Notebooks `src/ml_pipeline/instantaneous/InstantaneousModel.ipynb` & `src/ml_pipeline/temporal/TemporalModel.ipynb`.

Further details are outlined as comments in the notebooks and the pyhton scripts that contain the program logic.

## Dependencies

bokeh                     2.4.2              
cartopy                   0.20.2              
cudatoolkit               11.3.1             
dask                      2022.3.0             
dask-core                 2022.3.0             
datashader                0.13.0               
holoviews                 1.14.8               
hvplot                    0.7.3               
matplotlib                3.5.0              
numba                     0.55.1               
numpy                     1.21.2             
numpy-base                1.21.2             
nvidia-ml-py3             7.352.0             
pandas                    1.3.5              
pytorch                   1.10.2         
pytorch-lightning         1.5.9                
pytorch-mutex             1.0                  
scikit-image              0.19.3               
scikit-learn              1.0.2              
scipy                     1.7.3              
seaborn                   0.11.2             
shapely                   1.8.0                
tensorboard               2.8.0                
tensorboard-data-server   0.6.0                
tensorboard-plugin-wit    1.8.1                
shap                      0.39.0               
statsmodels               0.12.2               
xgboost                   1.4.0                

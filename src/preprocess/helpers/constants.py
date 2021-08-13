### physical constants ###
R = 287.058  # specific gas constant (for dry air)
g = 9.80665  # Standard gravity


### directories ###

## Reanalysis ##

# Merra
MERRA_INCOMING_DIR = '/net/n2o/wolke_scratch/kjeggle/MERRA2/incoming'
MERRA_PRE_PROC_DIR = "/net/n2o/wolke_scratch/kjeggle/MERRA2/preproc"
MERRA_REGRID_DIR = "/net/n2o/wolke_scratch/kjeggle/MERRA2/regrid"
MERRA_REGRID_FILESTUMPY = "merra2_regrid"
MERRA_CDO_SCRIPT_PATH = '/home/kjeggle/cirrus/src/preprocess/bash_scripts/merra_horizontal_remap.sh'

# ERA
ERA_INCOMING_DIR = '/net/n2o/wolke_scratch/kjeggle/ERA5/ECMWF_incoming'
ERA_CDO_SCRIPT_PATH = '/home/kjeggle/cirrus/src/preprocess/bash_scripts/era_preproc_single_file.sh'
ERA_PRE_PROC_DIR = '/net/n2o/wolke_scratch/kjeggle/ERA5/preproc'

## Dardar ##
DARDAR_INCOMING_DIR = "/net/n2o/wolke_scratch/kjeggle/DARDAR_NICE/DARNI_L2_PRO.v1.10"
DARDAR_GRIDDED_DIR = "/net/n2o/wolke_scratch/kjeggle/DARDAR_NICE/gridded/hourly"

## Datacube ##
DATA_CUBE_PRE_PROC_DIR = "/net/n2o/wolke_scratch/kjeggle/DATA_CUBE/pre_proc"
DATA_CUBE_PRE_PROC_FILESTUMPY = "data_cube_perproc"
DATA_CUBE_FILTERED_DIR = "/net/n2o/wolke_scratch/kjeggle/DATA_CUBE/filtered_cube"  # contains only entries with data mask true
DATA_CUBE_DF_DIR = "/net/n2o/wolke_scratch/kjeggle/DATA_CUBE/dataframes"  # 2d data frame with all ice cloud ovservations
DATA_CUBE_FEATURE_ENGINEERED_DF_DIR = "/net/n2o/wolke_scratch/kjeggle/DATA_CUBE/dataframes/feature_engineered"
DATA_ONLY_DF_FILESTUMPY = "ice_in_cloud_df"
OBSERVATIONS_DF_FILESTUMPY = "observations_df"

### misc ##
TEMP_THRES = 235.15 # cirrus cloud threshold
CLM_V2_ICE_CLOUD_MASKS = [1, 2, 9, 10]  # maybe add 10 → top of convective towers
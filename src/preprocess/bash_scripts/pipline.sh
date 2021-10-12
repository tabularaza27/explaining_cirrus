#!/bin/bash
CONFIG_ID=$1
/home/kjeggle/miniconda3/envs/cis_env/bin/python /home/kjeggle/cirrus/src/preprocess/dardar_nice_grid.py "2007-01-01" "2007-12-31" $CONFIG_ID 8
/home/kjeggle/miniconda3/envs/xarray/bin/python /home/kjeggle/cirrus/src/preprocess/parallelize_bash_scripts/era5_preproc_cdo.py 8 2007 $CONFIG_ID
/home/kjeggle/miniconda3/envs/xarray/bin/python /home/kjeggle/cirrus/src/preprocess/parallelize_bash_scripts/merra2_preproc_cdo.py 8 2007 $CONFIG_ID
/home/kjeggle/miniconda3/envs/xarray/bin/python /home/kjeggle/cirrus/src/preprocess/merge.py $CONFIG_ID 7 2007
#/home/kjeggle/miniconda3/envs/xarray/bin/python /home/kjeggle/cirrus/src/preprocess/data_cube_filters.py 2007 data # todo data cube filters has to be refactored




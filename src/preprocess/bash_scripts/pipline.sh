#!/bin/bash
CONFIG_ID=$1
YEAR=$2
WORKER=$3
/home/kjeggle/miniconda3/envs/xarray/bin/python /home/kjeggle/cirrus/src/preprocess/dardar_nice_grid.py "${YEAR}-01-01" "${YEAR}-12-31" "$CONFIG_ID" "$WORKER"
/home/kjeggle/miniconda3/envs/xarray/bin/python /home/kjeggle/cirrus/src/preprocess/parallelize_bash_scripts/era5_preproc_cdo.py "$WORKER" "$YEAR" "$CONFIG_ID"
/home/kjeggle/miniconda3/envs/xarray/bin/python /home/kjeggle/cirrus/src/preprocess/parallelize_bash_scripts/merra2_preproc_cdo.py "$WORKER" "$YEAR" "$CONFIG_ID"
/home/kjeggle/miniconda3/envs/xarray/bin/python /home/kjeggle/cirrus/src/preprocess/merge.py "$CONFIG_ID" "$WORKER" "$YEAR"
/home/kjeggle/miniconda3/envs/xarray/bin/python /home/kjeggle/cirrus/src/preprocess/data_cube_filters.py "$CONFIG_ID" "$YEAR" data "$WORKER"

#!/bin/bash
/home/kjeggle/miniconda3/envs/cis_env/python /home/kjeggle/cirrus/src/preprocess/dardar_nice_grid.py "2007-01-01" "2007-12-31" 14
/home/kjeggle/miniconda3/envs/xarray/python /home/kjeggle/cirrus/src/preprocess/merge.py 7 2007
/home/kjeggle/miniconda3/envs/xarray/python /home/kjeggle/cirrus/src/preprocess/data_cube_filters.py 2009
/home/kjeggle/miniconda3/envs/xarray/python /home/kjeggle/cirrus/src/preprocess/data_cube_filters.py 2007




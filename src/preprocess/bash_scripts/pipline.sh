#!/bin/bash

python /home/kjeggle/cirrus/src/preprocess/dardar_nice_grid.py "2007-01-01" "2007-12-31" 14
python /home/kjeggle/cirrus/src/preprocess/merge.py 7 2007
python /home/kjeggle/cirrus/src/preprocess/data_cube_filters.py 2009
python /home/kjeggle/cirrus/src/preprocess/data_cube_filters.py 2007




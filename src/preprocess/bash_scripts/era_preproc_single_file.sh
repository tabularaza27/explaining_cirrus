#!/bin/bash

# runs preprocessing steps for a single given filepath, check if processed file already exists needs to be done by outer
# scripts that calls this bash scrip

# filepath needs to be specified as first positional cmd argument when calling this file
filename=$1

# config_id needs to be specified as second positional cmd argument when calling this file
# config id specifies the horizontal resolution and directories
config_id=$2

# get directories for given config
Preproc_File_Directory=`python -c "from src.scaffolding.scaffolding import get_data_product_dir; from src.preprocess.helpers.constants import ERA_PRE_PROC_DIR; dir=get_data_product_dir('${config_id}', ERA_PRE_PROC_DIR); print(dir)"`
Intermediate_File_Directory=`python -c "from src.scaffolding.scaffolding import get_data_product_dir; from src.preprocess.helpers.constants import ERA_IM_DIR; dir=get_data_product_dir('${config_id}', ERA_IM_DIR); print(dir)"`
Template_Path=`python -c "from src.scaffolding.scaffolding import get_data_product_dir; from src.preprocess.helpers.constants import TEMPLATE_PATH; dir=get_data_product_dir('${config_id}', TEMPLATE_PATH); print(dir)"`
Config_File_Path=`python -c "from src.preprocess.helpers.constants import CONFIGS; print(CONFIGS)"`

# get horizontal extent from config file (configs.json) for given config_id
MIN_LON=`jq -r ".${config_id}.lonmin" $CONFIGS`
MAX_LON=`jq -r ".${config_id}.lonmax" $CONFIGS`
MIN_LAT=`jq -r ".${config_id}.latmin" $CONFIGS`
MAX_LAT=`jq -r ".${config_id}.latmax" $CONFIGS`

d=`echo $filename | grep -E -o '[0-9]{4}_[0-9]{2}_[0-9]{2}'`
t=`echo $filename | grep -E -o 'time_[0-9]{2}_[0-9]{2}_[0-9]{2}' | cut -f2- -d'_'`

echo "Start Processing $d $t"

FINAL_FILE=${Preproc_File_Directory}/all_era5_date_${d}_time_${t}.nc
if test -f "$FINAL_FILE"; then
  echo "Processed File already exists already exists."
  exit
fi

# 1. Rename r,q,t parameters (needed for afterburner), convert spectral to gaussian grid, set from  reduced gaussian to regular grid type
cdo -f nc -chparam,0.0.0,130,0.1.0,133,25.3.0,152 -sp2gpl -setgridtype,regular $filename ${Intermediate_File_Directory}/era5_date_${d}_time_${t}.nc

# 2. calculate rh with r,q,t via the cdo afterburner. Bash script Here document Limit string must be at beginning of line (https://tldp.org/LDP/abs/html/here-docs.html)
cdo after ${Intermediate_File_Directory}/era5_date_${d}_time_${t}.nc ${Intermediate_File_Directory}/rh_era5_date_${d}_time_${t}.nc << BREAK
    CODE=157 TYPE=20
BREAK

cdo chname,var157,rh ${Intermediate_File_Directory}/rh_era5_date_${d}_time_${t}.nc ${Intermediate_File_Directory}/rh_chname_era5_date_${d}_time_${t}.nc # change variable name
ncatted -a standard_name,rh,o,c,"relative_humidity" ${Intermediate_File_Directory}/rh_chname_era5_date_${d}_time_${t}.nc ${Intermediate_File_Directory}/rh_ncatted_era5_date_${d}_time_${t}.nc # set standard name according to cf con$

# 3. Interpolate to 0.25x0.25 grid
cdo -remapbil,$Template_Path ${Intermediate_File_Directory}/rh_ncatted_era5_date_${d}_time_${t}.nc ${Intermediate_File_Directory}/rh_remap_era5_date_${d}_time_${t}.nc # rh is extensive variable: conservative interp
cdo -remapbil,$Template_Path ${Intermediate_File_Directory}/era5_date_${d}_time_${t}.nc ${Intermediate_File_Directory}/remap_era5_date_${d}_time_${t}.nc # other variables are intensive variables:  bilinear interp

# 4. Select our target domain
cdo sellonlatbox,$MIN_LON,$MAX_LON,$MIN_LAT,$MAX_LAT ${Intermediate_File_Directory}/rh_remap_era5_date_${d}_time_${t}.nc ${Intermediate_File_Directory}/rh_sellatlon_era5_date_${d}_time_${t}.nc
cdo sellonlatbox,$MIN_LON,$MAX_LON,$MIN_LAT,$MAX_LAT ${Intermediate_File_Directory}/remap_era5_date_${d}_time_${t}.nc ${Intermediate_File_Directory}/sellatlon_era5_date_${d}_time_${t}.nc

# 5. Merge relative humidity and other variables
cdo merge ${Intermediate_File_Directory}/rh_sellatlon_era5_date_${d}_time_${t}.nc ${Intermediate_File_Directory}/sellatlon_era5_date_${d}_time_${t}.nc ${FINAL_FILE}

# 6. Delete intermediate files
rm  ${Intermediate_File_Directory}/*_date_${d}_time_${t}.nc
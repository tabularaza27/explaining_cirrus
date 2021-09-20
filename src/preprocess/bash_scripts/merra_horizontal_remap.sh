#!/bin/bash

# runs preprocessing steps for a single given filepath, check if processed file already exists needs to be done by outer
# scripts that calls this bash scrip

# filepath needs to be specified as first positional cmd argument when calling this file
filename=$1

# config_id needs to be specified as second positional cmd argument when calling this file
# config id specifies the horizontal resolution and directories
config_id=$2

# get directories for given config
Meteo_Directory=`python -c "from src.scaffolding.scaffolding import get_data_product_dir; from src.preprocess.helpers.constants import MERRA_METEO_INCOMING_DIR; dir=get_data_product_dir('${config_id}', MERRA_METEO_INCOMING_DIR); print(dir)"`
Intermediate_File_Directory=`python -c "from src.scaffolding.scaffolding import get_data_product_dir; from src.preprocess.helpers.constants import MERRA_IM_DIR; dir=get_data_product_dir('${config_id}', MERRA_IM_DIR); print(dir)"`
Preproc_File_Directory=`python -c "from src.scaffolding.scaffolding import get_data_product_dir; from src.preprocess.helpers.constants import MERRA_PRE_PROC_DIR; dir=get_data_product_dir('${config_id}', MERRA_PRE_PROC_DIR); print(dir)"`
Template_Path=`python -c "from src.scaffolding.scaffolding import get_abs_file_path; from src.preprocess.helpers.constants import TEMPLATE_PATH; path=get_abs_file_path('${config_id}', TEMPLATE_PATH); print(path)"`
Config_File_Path=`python -c "from src.preprocess.helpers.constants import CONFIGS; print(CONFIGS)"`

# get horizontal extent from config file (configs.json) for given config_id
MIN_LON=`jq -r ".${config_id}.lonmin" $Config_File_Path`
MAX_LON=`jq -r ".${config_id}.lonmax" $Config_File_Path`
MIN_LAT=`jq -r ".${config_id}.latmin" $Config_File_Path`
MAX_LAT=`jq -r ".${config_id}.latmax" $Config_File_Path`

d=`echo $filename | grep -E -o '[0-9]{8}'`

echo "Start Horizontal remapping $d"

FINAL_FILE=${Preproc_File_Directory}/all_merra2_date_${d}.nc
if test -f "$FINAL_FILE"; then
  echo "Processed File already exists already exists."
  exit
fi

# Merra files are called different based on year see https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf p.13
year=`echo $filename | grep -E -o '[0-9]{4}'| head -1`
if ((year<2011))
then
  meteo_file=${Meteo_Directory}/MERRA2_300.inst3_3d_asm_Nv.${d}.nc4.nc4
else
  meteo_file=${Meteo_Directory}/MERRA2_400.inst3_3d_asm_Nv.${d}.nc4.nc4
fi

# 1. remap horizontally
cdo remapbil,$Template_Path $meteo_file ${Intermediate_File_Directory}/remap_meteo_merra2_date_${d}.nc
cdo remapcon,$Template_Path $filename ${Intermediate_File_Directory}/remap_merra2_date_${d}.nc

# 2. select domain
#cdo sellonlatbox,$MIN_LON,$MAX_LON,$MIN_LAT,$MAX_LAT ${Intermediate_File_Directory}/remap_merra2_date_${d}.nc ${Intermediate_File_Directory}/sel_remap_merra2_date_${d}.nc
#cdo sellonlatbox,$MIN_LON,$MAX_LON,$MIN_LAT,$MAX_LAT ${Intermediate_File_Directory}/remap_meteo_merra2_date_${d}.nc ${Intermediate_File_Directory}/sel_remap_meteo_merra2_date_${d}.nc

# 3. Join aerosol data with temperature data
cdo merge ${Intermediate_File_Directory}/remap_merra2_date_${d}.nc ${Intermediate_File_Directory}/remap_meteo_merra2_date_${d}.nc $FINAL_FILE

# todo delete intermediate directories
# 6. Delete intermediate files
 rm  ${Intermediate_File_Directory}/*_date_${d}_time_${t}.nc
#!/bin/bash

# runs preprocessing steps for a single given filepath, check if processed file already exists needs to be done by outer
# scripts that calls this bash scrip

# filepath needs to be specified as first positional cmd argument when calling this file
filename=$1

MIN_LON=-75
MAX_LON=-15
MIN_LAT=0
MAX_LAT=60

Intermediate_File_Directory='/net/n2o/wolke_scratch/kjeggle/MERRA2/intermediate'
Preproc_File_Directory='/net/n2o/wolke_scratch/kjeggle/MERRA2/preproc'
Temperature_Directory='/net/n2o/wolke_scratch/kjeggle/MERRA2/temp_data/inst_incoming/'
Template_Path='/home/kjeggle/cirrus/src/config_files/template.nc'

d=`echo $filename | grep -E -o '[0-9]{8}'`

echo "Start Horizontal remapping $d"

FINAL_FILE=${Preproc_File_Directory}/all_merra2_date_${d}.nc
if test -f "$FINAL_FILE"; then
  echo "Processed File already exists already exists."
  exit
fi

temperature_file=${Temperature_Directory}/MERRA2_400.inst3_3d_asm_Nv.${d}.nc4.nc4

# 1. remap horizontally
cdo remapcon,$Template_Path $filename ${Intermediate_File_Directory}/remap_merra2_date_${d}.nc
cdo remapbil,$Template_Path $temperature_file ${Intermediate_File_Directory}/remap_temp_merra2_date_${d}.nc

# 2. Join aerosol data with temperature data
cdo merge ${Intermediate_File_Directory}/remap_merra2_date_${d}.nc ${Intermediate_File_Directory}/remap_temp_merra2_date_${d}.nc $FINAL_FILE

# 6. Delete intermediate files
# rm  ${Intermediate_File_Directory}/*_date_${d}_time_${t}.nc
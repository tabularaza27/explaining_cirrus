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
Meteo_Directory='/net/n2o/wolke_scratch/kjeggle/MERRA2/meteo_data/inst_incoming/' # temperature and geopotential height
Template_Path='/home/kjeggle/cirrus/src/config_files/template.nc'

d=`echo $filename | grep -E -o '[0-9]{8}'`

echo "Start Horizontal remapping $d"

FINAL_FILE=${Preproc_File_Directory}/all_merra2_date_${d}.nc
if test -f "$FINAL_FILE"; then
  echo "Processed File already exists already exists."
  exit
fi

# Merra files are called different based on year see https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf p.13
year=`echo $filename | grep -E -o '[0-9]{4}'`
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

# 6. Delete intermediate files
# rm  ${Intermediate_File_Directory}/*_date_${d}_time_${t}.nc
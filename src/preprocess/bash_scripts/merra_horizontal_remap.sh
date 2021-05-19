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
Grid_Spec_Path='/home/kjeggle/cirrus/src/preprocess/bash_scripts/gridspec'

d=`echo $filename | grep -E -o '[0-9]{8}'`

echo "Start Horizontal remapping $d"

FINAL_FILE=${Preproc_File_Directory}/all_merra2_date_${d}.nc
if test -f "$FINAL_FILE"; then
  echo "Processed File already exists already exists."
  exit
fi

temperature_file=${Temperature_Directory}/MERRA2_400.inst3_3d_asm_Nv.${d}.nc4.nc4

# 1. create template grid
cdo -f nc -sellonlatbox,-180,180,-90,90 -random,r1440x720 template.nc # create target grid
Grid_Spec_Path='/home/kjeggle/cirrus/src/preprocess/bash_scripts/gridspec'
cdo setgrid,${Grid_Spec_Path} template.nc template.nc # template grid has an offset for some reason, force correct starting point
cdo sellonlatbox,-75,-15,0,60 template.nc template.nc

# 2. remap horizontally
cdo remapcon,template.nc $filename ${Intermediate_File_Directory}/remap_merra2_date_${d}.nc
cdo remapbil,template.nc $temperature_file ${Intermediate_File_Directory}/remap_temp_merra2_date_${d}.nc

# 1. Join aerosol data with temperature data
cdo merge ${Intermediate_File_Directory}/remap_merra2_date_${d}.nc ${Intermediate_File_Directory}/remap_temp_merra2_date_${d}.nc $FINAL_FILE

# 6. Delete intermediate files
# rm  ${Intermediate_File_Directory}/*_date_${d}_time_${t}.nc
#!/bin/bash

# runs preprocessing steps for a single given filepath, check if processed file already exists needs to be done by outer
# scripts that calls this bash scrip

# filepath needs to be specified as first positional cmd argument when calling this file
filename=$1

MIN_LON=-75
MAX_LON=-15
MIN_LAT=0
MAX_LAT=60

Intermediate_File_Directory='/net/n2o/wolke_scratch/kjeggle/ERA5/intermediate'
Preproc_File_Directory='/net/n2o/wolke_scratch/kjeggle/ERA5/preproc'
Template_Path='/home/kjeggle/cirrus/src/config_files/template.nc'

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
#cdo -f nc -sellonlatbox,-180,180,-90,90 -random,r1440x720 template.nc # create target grid
#cdo setgrid,${Grid_Spec_Path} template.nc template.nc # template grid has an offset for some reason, force correct starting point
cdo -remapcon,$Template_Path ${Intermediate_File_Directory}/rh_ncatted_era5_date_${d}_time_${t}.nc ${Intermediate_File_Directory}/rh_remap_era5_date_${d}_time_${t}.nc # rh is extensive variable: conservative interp
cdo -remapbil,$Template_Path ${Intermediate_File_Directory}/era5_date_${d}_time_${t}.nc ${Intermediate_File_Directory}/remap_era5_date_${d}_time_${t}.nc # other variables are intensive variables:  bilinear interp

# 4. Select our target domain - is done in step 3 via Template
#cdo sellonlatbox,$MIN_LON,$MAX_LON,$MIN_LAT,$MAX_LAT ${Intermediate_File_Directory}/rh_remap_era5_date_${d}_time_${t}.nc ${Intermediate_File_Directory}/rh_sellatlon_era5_date_${d}_time_${t}.nc
#cdo sellonlatbox,$MIN_LON,$MAX_LON,$MIN_LAT,$MAX_LAT ${Intermediate_File_Directory}/remap_era5_date_${d}_time_${t}.nc ${Intermediate_File_Directory}/sellatlon_era5_date_${d}_time_${t}.nc

# 5. Merge relative humidity and other variables
cdo merge ${Intermediate_File_Directory}/rh_sellatlon_era5_date_${d}_time_${t}.nc ${Intermediate_File_Directory}/sellatlon_era5_date_${d}_time_${t}.nc ${FINAL_FILE}

# 6. Delete intermediate files
rm  ${Intermediate_File_Directory}/*_date_${d}_time_${t}.nc
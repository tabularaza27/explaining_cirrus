#!/bin/bash

# runs horizontal interpolation to template (0.25x0.25) grid and given domain

# year needs to be specified as first positional cmd argument when calling this file
year=$1

Source_File_Directory='/net/n2o/wolke_scratch/kjeggle/IC_CIR/incoming'
Intermediate_File_Directory='/net/n2o/wolke_scratch/kjeggle/IC_CIR/intermediate'
Template_Path='/home/kjeggle/cirrus/src/config_files/template.nc'

mkdir -p ${Intermediate_File_Directory}/${year}

for filename in $(ls ${Source_File_Directory}/${year}/*.nc)
do
  day=`echo $filename | grep -E -o '[0-9]{3}' | tail -n1`
  echo "Start Horizontal remapping $year $day"

  FINAL_FILE=${Intermediate_File_Directory}/${year}/remapped_ic_cir_${year}_${day}.nc
  if test -f "$FINAL_FILE"; then
    echo "Processed File already exists already exists."
    exit
  fi

  # 1. remap horizontally
  cdo remapnn,$Template_Path $filename $FINAL_FILE
done

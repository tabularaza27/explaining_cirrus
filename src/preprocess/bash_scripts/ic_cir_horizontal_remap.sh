#!/bin/bash

# runs preprocessing steps for a single given filepath, check if processed file already exists needs to be done by outer
# scripts that calls this bash scrip

# filepath needs to be specified as first positional cmd argument when calling this file
filename=$1

echo $filename

Intermediate_File_Directory='/net/n2o/wolke_scratch/kjeggle/IC_CIR/intermediate'
Template_Path='/home/kjeggle/cirrus/src/config_files/template.nc'

year=`echo $filename | grep -E -o '[0-9]{4}' | tail -n1`
day=`echo $filename | grep -E -o '[0-9]{3}' | tail -n1`

echo "Start Horizontal remapping $year $day"

FINAL_FILE=${Intermediate_File_Directory}/remapped_ic_cir_${year}_${day}.nc
if test -f "$FINAL_FILE"; then
  echo "Processed File already exists already exists."
  exit
fi

# 1. remap horizontally
cdo remapnn,$Template_Path $filename $FINAL_FILE

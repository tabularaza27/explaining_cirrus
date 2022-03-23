#!/bin/bash

# runs horizontal interpolation to template (0.25x0.25) grid and given domain

# year needs to be specified as first positional cmd argument when calling this file
year=$1

# config_id needs to be specified as second positional cmd argument when calling this file
# config id specifies the horizontal resolution and directories
config_id=$2

# get directories for given config
Source_File_Directory=`python -c "from src.scaffolding.scaffolding import get_data_product_dir; from src.preprocess.helpers.constants import IC_CIR_INCOMING; dir=get_data_product_dir('${config_id}', IC_CIR_INCOMING); print(dir)"`
Intermediate_File_Directory=`python -c "from src.scaffolding.scaffolding import get_data_product_dir; from src.preprocess.helpers.constants import IC_CIR_INTERMEDIATE; dir=get_data_product_dir('${config_id}', IC_CIR_INTERMEDIATE); print(dir)"`
Template_Path=`python -c "from src.scaffolding.scaffolding import get_abs_file_path; from src.preprocess.helpers.constants import TEMPLATE_PATH; path=get_abs_file_path('${config_id}', TEMPLATE_PATH); print(path)"`

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

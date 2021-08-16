#!/bin/bash

YEAR=2007
SOURCE_DIR='/net/n2o/wolke_scratch/tullyc/cscs/project-temp/Kai/Raw'
FILE_STUMPY='Kai_200701.01'
INTERIM_DIR='/net/n2o/wolke_scratch/kjeggle/ECHAM/interim'
DATACUBE_DIR='/net/n2o/wolke_scratch/kjeggle/ECHAM/datacube'

# Define Variables of interest
REMOS_VARS=drieff,dupdraft,dnihom,dnidet,dninuc,dninuc_hom,dninuc_het,dninuc_dust
ECHAM_VARS=relhum,xi,aclcac,aclcov
TRACER_VARS=SO4_AS,SO4_CS,SO4_KS,SO4_NS,DU_AI,DU_AS,DU_CI,DU_CS
CIRRUS_VARS=homoAeroAllConc,dustImmAeroAllConc,dustDepAeroAllConc,sat
ACTIV_VARS=W_LARGE,W

for i in {01..12};
do
  FILE_STUMPY=Kai_${YEAR}${i}.01
  echo "${FILE_STUMPY}"

  # temperature: convert spectral to gaussian grid
  cdo sp2gp -select,name=st ${SOURCE_DIR}/${FILE_STUMPY}_echam.nc ${INTERIM_DIR}/${FILE_STUMPY}_echam_st.nc
  # calculate horizontal winds
  cdo dv2uv -select,name=sd,svo ${SOURCE_DIR}/${FILE_STUMPY}_echam.nc ${INTERIM_DIR}/${FILE_STUMPY}_echam_uv.nc

  # select variables from streams and create files
  cdo -select,name=${REMOS_VARS} ${SOURCE_DIR}/${FILE_STUMPY}_remos.nc ${INTERIM_DIR}/${FILE_STUMPY}_remos.nc
  cdo -select,name=${ECHAM_VARS} ${SOURCE_DIR}/${FILE_STUMPY}_echam.nc ${INTERIM_DIR}/${FILE_STUMPY}_echam.nc
  cdo -select,name=${TRACER_VARS} ${SOURCE_DIR}/${FILE_STUMPY}_tracer.nc ${INTERIM_DIR}/${FILE_STUMPY}_tracer.nc
  cdo -select,name=${CIRRUS_VARS} ${SOURCE_DIR}/${FILE_STUMPY}_cirrus.nc ${INTERIM_DIR}/${FILE_STUMPY}_cirrus.nc
  cdo -select,name=${ACTIV_VARS} ${SOURCE_DIR}/${FILE_STUMPY}_activ.nc ${INTERIM_DIR}/${FILE_STUMPY}_activ.nc

  # merge to monthly datacube file
  cdo merge ${INTERIM_DIR}/${FILE_STUMPY}_remos.nc ${INTERIM_DIR}/${FILE_STUMPY}_echam.nc ${INTERIM_DIR}/${FILE_STUMPY}_tracer.nc ${INTERIM_DIR}/${FILE_STUMPY}_cirrus.nc ${INTERIM_DIR}/${FILE_STUMPY}_activ.nc ${INTERIM_DIR}/${FILE_STUMPY}_echam_st.nc ${INTERIM_DIR}/${FILE_STUMPY}_echam_uv.nc ${DATACUBE_DIR}/${FILE_STUMPY}_datacube.nc

done




## temperature: convert spectral to gaussian grid
#cdo sp2gp -select,name=st ${SOURCE_DIR}/${FILE_STUMPY}_echam.nc ${INTERIM_DIR}/${FILE_STUMPY}_echam_st.nc
## calculate horizontal winds
#cdo dv2uv -select,name=sd,svo ${SOURCE_DIR}/${FILE_STUMPY}_echam.nc ${INTERIM_DIR}/${FILE_STUMPY}_echam_uv.nc
#
## select variables from streams and create files
#cdo -select,name=${REMOS_VARS} ${SOURCE_DIR}/${FILE_STUMPY}_remos.nc ${INTERIM_DIR}/${FILE_STUMPY}_remos.nc
#cdo -select,name=${ECHAM_VARS} ${SOURCE_DIR}/${FILE_STUMPY}_echam.nc ${INTERIM_DIR}/${FILE_STUMPY}_echam.nc
#cdo -select,name=${TRACER_VARS} ${SOURCE_DIR}/${FILE_STUMPY}_tracer.nc ${INTERIM_DIR}/${FILE_STUMPY}_tracer.nc
#cdo -select,name=${CIRRUS_VARS} ${SOURCE_DIR}/${FILE_STUMPY}_cirrus.nc ${INTERIM_DIR}/${FILE_STUMPY}_cirrus.nc
#cdo -select,name=${ACTIV_VARS} ${SOURCE_DIR}/${FILE_STUMPY}_activ.nc ${INTERIM_DIR}/${FILE_STUMPY}_activ.nc
#
## merge to monthly datacube file
#cdo merge ${INTERIM_DIR}/${FILE_STUMPY}_remos.nc ${INTERIM_DIR}/${FILE_STUMPY}_echam.nc ${INTERIM_DIR}/${FILE_STUMPY}_tracer.nc ${INTERIM_DIR}/${FILE_STUMPY}_cirrus.nc ${INTERIM_DIR}/${FILE_STUMPY}_activ.nc ${INTERIM_DIR}/${FILE_STUMPY}_echam_st.nc ${INTERIM_DIR}/${FILE_STUMPY}_echam_uv.nc ${DATACUBE_DIR}/${FILE_STUMPY}_datacube.nc
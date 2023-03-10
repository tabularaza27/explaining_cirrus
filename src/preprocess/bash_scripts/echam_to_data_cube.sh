#!/bin/bash

YEAR=2007
#SOURCE_DIR='/net/n2o/wolke_scratch/tullyc/cscs/project-temp/Kai/Raw' # avg data
SOURCE_DIR='/net/n2o/wolke_scratch/tullyc/cscs/project-temp/Kai_inst/Raw/' # instantaneous data

INTERIM_DIR='/net/n2o/wolke_scratch/kjeggle/ECHAM/interim'
DATACUBE_DIR='/net/n2o/wolke_scratch/kjeggle/ECHAM/datacube'

# define domai
MIN_LON=-75
MAX_LON=-15
MIN_LAT=0
MAX_LAT=60

# Define Variables of interest
#REMOS_VARS=drieff,dupdraft,dnihom,dnidet,dninuc,dninuc_hom,dninuc_het,dninuc_dust,drhoair
REMOS_VARS=drieff,dicnc_inst,dicncb_inst,dnihet_inst,dnihom_inst,dnidet_inst,dninuc_inst,dninuc_het_inst,dninuc_hom_inst,dninuc_dust_inst,dninuc_soot_inst,dninuc_seed_inst,cloudice_inst,cloudliquid_inst,dupdraft_inst,dupdraftmax_inst,dtke_inst,drhoair_inst
ECHAM_VARS=relhum,xi,aclcac,aclcov,geosp,aps
TRACER_VARS=SO4_AS,SO4_CS,SO4_KS,SO4_NS,DU_AI,DU_AS,DU_CI,DU_CS
CIRRUS_VARS=sat,homoAeroAllConc,dustImmAeroAllConc,dustDepAeroAllConc,dustDepCAeroInConc,dustDepAAeroInConc,newIceCrystalsTime,newIceCrystalsConc,newIceCrystalsRad,homoAeroInConc,homoIceConc,dustImmAeroInConc,dustImmIceConc,dustDepAIceConc,dustDepCIceConc
ACTIV_VARS=W_LARGE,W,ICNC,ICNC_instantaneous,IWC_ACC,CLOUD_TIME,CLIWC_TIME,SICE
VPHYSC=aclc

for i in {01..12};
do
  rm -r $INTERIM_DIR
  mkdir $INTERIM_DIR

  FILE_STUMPY=Kai_inst_${YEAR}${i}.01
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
  cdo -select,name=${VPHYSC} ${SOURCE_DIR}/${FILE_STUMPY}_vphysc.nc ${INTERIM_DIR}/${FILE_STUMPY}_vphysc.nc

  # merge to monthly datacube file
  cdo merge ${INTERIM_DIR}/${FILE_STUMPY}_remos.nc ${INTERIM_DIR}/${FILE_STUMPY}_echam.nc ${INTERIM_DIR}/${FILE_STUMPY}_tracer.nc ${INTERIM_DIR}/${FILE_STUMPY}_cirrus.nc ${INTERIM_DIR}/${FILE_STUMPY}_activ.nc ${INTERIM_DIR}/${FILE_STUMPY}_vphysc.nc ${INTERIM_DIR}/${FILE_STUMPY}_echam_st.nc ${INTERIM_DIR}/${FILE_STUMPY}_echam_uv.nc ${DATACUBE_DIR}/${FILE_STUMPY}_datacube.nc
  cdo sellonlatbox,$MIN_LON,$MAX_LON,$MIN_LAT,$MAX_LAT ${DATACUBE_DIR}/${FILE_STUMPY}_datacube.nc ${DATACUBE_DIR}/${FILE_STUMPY}_datacube_domain.nc

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
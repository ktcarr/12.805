#!/bin/bash

module load anaconda
source activate torch_env

FP=/vortexfs1/scratch/kcarr/era-interim

# cdo -daymean $FP/tp_all.nc $FP/tp_daily.nc
# cdo -daymean $FP/tp_us_all.nc $FP/tp_us_daily.nc
# cdo -daymean $FP/ttr_all.nc $FP/ttr_daily.nc
cdo -daymean $FP/u_all.nc $FP/u_daily.nc
cdo -daymean $FP/v_all.nc $FP/v_daily.nc
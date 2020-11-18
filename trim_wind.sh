#!/bin/bash
module load nco

# Script trims wind files to [-15,15] latitude and selects 200 and 850 hPa pressure levels
# Argument to script is either u or v (representing wind direction)

# Set file paths
in_fp=/mnt/cmip5-data/reanalysis/era.interim/pl/$1/oper
out_fp=/climodes/data4/kcarr/era-interim
echo Processing $1
for year in {1979..2017}

do
    echo Trimming $year
    ncks -d latitude,-15.0,15.0 -d level,3,7,3 $in_fp/$1.$year.nc $out_fp/$1.$year.nc
done
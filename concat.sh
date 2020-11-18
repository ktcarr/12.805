#!/bin/bash
module load nco

# script concatenates files after trimming (unpacks all files, then concatenates)
# script takes variable name as input ($1)

fp=/climodes/data4/kcarr/era-interim/
echo Processing $1
for year in {1979..2017}
do
    echo Unpacking $year
    in_file=$fp/$1.$year.nc
    out_file=$fp/$1_temp.$year.nc
    
    ncpdq -U $in_file $out_file
    mv $out_file $in_file
done
echo Concatenating files
ncrcat $fp/$1.*.nc $fp/$1_all.nc

echo Repacking files
ncpdq $fp/$1_all.nc $fp/$1_all_packed.nc
mv $fp/$1_all_packed.nc $fp/$1_all.nc
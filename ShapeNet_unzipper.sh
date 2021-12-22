#!/usr/bin/env sh
# This script checks if ShapeNetCore data is downloaded to Matyi's external drive, and if yes, unzips it.

#project specific output folder for data - added to gitignore to make sure it isn't available online
dataset_dir="/Volumes/MacMiklos/M/BME/2021_12-OcclusionEnvironment/Shapenet/"
zip_dir="/Volumes/MacMiklos/M/BME/2021_12-OcclusionEnvironment/Shapenet/"
#/2021_12\ -\ Occlusion\ Environment/Shapenet/"
zip_file="ShapeNetCore.v2.zip"

# if you have already had the same version of dataset, you can
# create soft link like this:
# >> ln -s <path/to/ShapeNetCore/> shapenetcore

cd "$zip_dir"
pwd
if [ -f $zip_file ];
then
  echo "Zip file found."
  mkdir $dataset_dir
  echo "Unzipping..."
  unzip ShapeNetCore.v2.zip && rm -f ShapeNetCore.v2.zip
  mv ShapeNetCore.v2/* $dataset_dir && rm -rf ShapeNetCore.v2
  cd $dataset_dir
  # shellcheck disable=SC2045
  for zipfile in `ls *.zip`; do unzip $zipfile; done
  cd ..
  echo "Done."

else
  echo "Please visit http://shapenet.cs.stanford.edu to request ShapeNet data and then put the zip file in this folder and then run this script again.."
fi

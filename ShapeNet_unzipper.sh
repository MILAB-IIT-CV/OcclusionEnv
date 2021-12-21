#!/usr/bin/env sh
# This script checks if ShapeNetCore data is downloaded and if yes, unzips it.

# do not change this name
dataset_dir="./data/shapenetcore"
zip_file="ShapeNetCore.v1.zip"

# if you have already had the same version of dataset, you can
# create soft link like this:
# >> ln -s <path/to/ShapeNetCore/> shapenetcore

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

if [ -f $zip_file ];
then
  echo "Zip file found."
  mkdir $dataset_dir
  echo "Unzipping..."
  unzip ShapeNetCore.v1.zip && rm -f ShapeNetCore.v1.zip
  mv ShapeNetCore.v1/* $dataset_dir && rm -rf ShapeNetCore.v1
  cd $dataset_dir
  for zipfile in `ls *.zip`; do unzip $zipfile; done
  cd ..
  echo "Done."

else
  echo "Please visit http://shapenet.cs.stanford.edu to request ShapeNet data and then put the zip file in this folder and then run this script again.."
fi

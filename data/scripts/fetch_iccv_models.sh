#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
URL=http://napoli18.eecs.umich.edu/public_html/data/iccv_2015/iccv_models.tar.gz
FILE=$DIR/iccv_models.tar.gz
CHECKSUM=879c27e531112de46d34102d36836f84

if [ -f "$FILE" ]; then
  echo "File already exists. Checking md5..."
  checksum=`md5sum $FILE | awk '{ print $1 }'`
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading finetuned CaffeNet models (0.6G)..."
wget $URL -P $DIR;

echo "Unzipping..."
tar -zxvf $FILE -C $DIR

echo "Done."

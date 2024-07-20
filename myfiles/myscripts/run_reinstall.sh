#!/bin/bash

cd /mydata
BASE_DIR=`pwd`
source /etc/profile.d/mimicnet.sh

echo "Installing INET..."
cp -r ${MIMICNET_HOME}/third_party/parallel-inet .
cd parallel-inet
./compile.sh
cd ..

#!/bin/bash

timestamp=$(date +%s)

mkdir -p tempFiles/${timestamp}
cp *.sh tempFiles/${timestamp}
cp *.py tempFiles/${timestamp}
cp -r ../pyutil tempFiles/${timestamp}
echo $timestamp

cd tempFiles/${timestamp}
# arg1 = 1: start locally
if [ "x1" == "x${1}" ]
then
	export IS_INTERACTIVE=true
	./run.sh
else
	export IS_INTERACTIVE=false
	sbatch ./run.sh
fi

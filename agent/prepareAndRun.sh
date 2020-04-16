#!/bin/bash

# use this file to execute run.sh

timestamp=$(date +%s)

mkdir -p tempFiles/${timestamp}
cp *.py  tempFiles/${timestamp}
cp *.sh tempFiles/${timestamp}
cp *.csv tempFiles/${timestamp}
cp -r ../pyutil tempFiles/${timestamp}
echo $timestamp

cd tempFiles/${timestamp}
# start with "1" as parameter to execute locally
if [ "x1" == "x${1}" ]
then
	export IS_INTERACTIVE=true
	if [[ $# -eq 2 ]] ; then
		./run.sh 0 ${2}
	else
		./run.sh
	fi
else
	export IS_INTERACTIVE=false
	if [[ $# -eq 2 ]] ; then
		sbatch ./run.sh 0 ${2}
	else
		sbatch ./run.sh
	fi
fi

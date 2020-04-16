#!/bin/bash

# execute this file inside a result folder to resume that run

# arg1 = 1: start locally
# arg2 = /absolute/path/to/mask: restore this approach area mask
if [ "x1" == "x${1}" ]
then
	export IS_INTERACTIVE=true
	if [[ $# -eq 2 ]]
	# if [ "x1" == "x${2}" ]
	then
		./run.sh 1 ${2}
	else
		./run.sh 1
	fi
else
	export IS_INTERACTIVE=false
	if [[ $# -eq 2 ]]
	# if [ "x1" == "x${2}" ]
	then
		sbatch ./run.sh 1 ${2}
	else
		sbatch ./run.sh 1
	fi
fi

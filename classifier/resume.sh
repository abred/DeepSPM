#!/bin/bash

# execute this file inside a result folder to resume that run

# arg1 = 1: start locally
if [ "x1" == "x${1}" ]
then
	export IS_INTERACTIVE=true
	./run.sh 1
else
	export IS_INTERACTIVE=false
	sbatch ./run.sh 1
fi

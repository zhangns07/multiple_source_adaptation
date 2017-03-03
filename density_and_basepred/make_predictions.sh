#!/bin/bash
if [ $# -lt 2 ]; then
	echo "Usage: $0 libsvm_model target.libsvm"
	exit -1
fi

export LIBSVM=../../tool_binaries/libsvm-2.88

MODEL=$1
TARGET=$2

$LIBSVM/svm-predict $TARGET $MODEL `basename $TARGET`-on-`basename $MODEL`.pred

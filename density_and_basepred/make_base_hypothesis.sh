#!/bin/bash 
if [ $# -lt 3 ]; then
	echo \
"Usage: $0 data.liblinear c_start c_stop 
Returns best hypothesis after searching parameters 
C = {2^c_start,...2^c_end}."
	exit -1
fi

LIBSVM=../../tool_binaries/libsvm-2.88
NUMFOLDS=10

DATA=$1
CSTART=$2
CEND=$3

BESTC=None
BESTERR=99999999999;

#LIBSVM_PARAMS="-s 3 -t 2"
LIBSVM_PARAMS="-s 3 -t 2 -g 0.00078"

#echo Cross-Validating...
#for i in `seq $CSTART $CEND`; do
#	C=`python -c "print 2**$i"`
#	echo C: $C
##	$LIBLINEAR/train -s 0 -c $C -v 10 $DATA
#	ERR=`$LIBSVM/svm-train $LIBSVM_PARAMS -c $C -v $NUMFOLDS $DATA | grep "error" | cut -f 2 -d'='`
#	echo $NUMFOLDS Cross-Validation MSE: $ERR
#	if [ `echo "$ERR < $BESTERR" | bc` -eq 1 ]; then
#		BESTERR=$ERR
#		BESTC=$C
#	fi
#done

#echo Best tradeoff paramter: C=$BESTC

#echo Builing-Model...
#$LIBSVM/svm-train $LIBSVM_PARAMS -c $BESTC $DATA

echo Builing-Model...
$LIBSVM/svm-train $LIBSVM_PARAMS -c 8 $DATA

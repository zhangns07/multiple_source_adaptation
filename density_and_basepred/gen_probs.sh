if [ $# -lt 3 ]; then
	echo \
"Usage: $0 model target_data.txt vocab.txt"
	exit -1
fi

MODEL=$1
DATATARGET=$2
VOCAB=$3

#FSMLIB=~/Experiments/tools/fsm-4.0/bin
#GRMLIB=~/Experiments/tools/grm-4.0/bin
FSMLIB=../../tool_binaries/fsm-4.0/bin
GRMLIB=../../tool_binaries/grm-4.0/bin
SCRIPTS=../scripts


echo Adding end symbols...
awk '{print $0, "</s>"}' $DATATARGET > $DATATARGET.endsymbol

echo Storing data as automata...
$FSMLIB/farcompilestrings -i $VOCAB -u "<eps>" $DATATARGET.endsymbol > datatarget.far

echo Calculating probabilites...
#echo $CMD
$FSMLIB/farfilter "$FSMLIB/fsmcompose - "$MODEL"| $FSMLIB/fsmrmepsilon | $FSMLIB/fsmdeterminize | $FSMLIB/fsmprint -i "$VOCAB" -o "$VOCAB" | awk '{s += \$4}; END {print s}'" datatarget.far > prob.txt 

#
#if [ `echo $(wc -w prob.txt) | cut -f1 -d' '` -eq 0 ]; then
#	echo ERROR: probability file is empty
#	exit -1
#else
#	echo done. Probablities stored in prob.txt;
#fi

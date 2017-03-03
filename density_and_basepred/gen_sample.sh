if [ $# -lt 3 ]; then
	echo \
"Usage: $0 data.fsm Nsample vocab.txt"
	exit -1
fi

MODEL=$1
NSAMPLE=$2
VOCAB=$3

FSMLIB=../../tool_binaries/fsm-4.0/bin
GRMLIB=../../tool_binaries/grm-4.0/bin
SCRIPTS=./

#echo -n "0" | $FSMLIB/fsmcompile > empty.fsm
#$FSMLIB/fsmdifference $MODEL empty.fsm > $MODEL.new

#$GRMLIB/grmshrink -c 1000 -F $MODEL.new $MODEL

rm -f sample.seq.txt
$FSMLIB/fsmrandgen -n $NSAMPLE $MODEL | \
		$FSMLIB/fsmrmepsilon |$FSMLIB/fsmdeterminize | \
		$FSMLIB/fsmprint -i $VOCAB -o $VOCAB  > fsmpath
$SCRIPTS/print_sequence.py fsmpath > sample.seq.txt
		


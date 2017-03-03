if [ $# -lt 3 ]; then
	echo \
"Usage: $0 model_data.txt ngram_order min_occurrences"
	exit -1
fi

DATA=$1
NGRAM_ORDER=$2
NUM_OCCUR=$3

FSMLIB=../../tool_binaries/fsm-4.0/bin
GRMLIB=../../tool_binaries/grm-4.0/bin
SCRIPTS=./


echo Adding end symbols...
awk '{print $0, "</s>"}' $DATA > $DATA.endsymbol

#echo Storing data as automata...
#$FSMLIB/farcompilestrings -i vocab.txt -u "<unk>" $DATA.endsymbol > data.far
#$FSMLIB/farcompilestrings -i vocab.txt -u "<unk>" $DATATARGET.endsymbol > datatarget.far
$FSMLIB/farcompilestrings -i vocab.txt -u "<eps>" $DATA.endsymbol > data.far

echo Counting occurences...
$GRMLIB/grmcount -n $NGRAM_ORDER -i vocab.txt -s "<s>" -f "</s>" data.far > data.counts.fsm

echo Creating $NGRAM_ORDER-gram language model...
$GRMLIB/grmmake data.counts.fsm > data.lm.fsm


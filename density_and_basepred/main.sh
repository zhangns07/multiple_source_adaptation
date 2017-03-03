SCRIPTS=./ 
DATA=../data/text
TYPES="dvd books kitchen electronics"
NSAMPLE=10000
NUM_OCCUR=2
NGRAM=2


for SPLIT in `seq 1 10`; do
    LMMODELDIR=languagemodels-$NGRAM\gram-$NUM_OCCUR\minoccur
    EXPDATADIR=exp$SPLIT
    EXPPREDDIR=$EXPDATADIR/predictions
    EXPPROBDIR=$EXPDATADIR/prob-$NGRAM\gram-$NUM_OCCUR\minoccur/
    EXPRAWDIR=$EXPDATADIR/rawdata
    mkdir $EXPDATADIR
    mkdir $EXPPREDDIR
    mkdir $EXPPROBDIR
    mkdir $EXPRAWDIR

    # ========== 
    # 1. Making language model and SVR models
    # ========== 

    # Make unigram vocab: occur at least twice in intersection of four domains
    echo "Find intersection of words across all domains..."
    $SCRIPTS/get_unigram_intesect.py $NUM_OCCUR $DATA/dvd.txt $DATA/books.txt $DATA/kitchen.txt $DATA/electronics.txt  > unigram.$NUM_OCCUR\.txt
    cp unigram.$NUM_OCCUR\.txt vocab.txt

    # Build language model for each domain
    mkdir $LMMODELDIR
    for i in $TYPES; do
        $SCRIPTS/gen_lm.sh $DATA/$i.txt $NGRAM $NUM_OCCUR
        mv data.lm.fsm $LMMODELDIR/data.$i.lm.fsm
    done


    # Convert original review text into libsvm format
    echo "Make features for libsvm regression..."
    for i in $TYPES; do
        $SCRIPTS/libsvm_unigram_features.py vocab.txt $DATA/$i.txt $DATA/$i.labels > $DATA/$i.libsvm
    done

    # Shuffle 
    echo "Shuffle data..."
    for i in $TYPES; do
        $SCRIPTS/shuffle.py $DATA/$i.libsvm $SPLIT > $EXPPREDDIR/$i.libsvm
        head -1600 $EXPPREDDIR/$i.libsvm > $EXPPREDDIR/$i.train.libsvm
        tail -400 $EXPPREDDIR/$i.libsvm > $EXPPREDDIR/$i.test.libsvm

        $SCRIPTS/shuffle.py $DATA/$i.txt $SPLIT > $EXPDATADIR/$i.txt
        tail -400 $EXPDATADIR/$i.txt > $EXPRAWDIR/$i.test.txt

        $SCRIPTS/shuffle.py $DATA/$i.labels $SPLIT > $EXPDATADIR/$i.labels
        tail -400 $EXPDATADIR/$i.labels > $EXPRAWDIR/$i.test.labels
    done

    # SVR using training data (original reviews)
    echo "Build libsvm model..."
    CSTART=1
    CEND=5
    for i in $TYPES; do
        $SCRIPTS/make_base_hypothesis.sh $EXPPREDDIR/$i.train.libsvm $CSTART $CEND
        mv $i.train.libsvm.model $EXPPREDDIR/
    done


    #==========
    # 2. Deal with test data
    #==========
    # Calculate probability of i.test on j
    for i in $TYPES; do
        for j in $TYPES; do
            $SCRIPTS/gen_probs.sh $LMMODELDIR/data.$i.lm.fsm $EXPRAWDIR/$j.test.txt vocab.txt
            mv prob.txt $EXPPROBDIR/$j.test-in-$i.prob
        done
    done

    # Make predictions on test
    echo "Make features for libsvm regression..."
    echo "Make predictions..."
    for i in $TYPES; do
        for j in $TYPES; do
            $SCRIPTS/make_predictions.sh $EXPPREDDIR/$j.train.libsvm.model $EXPPREDDIR/$i.test.libsvm
            mv $i.test.libsvm-on-$j.train.libsvm.model.pred $EXPPREDDIR/
        done
    done


    #==========
    # 3. Randgen samples from language model, and deal with their probabilites and predictions
    #==========
    # Draw samples from each language model
    mkdir $LMMODELDIR/rawdata 
    for i in $TYPES; do
        echo generating $NSAMPLE samples from domain $i...
        $SCRIPTS/gen_sample.sh $LMMODELDIR/data.$i.lm.fsm $NSAMPLE vocab.txt 
        mv sample.seq.txt $LMMODELDIR/rawdata/$i.train.txt
    done

    # Compute the union of all sequences
    rm -f $LMMODELDIR/rawdata/train.dup.txt
    for i in $TYPES; do
        cat $LMMODELDIR/rawdata/$i.train.txt |sort >> $LMMODELDIR/rawdata/train.dup.txt
    done
    cat $LMMODELDIR/rawdata/train.dup.txt |sort |uniq > $LMMODELDIR/rawdata/train.txt

    # Convert randomly sampled data into libsvm format
    INPUTSIZE=$(wc -l $LMMODELDIR/rawdata/train.txt |cut -f 1 -d " ")
    COUNTER=0
    rm -f $LMMODELDIR/rawdata/emptylabels.txt
    while [ $COUNTER -lt $INPUTSIZE ]; do
        echo 0 >> $LMMODELDIR/rawdata/emptylabels.txt
        let COUNTER=COUNTER+1 
    done

    echo "Make features for libsvm regression..."
    $SCRIPTS/libsvm_unigram_features.py vocab.txt $LMMODELDIR/rawdata/train.txt $LMMODELDIR/rawdata/emptylabels.txt > $LMMODELDIR/rawdata/train.libsvm

    # Make predictions for randomly drawn samples
    echo "Make predictions..."
    for i in $TYPES; do
        $SCRIPTS/make_predictions.sh $EXPPREDDIR/$i.train.libsvm.model $LMMODELDIR/rawdata/train.libsvm
        mv train.libsvm-on-$i.train.libsvm.model.pred $EXPPREDDIR/
    done


    # Calculate probability and predictions of randomly drawn samples on domain i
    for i in $TYPES; do
        $SCRIPTS/get_emp_density_pred.py $LMMODELDIR/rawdata/train.txt \
            $LMMODELDIR/rawdata/$i.train.txt \
            $EXPPREDDIR/train.libsvm-on-$i.train.libsvm.model.pred 

        mv empdist.prob $EXPPROBDIR/train-in-$i.prob
        mv domain.pred $EXPPREDDIR/train-on-$i.labels
        mv train.on.domain.pred $EXPPREDDIR/train.libsvm-on-$i.train.libsvm.model.pred 
    done

    # Make labels for randomly drawn samples: 
    # - take average prediction if sample occurs in multiple domains
    # - else if sample only occurs in one domain k, use hk
    $SCRIPTS/get_average_prediction.py $EXPPREDDIR/train-on-dvd.labels \
        $EXPPREDDIR/train-on-books.labels \
        $EXPPREDDIR/train-on-kitchen.labels \
        $EXPPREDDIR/train-on-electronics.labels > $EXPRAWDIR/train.labels

    cp $LMMODELDIR/rawdata/train.txt $EXPRAWDIR/


    #    #==========
    #    # Train on all domains combined
    #    
    #    echo "Build libsvm model..."
    #    CSTART=1
    #    CEND=5
    #    rm -f $EXPPREDDIR/joint.train.libsvm
    #    for i in $TYPES; do
    #        cat $EXPPREDDIR/$i.train.libsvm  >> $EXPPREDDIR/joint.train.libsvm
    #    done
    #    
    #    $SCRIPTS/make_base_hypothesis.sh $EXPPREDDIR/joint.train.libsvm $CSTART $CEND
    #    mv joint.train.libsvm.model $EXPPREDDIR/
    #    
    #    
    #    # Make predictions
    #    echo "Make features for libsvm regression..."
    #    for j in $TYPES; do
    #    	$SCRIPTS/make_predictions.sh $EXPPREDDIR/joint.train.libsvm.model $EXPPREDDIR/$j.test.libsvm
    #    	mv $j.test.libsvm-on-joint.train.libsvm.model.pred $EXPPREDDIR/
    #    done



done

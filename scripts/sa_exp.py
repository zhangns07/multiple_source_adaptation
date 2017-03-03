import numpy as np
import sys
import scipy
import argparse
sys.path.append('../datasets')
sys.path.append('../optimization')
sys.path.append('../util')
import eval as E
from find_z import find_z
from sentiment_analysis import sm_problem


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str,  help='directory storing dataset.')
    parser.add_argument('--minO', type=int, default=2, choices=[2,4,8,16,32],
            help='minimum number of word occurrences to be in vocab.')
    parser.add_argument('--ngram', type=int, default=1, choices=[1,2,3,4],
            help='size of n in the n-gram dictionary.')
    parser.add_argument('--sources', nargs='+', 
            default=('kitchen', 'dvd', 'books', 'electronics'),
            help='List of the domains to use for experiment.')
    parser.add_argument('--split', type=int, default=1, 
            choices=range(1,10), help='Choose the data split for experiment.')
    return parser.parse_args()

def load_train_test_data(args):
    datadir = args.datadir
    minO = args.minO
    ngram = args.ngram
    split = args.split
    sources = args.sources
    DP = sm_problem(datadir=datadir, minO=minO, ngram=ngram, 
                    split=split, sources=sources, dset='train')
    DP_te = sm_problem(datadir=datadir, minO=minO, ngram=ngram, 
                    split=split, sources=sources, dset='test')
    return DP, DP_te

def eval_and_print(z,DP,ind=None):
    z0 = 1.0/DP.p * np.ones(DP.p)
    for k in range(DP.p):
        print 'error k={:d}:'.format(k), E.compute_loss_k(k,DP,ind=ind)
    print 'error z:', E.compute_weighted_loss(z,DP,ind=ind)
    print 'error z0:', E.compute_weighted_loss(z0,DP,ind=ind)

def main(args):
    DP, DP_te = load_train_test_data(args)
    
    print 'Solving for z...'
    z,g,zL,gL = find_z(DP)

    print 'Evaluating training data...'
    eval_and_print(z,DP)

    print 'Evaluating test data: equal distribution...'
    eval_and_print(z,DP_te)

    print 'Evaluating test data: only first domain...'
    eval_and_print(z,DP_te, ind=range(0,400))

    z_unif = 1.0/DP.p * np.ones(DP.p)
    print 'Eval global obj on train...'
    print 'z val=', E.global_obj(z, DP), 'z_unif val=', E.global_obj(z_unif, DP)

    print 'Eval global obj on test...'
    print 'z val=', E.global_obj(z, DP_te), 'z_unif val=', E.global_obj(z_unif, DP_te)

    ind = range(0,800)
    print 'Eval global obj on test lamda=[0.5, 0.5, 0,0]...',
    print 'z val=', E.global_obj(z, DP_te,ind=ind), 'z_unif val=', E.global_obj(z_unif, DP_te, ind=ind)


if __name__ == "__main__":
    args = setup_args()
    print 'Running Experiment with arguments'
    print args
    main(args)

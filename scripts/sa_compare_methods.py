# This script takes in 
# - computed z
# - ngram, minO, split, sources
# - test mixture(s) paramter lam

# It compute and write to file errors of following predictor on test mixture: 
# h_z,  h_lam, lam_comb (conv), lam_unif, h1, h2, ...

import numpy as np
import sys
sys.path.append('../datasets')
sys.path.append('../optimization')
from sentiment_analysis_randgen import sm_problem
from compute_quantities_log import compute_hz, compute_Dz

def sq_loss(pred,y):
    return (pred-y)**2
    
def eval_mixture(z,lam,DP, testfull = True):
    # if testfull = True (only recommended when lam is uniform among its support),
    # then use domain k full test data if lam[k] >0;
    # otherwise, use lam[k] percent of domain k test data.

    y = DP.get_true_values()
    h = DP.get_regressor()
    nperdomain=DP.n/DP.p

    idx = []
    convh  = np.zeros(DP.n)
    unifh  = np.zeros(DP.n)
    count = 0
    for k in range(DP.p): 
        if testfull:
            if lam[k]>0:
                idx = np.append(idx,range(k*nperdomain, (k+1)*nperdomain ))
        else:
            idx = np.append(idx,range(k*nperdomain, k*nperdomain + int(nperdomain*lam[k])))
        convh +=  lam[k] * DP.h[:,k]
        unifh += DP.h[:,k]

    unifh = unifh / DP.p
    idx=idx.astype('int')

    loss_hz = 0
    loss_hlam = 0
    loss_conv = 0
    loss_unif = 0
    for x in idx:
	hz = compute_hz(x,z,DP)
        hlam = compute_hz(x,lam,DP)
        loss_hz += sq_loss(hz, y[x])
	loss_hlam += sq_loss(hlam, y[x])
        loss_conv += sq_loss(convh[x], y[x])
        loss_unif += sq_loss(unifh[x], y[x])
    loss_hz = loss_hz / len(idx)
    loss_hlam = loss_hlam / len(idx)
    loss_conv = loss_conv / len(idx)
    loss_unif = loss_unif / len(idx)

    errh = np.zeros(DP.p)
    for k in range(DP.p):
	errh[k] = sum((DP.y[idx]- DP.h[idx,k])**2)/len(idx)

    print 'error hz:', loss_hz ,'error hlam:', loss_hlam ,'error lam-comb:', loss_conv, 'error unif:', loss_unif
    return loss_hz, loss_hlam, loss_conv, loss_unif, errh


def main(datadir, sources, minO, ngram, split, z):
    np.random.seed(0)
    print z
    DP_te = sm_problem(datadir=datadir, minO=minO, ngram=ngram, 
                    split=split, sources=sources, dset='test')

    L1= []; L2= []; L3= []; L4 = []; ERRH = []; 
    testfull = False
    LAM=[[0.4,0.2,0.2,0.2], 
            [0.2,0.4,0.2,0.2], 
            [0.2,0.2,0.4,0.2], 
            [0.2,0.2,0.2,0.4], 
            [0.3,0.3,0.2,0.2], 
            [0.3,0.2,0.3,0.2], 
            [0.3,0.2,0.2,0.3], 
            [0.2,0.3,0.3,0.2], 
            [0.2,0.3,0.2,0.3], 
            [0.2,0.2,0.3,0.3]]

#    testfull = True
#    LAM=[[1,0,0,0], [0,1,0,0], # K, D, 
#            [0,0,1,0], [0,0,0,1],# B, E
#            [float(1)/2, float(1)/2, 0,0], #KD,
#            [float(1)/2, 0,float(1)/2,0], #KB
#            [ float(1)/2,0,0, float(1)/2], #KE
#            [0,float(1)/2, float(1)/2,0], #DB
#            [0, float(1)/2,0, float(1)/2], #DE
#            [0,0, float(1)/2, float(1)/2], #BE
#            [0,float(1)/3,float(1)/3,float(1)/3],
#            [float(1)/3,0,float(1)/3,float(1)/3],
#            [float(1)/3,float(1)/3,0,float(1)/3],
#            [float(1)/3,float(1)/3,float(1)/3,0],
#            [float(1)/4,float(1)/4,float(1)/4,float(1)/4]]


    for i in xrange(len(LAM)):
        l0 = LAM[i]
        l1,l2,l3,l4,errh = eval_mixture(z,l0,DP_te, testfull)
    	L1.append(l1); L2.append(l2); L3.append(l3); L4.append(l4)
    	ERRH.append(errh)

    fname = '_'.join(sources)+str(ngram)+'gram_'+str(minO)+'mino_'+str(split)+'_ontest.csv'
    ret = np.column_stack((L1,L2,L3,L4,ERRH))

    with open(fname,'a') as f_handle:
            np.savetxt(f_handle, ret, delimiter=",")


datadir = '/home/nz695/domainadap/randgen/'
sources = ['kitchen','dvd','books','electronics']
minO = 2 
ngram = 2
z = [0.23459073, 0.30207491, 0.26867261, 0.19466175]

for split in range(1,11):
    main(datadir, sources, minO, ngram, split, z)


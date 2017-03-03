import matplotlib.pyplot as plt
import numpy as np

def OKS(kp_pred, kp_gt, area=1.0):
    oks = 0 
    # sigma values learned from coco
    sigmas = np.array([ 0.087,  0.079,  0.107,  0.107,  0.079,  0.072,  0.062,  0.035,
                0.072,  0.062,  0.025,  0.035])/10.0
    var = (sigmas * 2)**2
    nz = np.where(np.array(kp_gt).sum(axis=1) > 0)[0]
    nnz = len(nz)
    #print 'nonzero ind', nz
    #print kp_gt[nz,:]
    #print kp_pred[nz,:]
    for i,c in enumerate(nz):
        d = ((np.array(kp_pred)[c,:]-np.array(kp_gt)[c,:])**2).sum()
        oks += np.exp(-d / var[c] / area/2)
    #print oks, nnz, oks/nnz
    return oks / nnz

def euclidean_kp_loss(h, kp):
    K,H,W = h.shape
    loss = np.zeros(K)
    for k in range(K):
        y,x = kp[k,:]/8
        loss[k] = ((h[k,:,:]**2).sum() - 2*h[k,x,y] + 1)/(H*W)
    return loss

def dist_euclidean_kp_loss(D, h, kp):
    K,H,W = h.shape
    loss = np.zeros(K)
    for k in range(K):
        y,x = kp[k,:]/8


def disp_img_tight(im):
    plt.imshow(im)
    plt.ylim([im.shape[0],0])
    plt.xlim([0,im.shape[1]])
    plt.grid()

def disp_kp(kp, ax=None, clr='b'):
    if len(kp.shape) == 1:
        disp_single_kp(kp, ax=ax, clr=clr)
    else:
        for k in range(kp.shape[0]):
            disp_single_kp(kp[k,:], ax=ax, clr=clr)

def disp_single_kp(kp, ax=None, clr='b'):
    if ax is None:
        plt.plot(kp[0],kp[1], 'o', color=clr)
    else:
        ax.plot(kp[0], kp[1], 'o', color=clr)

def computeAP(r, p):
    r = np.hstack([0, r, 1])
    p = np.hstack([0, p, 0])
    for i in np.arange(len(p)-2,1,-1):
        p[i]=max(p[i], p[i+1])
    ii = np.where(r[1:] != r[:-1])[0]
    ap = sum((r[ii+1]-r[ii]) * p[ii+1])
    return ap

def compute_pr(tp,sc):
    if isinstance(tp, list):
        tp_all = np.hstack(tp)
    else:
        tp_all = tp
    if isinstance(sc, list):
        sc_all = np.hstack(sc)
    else:
        sc_all = sc
    num_c = tp_all.shape[0]
    prec = []
    rec = []
    ap = []
    for c in range(tp_all.shape[0]):
        tp_c = tp_all[c,:]
        sc_c = sc_all[c,:]
        ind = sc_c.argsort()[::-1]
        tp_c = tp_c[ind]
        csum = np.cumsum(tp_c)
        r = csum/tp_c.sum()
        p = csum/np.arange(1,len(tp_c)+1)
        prec.append(p); rec.append(r)
        ap.append(computeAP(p=p,r=r))
    return prec,rec,ap


def print_ap(ap, cats):
    for c in range(len(ap)):
        print '{:s}: {:0.2f}'.format(cats[c], ap[c]*100)
    print 'mAP: {:0.2f}'.format(np.mean(ap)*100)


def plot_pr_curves(p,r,ap,cls,algs):
    numC = len(cls)
    fig = plt.figure(figsize=[12,8])
    for c in range(numC):
        plt.subplot(np.floor(np.sqrt(numC)),np.ceil(np.sqrt(numC)),c+1)
        for pa,ra,apa,alg in zip(p,r,ap,algs):
            plt.plot(ra[c], pa[c], label='{:s}: {:0.1f}'.format(alg,apa[c]*100))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(cls[c])
        plt.legend()
        plt.tight_layout()
    print 'Evaluation: mAP'
    for apa,alg in zip(ap,algs):
        print '{:s}: {:.1f}'.format(alg,np.mean(apa)*100)

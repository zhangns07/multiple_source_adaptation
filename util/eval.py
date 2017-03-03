import sys
sys.path.append('../optimization')
import numpy as np
from compute_quantities import compute_hz, compute_Dz

def sq_loss(pred,y):
    return (pred-y)**2
    
def compute_loss_k(k, DP, ind=None):
    hk = DP.get_regressor()[:,k]
    y = DP.get_true_values()
    loss = 0
    if ind == None:
        ind = range(DP.n)
    loss = sq_loss(hk[ind], y[ind]).sum() 
    return loss / len(ind)

def compute_LDkhz(k, z, DP, ind=None):
    if ind == None:
        ind = range(DP.n)
    y = DP.get_true_values()
    Dk = DP.get_marginal_density()[:,k]
    loss = 0

    for x in ind:
        hz = compute_hz(x,z,DP)
        loss += Dk[x] * sq_loss(hz, y[x])
    return loss

def compute_LDzhz(z, DP, ind=None):
    if ind == None:
        ind = range(DP.n)
    y = DP.get_true_values()
    loss = 0

    for x in ind:
        Dz = compute_Dz(x,z,DP)
        hz = compute_hz(x,z,DP)
        loss += Dz * sq_loss(hz, y[x])
    return loss

def compute_LDkhk(k, DP, ind=None):
    if ind == None:
        ind = range(DP.n)
    y = DP.get_true_values()
    Dk = DP.get_marginal_density()[:,k]
    hk = DP.get_regressor()[:,k]
    loss = 0

    for x in ind:
        loss += Dk[x] * sq_loss(h[x], y[x])
    return loss


def compute_weighted_loss(z, DP, ind=None):
    y = DP.get_true_values()
    loss = 0
    if ind == None:
        ind = range(DP.n)
    for x in ind:
        hz = compute_hz(x,z,DP)
        loss += sq_loss(hz, y[x])
    return loss / len(ind)

def global_obj(z, DP, ind=None):
    Lk = np.zeros(DP.p)
    if ind == None:
        ind = range(DP.n)
    for k in range(DP.p):
        Lk[k] = compute_LDkhz(k, z, DP, ind=ind)
    
    Lz = compute_LDzhz(z, DP, ind=ind)
    
    return max(Lk - Lz)

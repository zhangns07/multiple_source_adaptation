import sys
import numpy as np
from math import *

def log_plus(a,b=None): # controls overflow
        OVERFLOW_LIMIT = 700
        if b == None:
        #simply return original argument
                return a

        if a - b > OVERFLOW_LIMIT:
                return b
        if b - a > OVERFLOW_LIMIT:
                return a
        if a < b:
                return a - log(1 + exp(a-b))
        else:
                return b - log(1 + exp(b-a))


def compute_neglogJKz(x, z, DP):
        num_domains = len(z)
        denom = None
        sol = None
        for j in range(num_domains):
                if z[j] == 0:
                        continue
                denom = log_plus(DP.logD[x][j] - np.log(z[j]), denom)
                sol = log_plus(DP.logD[x][j] - np.log(z[j]) - np.log(max(DP.h[x][j], np.exp(-100))), sol)
        #sol = np.exp(denom - sol)
        return sol, denom

def compute_Dz(x, z, DP):
        sol,denom = compute_neglogJKz(x, z, DP)
        return np.exp(-denom)


def compute_hz(x, z, DP):
    sol, denom = compute_neglogJKz(x, z, DP)
    sol = np.exp(denom - sol)
    return sol

def compute_uv_new(z, DP,rho):
    D = DP.get_marginal_density()
    h = DP.get_regressor()
    y = DP.get_true_values()

    hz, logKz = np.zeros(DP.n), np.zeros(DP.n)
    for x in range(DP.n):
        sol, denom = compute_neglogJKz(x, z, DP)
        hz[x] = np.exp(denom-sol)
        logKz[x] = -denom

    Lk = np.zeros(DP.p)
    EklogK = np.zeros(DP.p)
    for k in range(DP.p):
        Lk[k] = sum(D[:,k] * (y - hz)**2)
        EklogK[k] = sum(D[:,k] * logKz)

    u = Lk - 2*DP.M * EklogK
    v = sum(Lk*z) - 2* DP.M * EklogK
    return u, v

def compute_guv_new(z, DP, rho):
    D = DP.get_marginal_density()
    h = DP.get_regressor()
    y = DP.get_true_values()

    hz, logKz = np.zeros(DP.n), np.zeros(DP.n)
    for x in range(DP.n):
        sol, denom = compute_neglogJKz(x, z, DP)
        hz[x] = np.exp(denom-sol)
        logKz[x] = -denom

    Lk = np.zeros(DP.p)
    gLk = np.zeros([DP.p, DP.p])
    gEklogK = np.zeros([DP.p, DP.p])

    for k in range(DP.p):
        Lk[k] = sum(DP.D[:,k] * (y -hz)**2)
        for i in range(DP.p):
            #gu[k,i] = sum(np.exp(-DP.logD[:,k] - DP.logD[:,i] - logKz) * (h[:,i] - hz - 2*DP.M))
            gLk[k,i] = sum(np.exp(-DP.logD[:,k] - DP.logD[:,i] - logKz) * (h[:,i] - hz ) * 2 * (hz - y))
            gEklogK[k,i] = sum(np.exp(-DP.logD[:,k] - DP.logD[:,i] - logKz))

    gLz = np.dot(z, gLk) + Lk
    gu = gLk - 2 * DP.M * gEklogK
    gv = gLz - 2 * DP.M * gEklogK

    return np.matrix(gu),np.matrix(gv)

get_ipython().magic(u'matplotlib inline')
import numpy as np
import scipy 
import random
from scipy.stats import multivariate_normal as mn
from sklearn.linear_model import LogisticRegression as LR


# ## Setup the synthetic data

eta = 0.01
nC = 20 # number of points per class
p  = 3 # number of domains
C  = 10 # number of classes
n  = nC*p*C
U = 1.0/n

def get_random_D_h(nC, p, C):
    D = np.random.rand(nC*p*C,C,p) # prob point n is in domain p and class C
    h = np.random.rand(nC*p*C,C,p) # score for point n in domain p and class C
    # normalize D and h
    for k in range(p): # loop over domain
        D[:,:,k] = D[:,:,k] / D[:,:,k].sum()
    
        for i in range(nC):
            h[i,:,k] = h[i,:,k]/h[i,:,k].sum() # since this is the output of a softmax function its normalized over classes 
    return D,h


# # Define a data problem


class synthetic_problem():
    
    def __init__(self, p=3, nC=20, c=10, seed=1337):
        self.p = p # number of domains
        self.nC = nC # pts per class
        self.C = C # number of classes
        self.n = nC * p * C # num pts
        self.U = 1.0 / self.n # unif dist
        self.eta = 0.01
        self.seed = seed
        
        np.random.seed(seed)
        self.generate_density()
        self.generate_regressor()
        self.generate_y()
        
    def generate_density(self):
        D = np.random.rand(self.n,self.p)
        # Normalize each domain
        #for k in range(self.p):
        #    D[:,k] = D[:,k] / D[:,k].sum()
        self.D = D
    
    def generate_regressor(self):
        h = np.random.rand(self.n,self.p)
        #for k in range(self.p):
        #    h[:,k] = h[:,k] / h[:,k].sum()
        self.h = h
        
        # compute H
        self.H = np.zeros(self.n)
        for i in range(self.n):
            self.H[i] = 1.0 / self.p * h[i,:].sum()
            
    def generate_y(self):
        y = np.random.rand(self.n)
        y = y / y.sum()
        self.y = y
        
    def get_marginal_density(self):
        return self.D
    
    def get_regressor(self):
        return self.h
    
    def get_true_values(self):
        return self.y
    
    def get_H(self):
        return self.H


# # Optimization

# ## Objective
# * z_{t+1} = argmin gamma
# * subject to
#     * u_k(z) - v_k(z_t) - (z - z_t) grad_{v_k(z_t)} <= g for all k in [p]
#     * -z_k <= 0 forall k in [p]
#     * sum_{k=1}^p z_k - 1 = 0


def compute_H(x, DP):
    return DP.get_H()[x]

def compute_Dz(x, z, DP):
    """ Dz = sum_k z_k * D_k(x)"""
    D = DP.get_marginal_density()[x,:]
    Dz = 0
    for k in range(DP.p):
        Dz += z[k] * D[k]
    return Dz

def compute_Jz(x, z, DP):
    const = DP.eta * DP.U * compute_H(x, DP)
    D = DP.get_marginal_density()[x,:]
    h = DP.get_regressor()[x,:]
#    zDh = z*D*h
    zDh = 0
    for k in range(DP.p):
        zDh += z[k] * D[k] * h[k]
#    return zDh + p*const 
    return zDh + const 
#    return sum(zDh + const)

def compute_Kz(x, z, DP):
    return compute_Dz(x, z, DP) + DP.eta * DP.U

def compute_hz(x, z, DP, Jz=None, Kz=None):
    if Jz is None:
        Jz = compute_Jz(x, z, DP)
    if Kz is None:
        Kz = compute_Kz(x, z, DP)
        
    return Jz / Kz

def compute_fz(x, z, DP, Jz=None, Kz=None):
    if not Jz:
        Jz = compute_Jz(x, z, DP)
    if not Kz:
        Kz = compute_Kz(x, z, DP)
    return (Jz + 1) ** 2 / (2*Kz)

def compute_gz(x, z, DP, Jz=None, Kz=None):
    if not Jz:
        Jz = compute_Jz(x, z, DP)
    if not Kz:
        Kz = compute_Kz(x, z, DP)
    return ((Jz**2) + 1) / (2*Kz)

def compute_Fz(x, z, DP, fz=None, gz=None):
    if not fz:
        fz = compute_fz(x, z, DP)
    if not gz:
        gz = compute_gz(x, z, DP)
    return 2 * (fz**2) + 2 * (gz**2)

def compute_Gz(x, z, DP, fz=None, gz=None):
    if not fz:
        fz = compute_fz(x, z, DP)
    if not gz:
        gz = compute_gz(x, z, DP)
    return (fz + gz)**2

def compute_grad_Jz(x, z, DP):
    D = DP.get_marginal_density()
    h = DP.get_regressor()
    return D[x,:] * h[x,:]

def compute_grad_Kz(x, z, DP):
    D = DP.get_marginal_density()[x,:]
    return D

def compute_grad_gz(x, z, DP, grad_Jz=None, 
                    grad_Kz=None, Jz=None,
                   Kz=None):
    if grad_Jz is None:
        grad_Jz = compute_grad_Jz(x, z, DP)
    if grad_Kz is None:
        grad_Kz = compute_grad_Kz(x, z, DP)
    if not Jz:
        Jz = compute_Jz(x, z, DP)
    if not Kz:
        Kz = compute_Kz(x, z, DP)
    return (Jz * grad_Jz) / Kz - (((Jz**2) + 1)*grad_Kz) / (2* (Kz**2))
#    return (Jz * grad_Jz) / Kz - (((Jz**2) + 1)*grad_Kz) / Kz

def compute_grad_fz(x, z, DP, Jz=None, Kz=None,
                    grad_Jz=None, grad_Kz=None):
    if grad_Jz is None:
        grad_Jz = compute_grad_Jz(x, z, DP)
    if grad_Kz is None:
        grad_Kz = compute_grad_Kz(x, z, DP)
    if not Jz:
        Jz = compute_Jz(x, z, DP)
    if not Kz:
        Kz = compute_Kz(x, z, DP)
#    return (Jz + 1)*grad_Jz / Kz - (Jz+1)**2 * grad_Kz / (Kz**2)
    return (Jz + 1)*grad_Jz / Kz - (Jz+1)**2 * grad_Kz /(2* (Kz**2))

def compute_grad_hz(x,z,DP, grad_fz=None, grad_gz=None):
    if grad_fz is None:
        grad_fz = compute_grad_fz(x, z, DP)
    if grad_gz is None:
        grad_gz = compute_grad_gz(x, z, DP)       
    return grad_fz - grad_gz
    

def compute_grad_Gz(x, z, DP, fz=None, gz=None,
                   grad_fz=None, grad_gz=None):
    if not fz:
        fz = compute_fz(x, z, DP)
    if not gz:
        gz = compute_gz(x, z, DP)
    if grad_fz is None:
        grad_fz = compute_grad_fz(x, z, DP)
    if grad_gz is None:
        grad_gz = compute_grad_gz(x, z, DP)
    return 2 * (fz + gz) * (grad_fz + grad_gz)



# Define u and v following proposition 9
def compute_u(z,DP):
    D = DP.get_marginal_density()
    h = DP.get_regressor()
    etaU = DP.eta * DP.U
    y = DP.get_true_values()
    
    const = np.zeros(DP.n)
    for k in range(DP.p):
        const += z[k] * D[:,k] * (y**2)

    u = np.zeros(DP.p)
    for x in range(DP.n):
        H = compute_H(x, DP)
        Dz = compute_Dz(x, z, DP)
        Jz = compute_Jz(x, z, DP)
        Kz = compute_Kz(x, z, DP)
        hz = compute_hz(x, z, DP, Jz=Jz, Kz=Kz)
        fz = compute_fz(x, z, DP, Jz=Jz, Kz=Kz)
        gz = compute_gz(x, z, DP, Jz=Jz, Kz=Kz)
        Fz = compute_Fz(x, z, DP, fz=fz, gz=gz)
        Gz = compute_Gz(x, z, DP, fz=fz, gz=gz)
        for k in range(DP.p):
            Dk = D[x,k]
            hk = h[x,k]

            v1 = Dk * (Fz + 2*y[x]*gz + y[x]**2)
            v2 = etaU*Fz + 2*y[x]*Jz + 2*etaU*y[x]*gz
            u[k] += v1 + v2 
    return u - const.sum()

def compute_v(z, DP):
    D = DP.get_marginal_density()
    h = DP.get_regressor()
    H = DP.get_H()
    y = DP.get_true_values()

    v = np.zeros(DP.p)
    for x in range(DP.n):
        Gz = compute_Gz(x, z, DP)
        fz = compute_fz(x, z, DP)
        Jz = compute_Jz(x, z, DP)
        hz = compute_hz(x, z, DP, Jz=Jz)

        etaU = DP.eta * DP.U
        va = D[x,:] * (Gz + 2*fz*y[x])
        vb = Jz * hz + etaU*Gz + 2*etaU*fz*y[x]
        v += va + vb
    return v


 
if 0: 
    # this is following proposition 10, which we do not have a 
    # good theoretical gaurantee for. 
    def compute_u(z, DP):
        D = DP.get_marginal_density()
        h = DP.get_regressor()
        etaU = DP.eta * DP.U

        # sum_k z_k * (sum_x Dk(x) * h_k(x)^2)
        const = 0
        v = (D * (h**2)).sum(axis=0)
        for k in range(DP.p):
            const += z[k] * v[k]

        u = np.zeros(DP.p)
        for x in range(DP.n):
            H = compute_H(x, DP)
            Dz = compute_Dz(x, z, DP)
            Jz = compute_Jz(x, z, DP)
            Kz = compute_Kz(x, z, DP)
            hz = compute_hz(x, z, DP, Jz=Jz, Kz=Kz)
            fz = compute_fz(x, z, DP, Jz=Jz, Kz=Kz)
            gz = compute_gz(x, z, DP, Jz=Jz, Kz=Kz)
            Fz = compute_Fz(x, z, DP, fz=fz, gz=gz)
            Gz = compute_Gz(x, z, DP, fz=fz, gz=gz)
            for k in range(DP.p):
                Dk = D[x,k]
                hk = h[x,k]

                v1 = Dk * (Fz + 2*hk*gz + hk**2)
                v2 = etaU*Fz + Jz*hz + 2*etaU*H*gz
                u[k] += v1 + v2 - const
        return u


    def compute_v(z, DP):
        D = DP.get_marginal_density()
        h = DP.get_regressor()
        H = DP.get_H()

        v = np.zeros(DP.p)
        for x in range(DP.n):
            Gz = compute_Gz(x, z, DP)
            fz = compute_fz(x, z, DP)
            etaU = DP.eta * DP.U
            etaUH = etaU * H[x]
            va = D[x,:] + etaU
            vb = 2 * h[x,:]*D[x,:] + 2 * etaUH
            v += (Gz * va) + (fz * vb)

        return v


    def compute_grad_v(z, DP):
        D = DP.get_marginal_density()
        h = DP.get_regressor()
        H = DP.get_H()

        grad_v = np.zeros(DP.p)
        for x in range(DP.n):
            grad_Gz = compute_grad_Gz(x, z, DP)
            grad_fz = compute_grad_fz(x, z, DP)
            etaU = DP.eta * DP.U
            etaUH = etaU * H[x]
            va = D[x,:] + etaU
            vb = 2 * h[x,:]*D[x,:] + 2 * etaUH

            grad_v += (grad_Gz * va) + (grad_fz * vb)

        return grad_v    



DP = synthetic_problem()
zp = np.repeat(1.0 / DP.p, DP.p)
x=0
print 'z:', zp, 'x:', x
print 'H(x)', compute_H(x, DP)
print 'Dz(x)', compute_Dz(x, zp, DP)
print 'Jz(x)', compute_Jz(x, zp, DP)
print 'Kz(x)', compute_Kz(x, zp, DP)
print 'fz(x)', compute_fz(x, zp, DP)
print 'gz(x)', compute_gz(x, zp, DP)
print 'Fz(x)', compute_Fz(x, zp, DP)
print 'Gz(x)', compute_Gz(x, zp, DP)
print 'grad_Jz(x)', compute_grad_Jz(x, zp, DP)
print 'grad_Kz(x)', compute_grad_Kz(x, zp, DP)
print 'grad_gz(x)', compute_grad_gz(x, zp, DP)
print 'grad_fz(x)', compute_grad_fz(x, zp, DP)
print 'grad_Gz(x)', compute_grad_Gz(x, zp, DP)



print 'z:', zp, 'x:', x
H = compute_H(x, DP)
Dz = compute_Dz(x, zp, DP)
Jz = compute_Jz(x, zp, DP)
Kz = compute_Kz(x, zp, DP)
fz = compute_fz(x, zp, DP, Jz=Jz, Kz=Kz)
gz = compute_gz(x, zp, DP, Jz=Jz, Kz=Kz)
Fz = compute_Fz(x, zp, DP, fz=fz, gz=gz)
Gz = compute_Gz(x, zp, DP, fz=fz, gz=gz)
u = compute_u(zp,DP)
v = compute_v(zp,DP)

grad_Jz = compute_grad_Jz(x, zp, DP)
grad_Kz = compute_grad_Kz(x, zp, DP)
grad_gz = compute_grad_gz(x, zp, DP, Jz=Jz, 
                          Kz=Kz, grad_Jz=grad_Jz,
                         grad_Kz=grad_Kz)
print 'fz(x)', fz
print 'gz(x)', gz
print 'Fz(x)', Fz
print 'Gz(x)', Gz
print 'grad_Jz(x)', grad_Jz
print 'grad_Kz(x)', grad_Kz
print 'grad_gz(x)', grad_gz
print 'grad_fz(x)', compute_grad_fz(x, zp, DP)
print 'grad_Gz(x)', compute_grad_Gz(x, zp, DP)

print 'u', u
print 'v', v


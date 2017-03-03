import cvxpy as cvx
import numpy as np

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
    const = (DP.eta * DP.U) * compute_H(x, DP)
    D = DP.get_marginal_density()[x,:]
    h = DP.get_regressor()[x,:]
    zDh = 0
    for k in range(DP.p):
        zDh += z[k] * (D[k] * h[k])
    return zDh + const

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
    return (Jz * grad_Jz) / Kz - (((Jz**2) + 1)*grad_Kz) / (2*(Kz**2))

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
    return (Jz + 1)*grad_Jz / Kz - (Jz+1)**2 * grad_Kz / (2* (Kz**2))

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

def compute_grad_Fz(x, z, DP, fz=None, gz=None,
                   grad_fz=None, grad_gz=None):
    if not fz:
        fz = compute_fz(x, z, DP)
    if not gz:
        gz = compute_gz(x, z, DP)
    if grad_fz is None:
        grad_fz = compute_grad_fz(x, z, DP)
    if grad_gz is None:
        grad_gz = compute_grad_gz(x, z, DP)
    return 4 * (fz*grad_fz + gz*grad_gz) 


# Define u and v following proposition 9
def compute_u(z,DP):
    D = DP.get_marginal_density()
    h = DP.get_regressor()
    etaU = DP.eta * DP.U
    y = DP.get_true_values()
    
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
        
        a0 = Dz * (y[x]**2)
        a1 = (Fz + 2*y[x]*gz + y[x]**2)
        a2 = etaU*Fz + 2*y[x]*Jz + 2*etaU*y[x]*gz
        u += a0 + D[x,:] * a1 + a2
    return u

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

def compute_grad_v(z, DP):
    D = DP.get_marginal_density()
    h = DP.get_regressor()
    H = DP.get_H()
    y = DP.get_true_values()
    
    grad_v = np.zeros([DP.p, DP.p])
    for x in range(DP.n):
        grad_Gz = compute_grad_Gz(x, z, DP)
        grad_fz = compute_grad_fz(x, z, DP)
        Jz = compute_Jz(x, z, DP)
        Kz = compute_Kz(x, z, DP)
        grad_Jz = D[x,:] * h[x,:]
        grad_Kz = D[x,:]
        etaU = DP.eta * DP.U
        etaUH = etaU * H[x]
        grad_Jzhz = 2*(Jz/Kz)*grad_Jz - ((Jz**2)/(Kz**2)) * grad_Kz 
        for k in range(DP.p): #kth element of v
            for i in range(DP.p): # ith grad dim
                a0 = (D[x,k] + etaU) * grad_Gz[i]
                a1 = (2*D[x,k]*y[x] + 2*etaU*y[x]) * grad_fz[i]
                a2 = grad_Jzhz[i]
                grad_v[k,i] += a0 + a1 + a2

    return np.matrix(grad_v)

def compute_grad_u(z, DP):
    D = DP.get_marginal_density()
    h = DP.get_regressor()
    H = DP.get_H()
    y = DP.get_true_values()
    
    grad_u = np.zeros([DP.p, DP.p])
    for x in range(DP.n):
        grad_gz = compute_grad_gz(x, z, DP)
        grad_Fz = compute_grad_Fz(x, z, DP)
        Jz = compute_Jz(x, z, DP)
        Kz = compute_Kz(x, z, DP)
        grad_Jz = D[x,:] * h[x,:]
        grad_Dz = D[x,:]
        etaU = DP.eta * DP.U
        etaUH = etaU * H[x]
        
        
        for k in range(DP.p): #kth element of v
            for i in range(DP.p): # ith grad dim
                a0 = D[x,k] * (grad_Fz[i] + 2*grad_gz[i] * y[x])
                a1 = - grad_Dz[i] * (y[x]**2)
                a2 = etaU*grad_Fz[i] + 2*y[x]*(grad_Jz[i] + etaU*grad_gz[i])
                grad_u[k,i] += a0 + a1 + a2

    return np.matrix(grad_u)


def compute_u_new(z, DP):
    D = DP.get_marginal_density()
    h = DP.get_regressor()
    y = DP.get_true_values()
    etaU = DP.eta * DP.U
    
    u = np.zeros(DP.p)
    for x in range(DP.n):
        Kz = compute_Kz(x, z, DP)
        hz = compute_hz(x, z, DP, Kz=Kz)
        a0 = (hz - y[x])**2 - 2*DP.M*np.log(Kz)
        for k in range(DP.p):
            u[k] += a0 * (D[x,k] + etaU)
    return u

def compute_u_cvx(z, DP):
    D = DP.get_marginal_density()
    h = DP.get_regressor()
    y = DP.get_true_values()
    etaU = DP.eta * DP.U
    
    u = []
    for k in range(DP.p):
        uk = 0
        for x in range(DP.n):
            Kz = compute_Kz(x, z, DP)
            hz = compute_hz(x, z, DP, Kz=Kz)
            a0 = (hz - y[x])**2 - 2*DP.M*cvx.log(Kz)
            uk += a0 * (D[x,k] + etaU)
        u.append(uk)
    return u

def compute_v_new(z, DP):
    D = DP.get_marginal_density()
    h = DP.get_regressor()
    y = DP.get_true_values()
    etaU = DP.eta * DP.U
    
    v = np.zeros(DP.p)
    for x in range(DP.n):
        Kz = compute_Kz(x, z, DP)
        hz = compute_hz(x, z, DP, Kz=Kz)
        v0 = Kz * (y[x]-hz)**2
        a1 = -2*DP.M*np.log(Kz)
        for k in range(DP.p):
            v1 = a1 * (D[x,k]+etaU)
            v[k] += (v0+v1) 
    return v


def compute_gv_new(z, DP):
    D = DP.get_marginal_density()
    h = DP.get_regressor()
    y = DP.get_true_values()
    etaU = DP.eta * DP.U

    gv = np.zeros([DP.p, DP.p])
    for x in range(DP.n):
        Kz = compute_Kz(x, z, DP)
        hz = compute_hz(x, z, DP, Kz=Kz)

        a0 = (hz-y[x])**2
        a1 = 2*(hz - y[x])
        for k in range(DP.p):
            a2 = -2*DP.M * ((D[x,k] + etaU) / Kz)

            for i in range(DP.p):
                v1 = a1 * (h[x,i]-hz)  
                gv[k,i] += D[x,i] * (a0 + v1 + a2) 
    return np.matrix(gv)

def compute_gu_new(z, DP):
    D = DP.get_marginal_density()
    h = DP.get_regressor()
    y = DP.get_true_values()
    etaU = DP.eta * DP.U

    gu = np.zeros([DP.p, DP.p])
    for x in range(DP.n):
        Kz = compute_Kz(x, z, DP)
        hz = compute_hz(x, z, DP, Kz=Kz)
        for k in range(DP.p):
            a0 = 2*(D[x,k]+etaU)
            for i in range(DP.p):
                a1 = D[x,i] / Kz
                a2 = (hz-y[x])*(h[x,i]-hz) - DP.M
                gu[k,i] += a0*a1*a2
    return np.matrix(gu)


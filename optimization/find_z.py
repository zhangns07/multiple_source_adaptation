import numpy as np
import scipy
from scipy.optimize import minimize
import sys
sys.path.append('../optimization')
from compute_quantities import compute_u, compute_v, compute_grad_v, compute_grad_u, compute_u_new, compute_v_new, compute_gv_new, compute_gu_new

def equality_constraint(x):
    z = x[:-1]
    return sum(z) - 1

def bnd(p):
    b = []
    for k in range(p):
#        b.append((0,1))
        b.append((1e-5,1))
    b.append((0, np.inf)) # gamma > 0
    return b

def u_minus_v(z0, DP):
    u = compute_u_new(z0, DP)
    v = compute_v_new(z0, DP)
    return u-v

def main_constraint_new(x, x0, DP, a=1):
    gamma = x[-1]
    z = np.squeeze(np.array(x[:-1]))
    z0 = np.squeeze(np.array(x0[:-1]))
    u_z = compute_u_new(z, DP)
    v_z0 = compute_v_new(z0, DP)
    gv_z0 = compute_gv_new(z0, DP)
#    return a * (gamma - (u_z - v_z0 - np.ravel(np.dot(gv_z0,z-z0))))
    return np.hstack([a*(gamma - (u_z - v_z0 - np.ravel(np.dot(gv_z0,z-z0)))),z])


def cst_jacobian_new(x, x0, DP, a=1):
    # gamma - (uz-v0 - (z-z0)*gv0)
    z = x[:-1]
    z0 = x0[:-1]
    gu_z = compute_gu_new(z, DP)
    gv_z0 = compute_gv_new(z0, DP)
    j = a * (gv_z0 - gu_z) # jacobian of second term
    out = np.zeros([2*DP.p, DP.p + 1])
    out[:DP.p,:DP.p] = j
    out[:DP.p,-1] = a
    out[DP.p:, :DP.p] = np.diag(np.ones(DP.p))
    return out

def main_constraint(x, x0, DP, a=1e-20):
    gamma = x[-1]
    z = np.squeeze(np.array(x[:-1]))
    z0 = np.squeeze(np.array(x0[:-1]))
    u_z = compute_u(z, DP)
    v_z0 = compute_v(z0, DP)
    gv_z0 = compute_grad_v(z0, DP)
    return a * (gamma - (u_z - v_z0 - np.squeeze(np.array((z-z0)*gv_z0))))

def main_constraint_per_k(x, x0, DP, k, a=1e-20):
    gamma = x[-1]
    z = np.squeeze(np.array(x[:-1]))
    z0 = np.squeeze(np.array(x0[:-1]))
    u_z = compute_u(z, DP)
    v_z0 = compute_v(z0, DP)
    gv_z0 = compute_grad_v(z0, DP)
    return a * (gamma - (u_z[k] - v_z0[k] - np.array((z-z0)*gv_z0)[k]))


def cst_jacobian(x, x0, DP, a=1e-20):
    z = x[:-1]
    z0 = x0[:-1]
    gu_z = compute_grad_u(z, DP)
    gv_z0 = compute_grad_v(z0, DP)
    j = a * (gv_z0 - gu_z) # jacobian of second term
    out = np.zeros([DP.p, DP.p + 1])
    out[:DP.p,:DP.p] = j
    out[:,-1] = a
    return out

def solve_iter(xp, DP):
    a = 1e10; a2 = 1.0
    ja = np.zeros(xp.shape[0])
    ja[-1] = a2
    fun = lambda x: a2 * x[-1] # bottom variable is gamma
    jac = lambda x: ja # jacobian of function
    eq_cst = dict(type='eq', fun=equality_constraint)
    main_cst = dict(type='ineq', fun=main_constraint_new, 
            jac=cst_jacobian_new, args=(xp, DP,a))
    cons = (eq_cst, main_cst)

    opt = dict(maxiter=1e2, disp=True, eps=1e-15, ftol=1e-15)
    res = minimize(fun, xp, method='SLSQP', jac=jac, #bounds=bnd(DP.p), 
            constraints=cons, tol=1e-12, options=opt )
    print res.message
    cval = main_constraint_new(xp, res.x, DP, a=a)
    print 'constraint value', cval
    return res.x[:-1], res.x[-1], res, cval

def find_z(DP):
    z0=np.repeat(1.0 / DP.p, DP.p)
    #z0 = np.random.rand(DP.p) 
    #z0 = z0 / z0.sum()
    uMv = u_minus_v(z0, DP)
    g0 = 10 #1 * max(uMv)
    print 'g0', g0
    x0 = np.hstack([z0, g0])
	
    max_iter = 50
    xp = x0
    tol = 1e-12
    z_list = []
    gamma_list = []
    for i in range(max_iter):
        z,gamma,res,cval = solve_iter(xp, DP)
        z_list.append(z); gamma_list.append(gamma)
        print 'iter {:d}: z='.format(i), z, 'gamma=',gamma
        if np.abs(xp[-1] - gamma) < tol  and (cval>0).all():
            print 'absolute change in gamma', xp[-1] - gamma
            break
        xp = np.hstack([z, gamma])

    return z, gamma, z_list, gamma_list


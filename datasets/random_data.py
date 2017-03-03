import numpy as np
import random

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
        for k in range(self.p):
            D[:,k] = D[:,k] / D[:,k].sum()
        self.D = D
    
    def generate_regressor(self):
        h = np.random.rand(self.n,self.p)
        for k in range(self.p):
            h[:,k] = h[:,k] / h[:,k].sum()
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


class synthetic_problem_input():#load synthetic data from given file
    def __init__(self,datadir = '', dset = '', split = 1):
        self.datadir = datadir
        self.split = split
        self.load_data()
        self.p  = (len(self.data[0])-1)/2
        self.n  = len(self.data)
        self.load_density()
        self.load_regressor()
        self.load_y()
        self.U = 1.0 / self.n # unif dist
        self.eta = 1e-10
        self.dset = dset
        self.load_M()

    def load_data(self):
        filename = self.datadir+'iter_'+str(self.split)+'.csv'
        data = np.genfromtxt(filename, delimiter=',')
        self.data = data

    def load_density(self):
        self.D = self.data[:,0:self.p]


    def load_regressor(self):
        self.h = self.data[:,self.p:(2*self.p)]
        self.H = np.zeros(self.n)
        for i in range(self.n):
            self.H[i] = 1.0 / self.p * self.h[i,:].sum()


    def load_y(self):
        self.y = self.data[:,-1]


    def load_M(self):
        M = 0
        for k in range(self.p):
            M = max(M,max((self.y - self.h[:,k])**2))
        self.M = M

    def get_marginal_density(self):
        return self.D

    def get_regressor(self):
        return self.h
 
    def get_true_values(self):
        return self.y

    def get_H(self):
        return self.H

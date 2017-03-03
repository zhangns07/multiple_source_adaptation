from os.path import join
import numpy as np

class sentiment_analysis_data():
    
    def __init__(self, datadir='', ):
        self.datadir = datadir
        self.all_domains = ('books', 'dvd', 'electronics', 'kitchen')
        self.label_fmt = 'exp{split}/rawdata/{domain}.{dset}.labels'
        self.data_fmt = 'exp{split}/rawdata/{domain}.{dset}.txt'
        self.M = 5**2 # y \in [0,5] to (y-h)**2 <= (5-0)**2
        
    def load_data(self, domain, split, dset):
        dname = self.data_fmt.format(split=split, 
                                        domain=domain, dset=dset)
        with open(join(self.datadir, dname), 'rb') as f:
            data = f.read().splitlines()
        return data
    
    def load_labels(self, domain, split, dset='train'):
        dname = self.label_fmt.format(split=split, domain=domain, dset=dset)
        with open(join(self.datadir, dname), 'rb') as f:
            data = f.read().splitlines()
        labels = np.array([float(x) for x in data])
        return labels
    
    def load_prob(self, tr_dom, dom, split, n, minO, dset, scale=1):
        """
        Loads the precomputed probs of each example in dom
        according to the model trained on tr_dom. Each is 
        computed over the particular data <split> using an
        <n>-gram language model with vocabulary based on the
        minimum occurances (minO) of each word across all domains
        in sentiment analysis dataset.
        """
        prob_fmt = 'exp{:d}/prob-{:d}gram-{:d}minoccur/{:s}.{:s}-in-{:s}.prob'
        prob_file = prob_fmt.format(split, n, minO, dom, dset, tr_dom)
        with open(join(self.datadir, prob_file), 'rb') as f:
            data = f.read().splitlines()
        #prob = np.array([np.exp(-float(x)/scale) for x in data])
        prob = scale * np.array([np.exp(-float(x)) for x in data])
        nlogprob = np.array([float(x) for x in data])
        return prob, nlogprob
    
    def load_pred(self, tr_dom, dom, split, dset):
        pred_fmt = 'exp{:d}/predictions/{:s}.{:s}.libsvm-on-{:s}.train.libsvm.model.pred'
        pred_file = pred_fmt.format(split, dom, dset, tr_dom)
        with open(join(self.datadir, pred_file), 'rb') as f:
            data = f.read().splitlines()
        pred = np.array([float(x) for x in data])
        return pred

class sm_problem():
    
    def __init__(self, split=1, datadir='', sources=None,
                ngram=1, minO=2, dset='train', eta=1e-100):
        self.datadir = datadir # directory where data is stored
        self.split = split
        self.ngram = ngram
        self.minO = minO
        self.splitdir = join(datadir, 'exp{:d}'.format(split))
        self.dset = dset
        self.sources = sources
        self.p = len(sources) # number of domains
        self.SA = sentiment_analysis_data(datadir=datadir)
        self.M = self.SA.M 
        self.load_y() # load the gt labels and set self.n
        self.load_density()
        self.load_regressor()
        self.U = 1.0 / self.n # unif dist
        self.eta = eta
        
        
    def load_density(self):
        D = np.zeros([self.n,self.p])
        logD = np.zeros([self.n, self.p])
        for (k,d_tr) in enumerate(self.sources):
            Dd = []; logD_d = [];
            for d in self.sources:
                prob_d, nlogp = self.SA.load_prob(d_tr, d, self.split, 
                             self.ngram, self.minO, self.dset)
                Dd.append(prob_d)
                logD_d.append(nlogp)
            D[:,k] = np.hstack(Dd)
            logD[:,k] = np.hstack(logD_d)
        self.D = D
        self.logD = logD
    
    def load_regressor(self):
        h = np.zeros([self.n,self.p])
        for (k, d_tr) in enumerate(self.sources):
            hd = []
            for d in self.sources:
                hd.append(self.SA.load_pred(d_tr, d, self.split, self.dset))
            h[:,k] = np.hstack(hd)
        self.h = h
        
        # compute H
        self.H = 1.0/ self.p * h.sum(axis=1)
            
    def load_y(self):
        y = []
        for s in self.sources:
            y.append(self.SA.load_labels(s, self.split, self.dset))
        self.y = np.hstack(y)
        self.n = self.y.shape[0]
        
    def get_marginal_density(self):
        return self.D
    
    def get_regressor(self):
        return self.h
    
    def get_true_values(self):
        return self.y
    
    def get_H(self):
        return self.H

    def subsample(self, src_ind, tgt_ind):
        n_d = self.n / self.p
        ind = []
        for j in tgt_ind:
            ind.extend(range(j*n_d, (j+1)*n_d))
        src_ind = np.array(src_ind, dtype=int)
        tgt_ind = np.array(tgt_ind, dtype=int)
        self.D = self.D[:,src_ind]
        self.D = self.D[ind,:]
        self.h = self.h[ind,:]
        self.h = self.h[:,src_ind]
        self.y = self.y[ind]
        self.p = len(src_ind)
        self.H = 1.0/self.p * self.h.sum(axis=1)
        self.n = self.D.shape[0]
        self.sources = [self.sources[t] for t in tgt_ind]


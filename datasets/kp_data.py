import numpy as np
import cv2 as cv 
import scipy
import sys
import time
from os.path import join, exists
import scipy.io as spio
import cPickle as pkl

class kpData(object):
    """Loads pose data. Now only load images with keypoints for exactly one person."""

    def __init__(self, datadir, dset='train'):
        self.datadir = datadir
        self.dset = dset
        self.data_kp_cats,self.kp_cats = self._load_kp_cats() 
        self.kde,self.sc_max = self._load_kde()
        self.joint_map, self.data_joint_map = self._load_joint_maps()
        self._load_annotations()
 
    def _load_kp_cats(self):
        kp_cats = np.load(join(self.datadir, 'kp_cats_model.npy'))
        data_kp_cats = np.load(join(self.datadir, 'kp_cats.npy'))
        return data_kp_cats, kp_cats

    def convert_kp2joint(self, kp):
        "Converts keypoint in data indexing to joint indexing."
        return kp[self.data_joint_map,:].copy()

    def convert_h2joint(self, h):
        "Converts h in model indexing to indexing format."
        return h[self.joint_map,...].copy()

    def _load_joint_maps(self):
        with open('exp/{:s}2joint_map.npy'.format(self.name)) as f:
            joint_map = np.load(f)

        with open('exp/data_{:s}2joint_map.npy'.format(self.name)) as f:
            data_joint_map = np.load(f)
        return joint_map, data_joint_map
    
    def _load_kde(self):
        fname = 'exp/kde_{:s}.pkl'.format(self.name)
        if exists(fname):
            with open(fname) as f:
                r = pkl.load(f)
                kde = r['kde']
                sc_max = r['sc_max']
            return kde, sc_max
        else:
            return None

    def load_Dh(self):
        with open('exp/{:s}Imgs_Dh.pkl'.format(self.name)) as f:
            Dh_dict = pkl.load(f)
        return Dh_dict['Dxy'], Dh_dict['h'], Dh_dict['y']

    def _load_annotations(self):
        "Implement per dataset."
        pass 

    def get_imgIds_annos(self, dset='train'):
        "Implement per dataset."
        pass

    def load_image(self, idx):
        imname = self.img_fmt.format(idx)
        return cv.imread(imname)

    def disp_image(self, im):
        return im[:,:,[2,1,0]] #BGR->RGB

    def compute_Dh(self, ft, h, kp, HW, ind=None, subsample=50):
        """
        Given a set of features (ft) compute D
        Given h return in the joint format
        Given kp return a gt vector in joint format
        HW is the height/width of features and h values
        subsample some number of bg points
        """
        H,W = HW
        num_pts = ft.shape[0]
        kp_ind = self.get_kp_ind(kp, H, W)
        if ind is None:
            if len(kp_ind) == 0:
                return [],[],[],[]
            bg_ind = np.random.choice(num_pts, subsample)
            ind = np.array(np.hstack([kp_ind, bg_ind]), dtype=np.uint32)

        ft = ft[ind, :]
        Dx = np.exp(self.kde.score_samples(ft) - self.sc_max)
        h = self.convert_h2joint(h[:,ind])
        y = self.get_y(kp, ind, H, W)

        # Compute Dxy = Dx * P(y|x)
        h_sm = np.exp(h)
        h_sm = h_sm / np.tile(h_sm.sum(axis=0), [h_sm.shape[0],1])
        Dxy = Dx * h_sm

        return Dxy, h, y, ind

    def get_kp_ind(self, kp, H, W):
        ind = []
        for k in kp:
            y,x = k
            if x>0 or y>0:
                ind.append(x*W + y)
        return ind

    def get_y(self, kp, ind, H, W):
        num_cls = kp.shape[0]
        num_pts = len(ind)
        y = np.zeros([num_cls, num_pts])
        for c in range(num_cls):
            y_c, x_c = kp[c,:]
            ii = np.uint32(x_c*W + y_c)
            if ii in ind:
                ic = np.where(ii==ind)[0]
                y[c,ic] = 1
            else:
                print 'Missing labeled kp', c, ii
        return y

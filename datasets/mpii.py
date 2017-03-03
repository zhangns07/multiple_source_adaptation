import numpy as np
import cv2 as cv 
import scipy
import sys
import time
from os.path import join
import scipy.io as spio
import cPickle as pkl
from datasets import kp_data
reload(kp_data)

class MPII( kp_data.kpData ):
    """Loads MPII pose data. Now only load images with keypoints for exactly one person."""

    def __init__(self, datadir, dset='train'):
        self.name = 'mpii'
        super(self.__class__, self).__init__(datadir, dset=dset)
        self.img_fmt = join(datadir,'images', '{:s}')
  

    def _load_annotations(self):
        M = loadmat(join(self.datadir, 'mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'))
        anno = M['RELEASE']
        ind = self._get_ind_single_person_with_kp(anno)
        self.img_train = anno['img_train'][ind]
        if self.dset == 'train':
            self.ind = np.where(self.img_train)[0]
        elif self.dset == 'test':
            self.ind = np.where((self.img_train + 1) % 2)[0]
        else:
            print 'Unknown dset', dset
            return

        print 'num unique inds', len(np.unique(ind))
        self.all_imgIds = self._get_imgs(anno['annolist'][ind]) 
        self.all_kp = self._get_keypoints(anno['annolist'][ind])
        self.area = self._get_areas(anno['annolist'][ind])

    def _is_single_person(self, sp):
        return isinstance(sp, int) and (sp==1)

    def _has_kp(self, a):
        return 'annopoints' in a['annorect'].keys()

    def _get_ind_single_person_with_kp(self, anno):
	ind = []
	single_person = anno['single_person']
        annolist = anno['annolist']

	for i in range(len(annolist)):
	    sp = single_person[i]
	    
	    if self._is_single_person(sp):
	        annorect = annolist[i]['annorect']
	        if isinstance(annorect,dict) and annorect.has_key('annopoints'):
	            ind.append(i)

        return ind


    def _get_kp(self, a):
        pts = a['annorect']['annopoints']['point']
        kp = np.zeros([len(self.data_kp_cats),2], dtype=np.uint32)
        for v in pts:
            x,y,c = v['x'],v['y'],v['id']
            visible = isinstance(v['is_visible'],(int,unicode)) and (int(v['is_visible']) ==1)
            if visible:
                kp[c,:] = [x-1,y-1]
            else:
                kp[c,:] = [0,0] # ignore kp which are not visible
        return kp

    def _get_keypoints(self, annolist):
        return np.array([self._get_kp(a) for a in annolist])
    
    def _get_imgs(self, annolist):
        return np.array([a['image']['name'] for a in annolist])
    
    def _get_areas(self, annolist):
        area = np.zeros(len(annolist))
        for i,a in enumerate(annolist):
            x1 = a['annorect']['x1']
            y1 = a['annorect']['y1']
            x2 = a['annorect']['x2']
            y2 = a['annorect']['y2']
            area[i] = (x2-x1)*(y2-y1)
        return area
            

    def get_imgIds_annos(self):
        return self.all_imgIds[self.ind], self.all_kp[self.ind], self.area[self.ind]


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        elif isinstance(elem, np.ndarray): # check each element in list
            for i,e in enumerate(elem):
                if isinstance(e, spio.matlab.mio5_params.mat_struct):
                    elem[i] = _todict(e)
            dict[strg] = elem
        else:
            dict[strg] = elem
    return dict

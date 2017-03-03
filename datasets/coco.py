import numpy as np
import cv2 as cv 
import scipy
import sys
import time
from os.path import join
from datasets import kp_data
reload(kp_data)

class Coco(kp_data.kpData):

    def __init__(self, datadir, dset='train2014'):
        self.name = 'coco'
        super(self.__class__,self).__init__(datadir, dset=dset)
        img_prefix = 'COCO_{:s}_'.format(dset)
        self.img_fmt = join(datadir, dset, img_prefix+'{:012d}.jpg') 

    def get_imgIds(self):
        return self.coco_kps.getImgIds()

    def load_anno(self, idx):
        annId = self.coco_kps.getAnnIds(imgIds=idx)
        return self.coco_kps.loadAnns(annId)

    def _load_annotations(self):
        sys.path.append(join(self.datadir, 'coco/PythonAPI'))
        from pycocotools.coco import COCO
        annFile = '{:s}/annotations/person_keypoints_{:s}.json'.format(self.datadir, self.dset)
        self.coco_kps = COCO(annFile)
        imgIds = np.array(self.coco_kps.getImgIds())
        ind = self._get_ind_single_person_with_kp(imgIds)
        self.imgIds = imgIds[ind]
        self.kp = self._get_keypoints(self.imgIds)
        self.area = self._get_areas(self.imgIds)

    def _has_kp(self, anno):
        return anno['num_keypoints'] > 0

    def _is_single_person(self, anno):
        return len(anno) == 1

    def _get_ind_single_person_with_kp(self, imgIds):
        ind = []
        for i,imId in enumerate(imgIds):
            anno = self.load_anno(imId)
            if self._is_single_person(anno) and self._has_kp(anno[0]):
                ind.append(i)
        return ind

    def _get_keypoints(self, imgIds):
        all_kp = []
        for i,imId in enumerate(imgIds):
            anno = self.load_anno(imId)[0]
            kp = np.zeros([len(self.data_kp_cats), 2], dtype=np.uint32)
            kp_i = anno['keypoints']
            for k in range(0,len(kp_i),3):
                if kp_i[k+2] >= 1: #exists and visible
                    kp[k/3,:] = kp_i[k:k+2]

            all_kp.append(kp)
        return all_kp

    def _get_areas(self, imgIds):
        areas = np.zeros(len(imgIds))
        for i,imId in enumerate(imgIds):
            anno = self.load_anno(imId)[0]
            areas[i] = anno['area']
        return areas
    def get_imgIds_annos(self):
        return self.imgIds, self.kp, self.area

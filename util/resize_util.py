import cv2 as cv
import numpy as np

def resize_img_by_height(im, height=368):
    Ho,Wo = im.shape[:2]
    scale = float(height) / Wo #TODO: seems like H is correct, Wo
    Hn,Wn = int(Ho*scale), int(Wo*scale)
    im2 = cv.resize(im, (Wn,Hn), interpolation=cv.INTER_CUBIC)
    #im2 = cv.resize(im, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    return im2, scale

def resize_img_by_height_safe(im, height):
    if height is None:# or im.shape[0] < height:
        im_sc = im; scale=1.0
    else:
        im_sc,scale = resize_img_by_height(im, height=height)
    return im_sc, scale

def resize_image_by_scale(im, i_scale):
    return cv.resize(im, (0,0), fx=i_scale, fy=i_scale, interpolation=cv.INTER_CUBIC)

def upsample_hc_scores(hc, H_curr, W_curr, H_des, W_des):
    tmp = hc.reshape([H_curr,W_curr])
    h_des = cv.resize(tmp, (W_des,H_des), interpolation=cv.INTER_CUBIC)
    return h_des.reshape(H_des*W_des)

def upsample_h_scores(h, Hi, Wi, H, W):
    num_c = h.shape[0]
    h_up = np.zeros([num_c,H*W])
    for c in range(num_c):
        h_up[c,:] = upsample_hc_scores(h[c,:], Hi, Wi, H, W)
    return h_up

def model_output_dim(im,model_scale):
    return int(np.ceil(im.shape[0]/model_scale)),int(im.shape[1]/model_scale)

def get_image_scales(data):
    if data.name == 'coco': # scales to average over based on img dataset
        scales = [0.5,1.0] # [0.5, 1.0, 1.5, 2]
    else:
        scales = [1.0] #[0.7, 1.0, 1.3]
    return scales

def clip_scores(h, thresh=0.01):
    h[h<thresh] = 0
    return h

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad



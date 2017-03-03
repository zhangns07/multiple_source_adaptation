import cv2 as cv 
import numpy as np
import sys
import time

def feed_image(net, im):
    net.blobs['data'].reshape(*(1, 3, im.shape[0], im.shape[1]))
    #net.forward() # dry run
    net.blobs['data'].data[...] = np.transpose(np.float32(im[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;

def process_net(net):
    start_time = time.time()
    output_blobs = net.forward()
    return output_blobs
    
def reshape_image(im, model, scale=1.0):
    im_sc = cv.resize(im, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    im_sc_padded, pad = util.padRightDownCorner(im_sc, model['stride'], model['padValue'])
    print im_sc_padded.shape
    return im_sc_padded, pad


def extract_sc(net, layer='Mconv7_stage6_L2'):
    C,H,W = net.blobs[layer].data[0].shape
    sc = net.blobs[layer].data[0].reshape(C,H*W)
    return sc

def extract_ft(net, layer='Mconv6_stage6_L2'):
    D,H,W = net.blobs[layer].data[0].shape
    ft = net.blobs[layer].data[0].transpose(1,2,0).reshape(H*W, D)
    return ft

def forward_image(net, im):
    feed_image(net,im)
    return process_net(net)

def get_kp_ind(kp, H, W):
    ind = []
    for k in kp:
        y,x = k
        if x>0 and y>0:
            try:
                i = np.ravel_multi_index(([x,y]), [H,W])
                ind.append(i)
            except ValueError:
                print 'Error: Found keypoint which cannot be converted into scale'
                print 'kp:', (x,y), 'im size', (H,W)
                print 'manual comp', x*W+y, H*W
                raise
    return ind

def get_y(kp, ind, H, W):
    numCls = kp.shape[0]
    N = len(ind)
    y = np.zeros([numCls, N])
    for c in range(numCls):
        y_c, x_c = kp[c,:] 
        if x_c > 0 or y_c > 0:
            ii = np.uint32(x_c*W + y_c)
            if ii in ind:
                ic = np.where(ii==ind)[0]
                y[c,ic] = 1
            else:
                print 'Missing', c, ii
    return y

def run_net(net, im, ft_layer, sc_layer):
    "Run network with im and extract ft and scores."
    _ = forward_image(net, im)
    ft = extract_ft(net, layer=ft_layer)
    H,W = net.blobs[ft_layer].data[0].shape[1:3]
    h = extract_sc(net, layer=sc_layer)
    return ft,h,H,W

def get_subsample_ind(kp, H, W, N, n=50):
    "Return keypoint values and n random bg samples."
    kp_ind = get_kp_ind(kp, H, W)
    if len(kp_ind) == 0: # no kp in image
        return None
    bg_ind = np.random.choice(N, 50) # randomly sample other pts
    ind = np.array(np.hstack([kp_ind, bg_ind]), dtype=np.uint32)
    return ind


def score2prob(h):
    h_sm = np.exp(h)
    h_sm = h_sm / np.tile(h_sm.sum(axis=0), [h_sm.shape[0], 1])
    return h_sm

def compute_Dh(net, data, im, kp, ft_layer, sc_layer, ind=None):
    """Computes D and h on image using data model"""
    ft,h,H,W = run_net(net, im, ft_layer, sc_layer)
    
    sampled_ind = ind is None
    if ind is None:
        N = ft.shape[0]
        ind = get_subsample_ind(kp, H, W, N, n=20)
        if ind is None:
            return [],[],[],[]
        if max(ind) > N:
            print 'sampled a value greater than N!'
    
    if max(ind) > ft.shape[0]:
        print 'ind > ft shape', ft.shape, sampled_ind
        print ind
    if len(ft[ind,:].shape) == 3:
        print ind.shape, ft.shape
    Dx = np.exp(data.kde.score_samples(ft[ind,:])-data.sc_max)
    
    h = data.convert_h2joint(h[:,ind])
    y = get_y(kp, ind, H, W)
    
    # compute Dxy = Dx * P(y|x)
    h_sm = score2prob(h)
    Dxy = np.tile(Dx[np.newaxis,:],[h_sm.shape[0],1]) * h_sm
        
    return Dxy, h, y, ind

def run_net_multiscale(net, im, im_scales):
    h_i_set = []
    for i_scale in img_scales:
        im_sc = resize_util.resize_image_by_scale(im, i_scale)
        _,hi,Hi,Wi = RUN.run_net(net, im_sc, ft_layer, sc_layer)
        hi_up = resize_util.upsample_h_scores(hi, Hi, Wi, H, W)
        h_i_set.append(hi_up[:,:,np.newaxis])
    hi_all = np.dstack(h_i_set)
    h_avg = hi_all.mean(axis=2)
    return h_avg

def load_image_and_rescale(im_id, data, height):
    im = data.load_image(im_id)
    im_sc, o_scale = resize_util.resize_img_by_height_safe(im, height=height)
    return im,im_sc,o_scale


def downsample_kp(kp,scale=8.0):
    return np.uint32(np.floor(kp / scale))

def convert_and_scale_kp(kp, data, scale):
    kp = data.convert_kp2joint(kp)
    kp_s = downsample_kp(kp, scale=scale) 
    return kp, kp_s


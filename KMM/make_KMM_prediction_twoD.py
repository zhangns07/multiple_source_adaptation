import numpy as np
import sklearn.svm as svm

SOURCE_TYPE = ["kitchen","books"]; SOURCE_TYPE_I = 'KB'
#SOURCE_TYPE = ["dvd","electronics"]; SOURCE_TYPE_I = 'DE'

split=2
datadir = '../data/'
sigma = 0.00078
ret_file = 'split_'+str(split)+'_source_'+SOURCE_TYPE_I+'_KMM.csv'

# load traning data and weights
N = 2559 # number of features
X_train = np.zeros((0,N))
Y_train = np.array([])
train_sample_rate = 0.5
for source_type in SOURCE_TYPE:
    X_loaded = np.genfromtxt(datadir+'exp'+str(split)+'/'+source_type+'.X.train.csv', delimiter=',')
    Y_loaded = np.genfromtxt(datadir+'exp'+str(split)+'/'+source_type+'.train.labels', delimiter=',')
    idx = range(0, int(len(X_loaded)*train_sample_rate))
    X_train = np.vstack((X_train,X_loaded[idx,:]))
    Y_train = np.hstack((Y_train,Y_loaded[idx]))

ntrain = len(X_train)

ret = [] # stores: [lam, test error]
for lam in np.linspace(0,1,11):
    if lam == 0:
        lam = 0

    if lam==1:
        lam=1
    LAM = np.array([lam,1-lam])

    # load weights
    weight_file = 'twoD/split_'+str(split)+'_source_'+SOURCE_TYPE_I+'_targetmix_'+str(lam)+'_sigma_'+str(sigma)+'.KMM_sol.csv'
    weights_train = np.genfromtxt(weight_file,delimiter = ',')
    weights_train = np.maximum(weights_train,0)

    # load testing data
    X_test = np.zeros((0,N))
    Y_test = np.array([])
    for i,source_type in enumerate(SOURCE_TYPE):
        X_loaded = np.genfromtxt(datadir+'exp'+str(split)+'/'+source_type+'.X.test.csv', delimiter=',')
        Y_loaded = np.genfromtxt(datadir+'exp'+str(split)+'/'+source_type+'.test.labels', delimiter=',')
        idx = range(0, int(len(X_loaded)*LAM[i]))
        X_test = np.vstack((X_test,X_loaded[idx,:]))
        Y_test = np.hstack((Y_test,Y_loaded[idx]))


    # load svm object
    svr = svm.SVR(kernel='rbf',  gamma=sigma, C=8.0, shrinking=False, cache_size=200, verbose=False)

    # since sample_weight rescales C, so make sample_weight sum up to number of training samples
    svr.fit(X_train, Y_train, sample_weight = ntrain*weights_train)
    Y_test_pred = svr.predict(X_test)
    error = sum((Y_test - Y_test_pred)**2)/len(Y_test) 
    print [lam,error]
    ret.append([lam,error])

np.savetxt(ret_file, ret, delimiter=",")


#### Test if sample_weight works. At least it returns different prediction results.
#weights = np.random.rand(len(Y_train))
#svr.fit(X_train,Y_train, sample_weight = weights)
#y_pred_wt = svr.predict(X_train)
#sum((y_pred_wt - y_pred)**2) / len(y_pred)


#### Test if sklearn.svm gives same results as libsvm. YESSSSS! GOOD. 
#svr.fit(X_train,Y_train)
#
#y_pred = svr.predict(X_train)
#y_pred_libsvm = np.genfromtxt('/home/nz695/domainadap/KMM/compare_svm_sklearn/books.libsvm.pred',delimiter = ',')
#
#sum((y_pred - y_pred_libsvm)**2) / len(y_pred)
#3.1636399093529687e-08


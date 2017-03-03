import numpy as np
import sklearn.svm as svm

SOURCE_TYPES = ["kitchen","dvd","books","electronics"]
TARGET_TYPES = [["kitchen","books"],
        ["kitchen","electronics"],
        ["dvd","books"],
        ["dvd","electronics"]]
TARGET_TYPES_I = ['KB','KE','DB','DE']

#TARGET_TYPES =[ ["kitchen"],
#        ["dvd"],
#        ["books"],
#        ["electronics"],
#        ["kitchen","dvd"],
#        ["books","electronics"],
#        ["dvd","books","electronics"],
#        ["kitchen","books","electronics"],
#        ["kitchen","dvd","electronics"],
#        ["kitchen","dvd","books"],
#        ["kitchen","dvd","books","electronics"]]
#TARGET_TYPES_I =['K','D','B','E','KD','BE','DBE','KBE','KDE','KDB','KDBE']


#split=1
for split in xrange(1,11):
    datadir = '../data/'
    sigma = 0.00078
    ret_file = 'split_'+str(split)+'_4sources_KMM.csv'
    
    # load traning data and weights
    N = 2559 # number of features
    X_train = np.zeros((0,N))
    Y_train = np.array([])
    train_sample_rate = 0.25
    for source_type in SOURCE_TYPES:
        X_loaded = np.genfromtxt(datadir+'exp'+str(split)+'/'+source_type+'.X.train.csv', delimiter=',')
        Y_loaded = np.genfromtxt(datadir+'exp'+str(split)+'/'+source_type+'.train.labels', delimiter=',')
        idx = range(0, int(len(X_loaded)*train_sample_rate))
        X_train = np.vstack((X_train,X_loaded[idx,:]))
        Y_train = np.hstack((Y_train,Y_loaded[idx]))
    
    ntrain = len(X_train)
    
    ret = [] # stores: [lam, test error]
    
    # load testing data
    for i,target_types in enumerate(TARGET_TYPES):
        X_test = np.zeros((0,N))
        Y_test = np.array([])
        test_sample_rate = 0.5
    
        for target_type in target_types:
            X_loaded = np.genfromtxt(datadir+'exp'+str(split)+'/'+target_type+'.X.test.csv', delimiter=',')
            Y_loaded = np.genfromtxt(datadir+'exp'+str(split)+'/'+target_type+'.test.labels', delimiter=',')
            idx = range(0, int(len(X_loaded)*test_sample_rate))
            X_test = np.vstack((X_test,X_loaded[idx,:]))
            Y_test = np.hstack((Y_test,Y_loaded[idx]))
    
        # load weights
        weight_file = 'fourD/split_'+str(split)+'_4sources_target_'+TARGET_TYPES_I[i]+'_sigma_'+str(sigma)+'.KMM_sol.csv'
        weights_train = np.genfromtxt(weight_file,delimiter = ',')
    
        # load svm object
        svr = svm.SVR(kernel='rbf',  gamma=sigma, C=8.0, shrinking=False, cache_size=200, verbose=False)
        
        # since sample_weight rescales C, so make sample_weight sum up to number of training samples
        svr.fit(X_train, Y_train, sample_weight = ntrain*weights_train)
        Y_test_pred = svr.predict(X_test)
        error = sum((Y_test - Y_test_pred)**2)/len(Y_test) 
        ret.append([i,error])
        
    np.savetxt(ret_file, ret, delimiter=",")
        
        

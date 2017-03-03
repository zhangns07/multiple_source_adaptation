#!/usr/bin/python
import sys
import numpy as np
from math import *

if len(sys.argv) < 0 :
	print "Usage: " + sys.argv[0] + " all.txt D1.txt all-on-D1.pred"
	sys.exit(1)

allinput = open(sys.argv[1]).readlines()
allpred = map(float,open(sys.argv[3]).readlines())
#allinput = open(a1).readlines()
#allpred = map(float,open(a3).readlines())


allinputdict = {}
allpreddict = {}
domainpreddict = {}
for i in range(len(allinput)):
	allinputdict[allinput[i]] = 0
	allpreddict[allinput[i]] = allpred[i]
	domainpreddict[allinput[i]] = -1

domain = open(sys.argv[2]).readlines()
for j in range(len(domain)):
	allinputdict[domain[j]]+=1
	domainpreddict[domain[j]] = allpreddict[domain[j]]

empdist = np.array(allinputdict.values()) / float(len(domain))

np.savetxt('empdist.prob',empdist)
np.savetxt('domain.pred',np.array(domainpreddict.values()))
np.savetxt('train.on.domain.pred',np.array(allpreddict.values()))

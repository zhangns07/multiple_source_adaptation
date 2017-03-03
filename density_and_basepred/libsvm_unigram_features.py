#!/usr/bin/python
import sys
from util import *

#MAIN
if len(sys.argv) < 3:
	print "Usage: " + sys.argv[0] + " unigrams.txt data.txt labels.txt"
	sys.exit(-1)

vocab = open(sys.argv[1]).readlines()
lines = open(sys.argv[2]).readlines()
labels = map(str.strip,open(sys.argv[3]).readlines())

if(len(labels) != len(lines)):
	print "Error: data file and label file must be same length"
	sys.exit(-1)

unigrams = {}
for word in vocab:
	uni = word.split()[0]
	index = int(word.split()[1])
	unigrams[uni] = index

print_libsvm_format(lines, labels, unigrams)

#!/usr/bin/python
import sys
from math import *
from util import *

if len(sys.argv) < 0 :
	print "Usage: " + sys.argv[0] + " pred.1 [pred.2 ...]"
	sys.exit(1)


preds = []
for i in range(len(sys.argv)-1):
	pred_lines = map(float,open(sys.argv[i+1]).readlines())
	preds.append([])
	for j in range(len(pred_lines)):
		preds[i].append(pred_lines[j]);

num_points = len(preds[0])
num_domains = len(preds)

for i in range(num_points):
	p = 0
	n = 0
	for j in range(num_domains):
		if preds[j][i]!=-1:
			p+=preds[j][i]
			n+=1
	print p/n

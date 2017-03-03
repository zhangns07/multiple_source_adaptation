#!/usr/bin/python
import sys

if len(sys.argv) < 2 :
	print "Usage: " + sys.argv[0] + " fsmsample.txt"
	sys.exit(1)

statedict = {}
statedict['0'] = ''

lines = open(sys.argv[1]).readlines()
#lines = open('fsmpath').readlines()
curr = '0'
for i in range(len(lines)):
	path = lines[i].split()
	if len(path)==1:
		print statedict[path[0]]
	else:
		if path[0]!=curr:
			del statedict[curr]
			curr = path[0]
		statedict[path[1]] = statedict[path[0]]+' '+path[2]
	


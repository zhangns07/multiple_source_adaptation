#!/usr/bin/python
import sys
import random

if len(sys.argv) < 3:
	print "Usage: " + sys.argv[0] + " file.txt seed"
	sys.exit(1)

random.seed(int(sys.argv[2]))

lines = open(sys.argv[1]).readlines()
random.shuffle(lines)

for line in lines:
	print line.strip()

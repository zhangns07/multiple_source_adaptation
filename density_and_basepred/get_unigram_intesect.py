#!/usr/bin/python
import sys
from util import *
from sets import *


if len(sys.argv) < 3:
	print """Usage: """ + sys.argv[0] + """ cutoff data1.txt [data2.txt ...]
	Returns the unigrams found in the intersection of all datasets, each
	occuring at least 'cutoff' times."""
	sys.exit(-1)

cutoff = int(sys.argv[1])
s = []
d = []
for i in range(2,len(sys.argv)):
	lines = open(sys.argv[i]).readlines()
	u = find_unigrams(lines, cutoff)
	d.append(u)
	s.append(Set(u.keys()))

inter_set = s[0]
for i in range(1,len(s)):
	inter_set = inter_set.intersection(s[i])

words = inter_set
print "<eps> 0"
print "<s> 1"
print "</s> 2"
print "<unk> 3"
i = 4;
for word in words:
	print word + " " +  str(i) 
	i += 1

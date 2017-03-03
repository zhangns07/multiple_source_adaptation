from math import *
from random import *

def print_libsvm_format(lines, labels, unigrams):

	for i in range(len(labels)):
		s = labels[i]
		f_vector = {}
		
		for token in lines[i].split():
			f_token = token.strip()
			if unigrams.has_key(f_token):
				if f_vector.has_key(unigrams[f_token]):
					f_vector[unigrams[f_token]] += 1
				else:
					f_vector[unigrams[f_token]] = 1

		keys = f_vector.keys()
		keys.sort()
		
		for key in keys:
			s += " " + str(key) + ":" + str(f_vector[key])

		print s


def find_unigrams(lines, min_count=2):
	unigrams = {}
	for line in lines:
		tokens = line.split()

		for token in tokens:
			f_token = token.strip()
			if unigrams.has_key(f_token):
				unigrams[f_token] += 1
			else:
				unigrams[f_token] = 1

	cutoff_unigrams = {}
	index=1

	for token in unigrams.keys():
		if unigrams[token] >= min_count:
			cutoff_unigrams[token] = unigrams[token]
			index += 1

	return cutoff_unigrams

OVERFLOW_LIMIT = 700
def log_plus(a,b=None):
	if b == None:
	#simply return original argument
		return a	

	if a - b > OVERFLOW_LIMIT:
		return b
	if b - a > OVERFLOW_LIMIT:
		return a
	if a < b:
		return a - log(1 + exp(a-b))
	else:
		return b - log(1 + exp(b-a))

def log_normalize(p_vec):
	s = None
	for p in p_vec:
		s = log_plus(p, s)

	for i in range(len(p_vec)):
		p_vec[i] = p_vec[i] - s
		
	return p_vec

def sign(x):
	if x >= 0:
		return 1
	else:
		return -1

def gauss_pdf(x, mu, sigma):
	return 1.0 / (sigma * (2 * pi)**0.5) * exp( -(x - mu)**2 / (2.0 * sigma**2) )



class Gauss2d:
	def __init__(self, x_mean, x_stddev, y_mean, y_stddev):
		self.x_m = x_mean
		self.x_s = x_stddev
		self.y_m = y_mean
		self.y_s = y_stddev

	def gen_points(self, num_points):
		x = []
		y = []
	
		for i in range(num_points):
			x.append(gauss(self.x_m,self.x_s))
			y.append(gauss(self.y_m,self.y_s))
	
		return x,y

	def gauss2d_pdf(self, x, y):
		return gauss_pdf(x, self.x_m, self.x_s) * gauss_pdf(y, self.y_m, self.y_s)



class Gauss2d_Unif_Mixture:
	def __init__(self, gausses):
		self.g = gausses

	def gen_points(self, num_points):
		x = []
		y = []

		for i in range(num_points):
			r = randint(0,len(self.g) - 1)	
			xi,yi = self.g[r].gen_points(1)
			x.append(xi[0])
			y.append(yi[0])

		return x,y

	def pdf(self, x, y):
		p = 0
		for d in self.g:
			p += d.gauss2d_pdf(x,y)

		return p / len(self.g)



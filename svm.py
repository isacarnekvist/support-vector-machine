from cvxopt.solvers import qp
from cvxopt.base import matrix
from itertools import product
import numpy as np
import pylab as pl

# Kernels
def linear_kernel(x, y):
	return np.dot(x, y) + 1

def polynomial_kernel(x, y, p):
	return np.power(linear_kernel(x, y), p)

def radial_basis_kernel(x, y, sigma=1):
	return np.exp(-np.sum(np.power((x-y), 2))/(2*sigma**2))

def sigmoid_kernel(x, y, k, delta):
	return np.tanh(k*np.dot(x, y) - delta)

def build_P_matrix(xs, t, K):
	"""
	:param xs: matrix were each row is a feature vector
	:param t: labels {-1, 1} for each of the rows in xs
	:param kernel: handle to a kernel function which only takes two vectors as arguments
				   (rewrite as lambda function if necessary)
	:returns: the matrix t_i t_j K(x_i, x_j)
	"""
	assert len(xs.shape) == 2, 'xs should be 2-dimensional'
	n = np.size(xs,0)
	xs_t = list(zip(xs, t))
	# List comprehension to get all combinations and apply t_i*t_j*K(x_i, x_j)
	P = np.array([x[1]*y[1]*K(x[0], y[0]) for x, y in product(xs_t, xs_t)])
	return P.reshape((n,n))

x = np.array([[1,-1],[-1,1],[-1,-1]])
t = np.array([1,-1,-1])
print(x,t)
print(build_P_matrix(x, t, lambda x, y: polynomial_kernel(x,y,3)))

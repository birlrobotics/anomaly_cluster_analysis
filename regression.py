#!/usr/bin/python
# Filename: regression.py

import numpy as np

# para of Gaussian function
num_basis=31
sigma_basis=0.05

# the 1-d data for test
Y = np.ones([150,1])
len_data=len(Y)


x = np.linspace(0.0, 1.0, len_data)     # the time stamp
C = np.arange(0,num_basis)/(num_basis-1.0)     # the mean of basis func
sigma_basis = sigma_basis      # the sigma of basis func

Phi = np.exp(-.5*(np.array(map(lambda x: x-C, np.tile(x, (num_basis, 1)).T)).T**2 / (sigma_basis**2)))
W = np.dot(np.linalg.inv(np.dot(Phi, Phi.T)), np.dot(Phi, Y)).T

print W

print 'the W*Phi'
print np.dot(W, Phi)


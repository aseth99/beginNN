# 001 0
# 011 1
# 101 1
# 111 0
#harder problem, output based on 1&2 columns, 3rd column redundant

import numpy as np 

#sigmoid function, maps any value to a value between 0-1
def nonlin(x,deriv=False):
	if (deriv==True):
		return x*(1-x)
	return(1/(1+np.exp(-x)))

#input data
X = np.array([[0,0,1],
			[0,1,1],
			[1,0,1],
			[1,1,1]])

#output data
y = np.array([[0,1,1,0]]).T

#seed random number, 
#keeps it predictable for testing (deterministic)
np.random.seed(1)

#synapses, connections between neurons in each layer, initialize weights
syn0 = 2*np.random.random((3,1)) - 1

for i in range(10000):

	#forward propogation
	l0 = X
	l1 = nonlin(np.dot(l0,syn0))
	
	#error?
	l1_error = y - l1
	
	#multiply hwo much we missed by slope of sigmoid... not sure what this is...
	l1_delta = l1_error * nonlin(l1,True)

	#update weights
	# syn1 += l1.T.dot(l2_delta)
	syn0 += np.dot(l0.T,l1_delta)

print("output after training")
print(l1)














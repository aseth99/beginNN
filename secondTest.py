# 001 0
# 011 1
# 101 1
# 111 0
#harder problem, output based on 1&2 columns, 3rd column redundant
#CALLED NON LINEAR PATTERN, 1 to 1 relationship between COMBO of inputs

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
np.random.seed(1000)

#synapses, connections between neurons in each layer, initialize weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1 #3 input nodes, 4 output nodes, first layer, amplifies correlation
syn1 = 2*np.random.random((4,1)) - 1 #4 input nodes from syn0, 1 output node for our guess at output

for i in range(60000):

	#forward propogation, feed forward through the layers
	l0 = X
	l1 = nonlin(np.dot(l0,syn0))
	l2 = nonlin(np.dot(l1,syn1))
	
	#error?
	#l1_error = y - l1
	#now we getting error for 2nd layer (our output, which will be right size to subtract from y to get error!)
	l2_error = y - l2
	#lets print error..only a couple times tho
	if((i%10000)==0):
		print("ERROR: " + str(np.mean(np.abs(l2_error))))
	
	#multiply hwo much we missed by slope of sigmoid... not sure what this is...
	#prev confusion: we do this because our "confidence" value is gona be between 0 and 1
	#if its near 1 or near 0, means its pretty confident in answer
	#sigmoid function has higher slope in middle and low slope near 0 & 1
	#thus this will either punish low confidence answer by adjusting them, or reward higher confidence answers by not affecting them
	#l2 delta will adjust based on final result
	l2_delta = l2_error * nonlin(l2,deriv=True)

	#did l1 val contrubute to l2 error? l1 output.syn1 is predicted l2
	#l1 error calculated from l2 error --> BACKPROPAGATING
	l1_error = l2_delta.dot(syn1.T)
	#l1 delta will adjust based on l2 vals
	l1_delta = l1_error * nonlin(l1,deriv=True)

	#update weights using delta
	syn1 += l1.T.dot(l2_delta)
	syn0 += np.dot(l0.T,l1_delta)

print("output after training")
print(l2)














import numpy as np

class Neural_Network(object):
	#instantiates consts & variables, "self."" makes it accessible
	def __init__(self):
		#def hyperparameters
		self.inputLayerSize = 2
		self.outputLayerSize = 1
		self.hiddenLayerSize = 3

		#weighs (parameters)
		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)

		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)


	def forward(self, X):
		#prop inputs throuhg network
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3)
		return yHat


	def sigmoid(self, z):
		return(1/(1+np.exp(-z)))

	#derivative of sigmoid fxn
	def sigmoidPrime(self, z):
		return np.exp(-z)/((1+np.exp(-z))**2)

	def costFunction(self, X, y):
		self.yHat = self.forward(X)
		costCalc = 0.5*(sum(y-self.yHat)**2)
		return costCalc

	def costFunctionPrime(self, X, y):
		#derivative with respect to W1&W2

		self.yHat = self.forward(X)
		delta3 = np.multiply(-(y-self.yHat),self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T,delta3)

		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2)

		return dJdW1, dJdW2

	def getParams(self):
		#w1 & w2 rolled into vectors
		params = np.concatenate((self.W1.ravel(),self.W2.ravel()))
		return params

	def setParams(self, params):
		#set w1&w2 using single parameter vector
		W1_start = 0
		W1_end = self.hiddenLayerSize * self.inputLayerSize
		self.W1 = np.reshape(params[W1_start:W1_end],(self.inputLayerSize,self.hiddenLayerSize))
		W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
		self.W2 = np.reshape(params[W1_end:W2_end],(self.hiddenLayerSize,self.outputLayerSize))
		
	def computeGradients(self,X,y):
		dJdW1, dJdW2 = self.costFunctionPrime(X,y)
		return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


	def computeNumericalGradient(self,N,X,y):
		paramsInitial = N.getParams()
		numgrad = np.zeros(paramsInitial.shape)
		perturb = np.zeros(paramsInitial.shape)
		e = 1e-4

		for p in range(len(paramsInitial)):
			#set perturbation vector
			perturb[p] = e
			N.setParams(paramsInitial + perturb)
			loss2 = N.costFunction(X,y)

			N.setParams(paramsInitial - perturb)
			loss1 = N.costFunction(X,y)

			#compute numerical gradient
			numgrad[p] = (loss2-loss1)/(2*e)

			#retunr val we changed to 0
			perturb[p] = 0

		#return params to orginal
		N.setParams(paramsInitial)

		return numgrad








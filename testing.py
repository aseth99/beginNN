from nnClass import Neural_Network
from trainClass import trainer
import numpy as np
# from numpy import linalg

#testing
X = np.array([[3,5],[5,1],[10,2]])
y = np.array([[.75],[.82],[.93]])

NN = Neural_Network()

T = trainer(NN)

T.train(X,y)

cost1 = NN.costFunction(X,y)

dJdW1, dJdW2 = NN.costFunctionPrime(X,y)

print(NN.costFunctionPrime(X,y))
print(NN.forward(X))
print(y)
# print(dJdW1)
# print(dJdW2)
#costfunction is impacted by these

# numgrad = NN.computeNumericalGradient(NN,X,y)
# grad= NN.computeGradients(X,y)

# print((np.linalg.norm(grad-numgrad))/(np.linalg.norm(grad+numgrad)))

# print(numgrad)
# print(grad)

# yHat = NN.forward(X)
# print(yHat)
# print(y)


def sigmoid(z):
	"""
		To calculate the sigmoid of z
	"""

	s = 1/(1+np.exp(-z))

	return s


def propagation(w,b,X,Y):
	"""
		Implementing the cost function and its gradient for the neural network

		Arguments:
		w - weights for each neuron (num_px*num_py*3,1)
		b - bias
		X - input matrix
		Y - output matrix

		Return:
		cost: the cost value for the input
		dw: gradient with respect to w , matrix of size w
		db: gradient with respect to b

	"""

	#activation function
	A = sigmoid(np.dot(w.T,X) + b)
	cost = -1/m*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

	dw = 1/m*np.dot(X,(A-Y).T)
	db = 1/m*np.sum(A-Y)

	grads = {"dw":dw , "db" : db}

	return grads, cost


def optimize(w,b,X,Y,num_iterations,learning_rate):
	"""
		Implementing the gradient descent algorithm 

		Return:
		params - the final value of the parameters or the weight

	"""

	for i in range(num_iterations):

		grad, cost = propagation(w,b,X,Y)
		dw = grad["dw"]
		db = grad["db"]

		w = w - learning_rate*dw
		b = b - learning_rate*db

	params = {"w":w , "b" :b}
	grad = {"dw":dw , "db":db}

	return params,grad, cost 
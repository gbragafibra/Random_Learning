import numpy as np 

class Layer:

	def __init__(self):
		self.input = None
		self.output = None

	def forward(self, input):

		pass


	def backward(self, output_grad, learning_rate):

		pass

# General activation Layer

class Activation(Layer):

	def __init__(self, activation, activation_prime):
		self.activation = activation
		self.activation_prime = activation_prime

	def forward(self, input):
		self.input = input 

		return self.activation(self.input)

	def backward(self, output_grad, learning_rate):

		return np.multiply(output_grad, self.activation_prime(self.input))


class ReLU(Activation):
    def __init__(self):
        def relu(x):
            if isinstance(x, tuple):  # For tuples
                return tuple(np.maximum(0, elem) for elem in x)
            else:  # A single array
                return np.maximum(0, x)

        def relu_prime(x):
            if isinstance(x, tuple):  # For tuples
                return tuple(np.where(elem > 0, 1, 0) for elem in x)
            else:  # For array
                return np.where(x > 0, 1, 0)

        super().__init__(relu, relu_prime)


class Softmax(Layer):
	def forward(self, input):

		expo = np.exp(input)
		self.output = expo/np.sum(expo)

		return self.output

	def backward(self, output_grad, learning_rate):
        
		n = np.size(self.output)

		return np.dot((np.identity(n) - self.output.T) * self.output, output_grad)	
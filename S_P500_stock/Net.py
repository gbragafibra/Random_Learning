import numpy as np


# General Layer class

class Layer:

	def __init__(self):
		self.input = None
		self.output = None

	def forward(self, input):

		pass

	def backward(self, Δ_out, α):
		# Δ_out because ∇ isn't valid :)
		# α is learning rate

		pass

# Base activation layer

class Activation(Layer):

	def __init__(self, activation, activation_prime):
		self.activation = activation
		self.activation_prime = activation_prime

	def forward(self, input):
		self.input = input

		return self.activation(self.input)

	def backward(self, Δ_out, α):

		return Δ_out * self.activation_prime(self.input)


class Sigmoid(Activation):

	def __init__(self):

		def sigmoid(x):
			return 1/(1 + np.exp(-x))

		def sigmoid_prime(x):
			s = sigmoid(x)

			return s * (1 - s)

		super().__init__(sigmoid, sigmoid_prime)

x = np.array([1,2,3])
sig_layer = Sigmoid()
out = sig_layer.forward(x)
back_test = sig_layer.backward(x, 0.05)
print(out)
print(back_test)

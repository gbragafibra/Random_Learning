import numpy as np
import matplotlib.pyplot as plt


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
		super().__init__(sigmoid, sigmoid_prime)
		"""
		Dont need an instance.
		i.e. Can call Sigmoid.sigmoid()
		directly, with @staticmethod 
		"""

	@staticmethod 
	def sigmoid(x):
		return 1/(1 + np.exp(-x))
	
	@staticmethod
	def sigmoid_prime(x):
		s = Sigmoid.sigmoid(x)

		return s * (1 - s)

		



class Tanh(Activation):

	def __init__(self):
		super().__init__(tanh, tanh_prime)

	@staticmethod
	def tanh(x):
		return np.tanh(x)
	@staticmethod
	def tanh_prime(x):
		return (1 - np.tanh(x)**2)

		

"""
KAN paper:
https://arxiv.org/pdf/2404.19756

Also a nice resource
https://medium.com/@sidhuser/from-scratch-implementation-of-kolmogorov-arnold-networks-kan-and-mlp-14a021376386
"""

# Kolmogorov-Arnold Layer
class KAN(Layer):
	"""
	Using sigmoids as basis, instead
	of typical 3rd degree (?) B-splines
	"""
	def __init__(self, input_dim, n_basis_funcs):
		self.input_dim = input_dim
		self.n_basis_funcs = n_basis_funcs

		#Assuming b = 0
		self.w = np.random.randn(n_basis_funcs, input_dim)
		self.coefficients = np.random.randn(n_basis_funcs)

	def forward(self, x):
		self.x = x #To use in backward pass
		σ_out = Sigmoid.sigmoid(np.dot(x, self.w.T))
		self.σ_out = σ_out.reshape(-1, 1) #To use in backward pass
		
		ϕ = np.dot(σ_out.reshape(-1, 1), self.coefficients.reshape(1, -1))

		return ϕ

	def backward(self, Δ_out, α):

		Δ_out = Δ_out.reshape(-1, 1)
		
		coef_Δ = np.dot(Δ_out, self.σ_out)
		#coef_Δ = np.sum(Δ_out * self.σ_out, axis = 0)

		#dσ_out = np.dot(Δ_out, self.coefficients)
		dσ_out = Δ_out * self.coefficients.reshape(1, -1)

		σ_prime = Sigmoid.sigmoid_prime(np.dot(self.x, self.w.T))

		w_Δ = np.dot((dσ_out * σ_prime).T, self.x.reshape(-1, 1))

		self.w -= α * w_Δ
		self.coefficients -= α * coef_Δ

		#input grad
		"""
		Same thing as w_Δ but taking into account
		dx, so we have dot product including
		the weights
		"""
		in_Δ = np.dot(dσ_out * σ_prime, self.w)

		return in_Δ

# Mean-squared error loss function
def mse(y_true, y_pred):
	return np.mean((y_true - y_pred)**2)

def mse_prime(y_true, y_pred):
	return 2 * (y_pred - y_true)/np.size(y_true)


if __name__ == "__main__":

	x = np.linspace(0, 1, 100).reshape(-1, 1)
	y_real = np.sin(2 * np.pi * x)

	#Initialize Layer
	KAN = KAN(x.shape[0], 10)


	α = 0.05
	epochs = 1000


	for epoch in range(epochs + 1):
		ϕ_out = KAN.forward(x)
		#Loss
		L = mse(y_real, ϕ_out)

		back_Δ_compute = mse_prime(y_real, ϕ_out)

		back_Δ = KAN.backward(back_Δ_compute, α)

		if epoch % 100 == 0:
			print(f"Epoch {epoch}/{epochs}. Loss: {L}")


	plt.plot(x, ϕ_out, "k.", label = "$\\hat{y}$")
	plt.plot(x, y_real, "r", label = "y")
	plt.xlabel("x")
	plt.ylabel("ϕ_out")
	plt.legend()
	plt.show()
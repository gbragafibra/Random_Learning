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
		s = sigmoid(x)

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
		self.σ_out = σ_out #To use in backward pass
		ϕ = np.dot(σ_out, self.coefficients)

		return ϕ

	def backward(self, Δ_out, α):
		
		coef_Δ = np.dot(Δ_out.T, self.σ_out)

		dσ_out = np.dot(Δ_out, self.coefficients)
		σ_prime = Sigmoid.sigmoid_prime(np.dot(self.x, self.w.T))

		w_Δ = np.dot((dσ_out * σ_prime).T, self.x)

		self.w -= α * w_Δ
		self.coefficients -= α * coef_Δ

		in_Δ = np.dot(dσ_out * σ_prime, self.w)

		return in_Δ

# Mean-squared error loss function
def mse(y_true, y_pred):
	return np.mean((y_true - y_pred)**2)

def mse_prime(y_true, y_pred):
	return 2 * (y_pred - y_true)/np.size(y_true)

x = np.array([1,2,3])
KAN = KAN(x.shape[0], 2)
out = KAN.forward(x)
back = KAN.backward(x, 0.05)
print(out)
print(back)
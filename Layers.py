import numpy as np 

#General Layer class
class Layer:

	def __init__(self):
		self.input = None
		self.output = None

	def forward(self, input):

		pass


	def backward(self, output_grad, learning_rate):

		pass


"""
Based on:
# http://yingzhenli.net/home/pdf/imperial_dlcourse2022_rnn_notes.pdf
# https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks
"""
class RNN(Layer):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.Wx = np.random.randn(hidden_size, input_size) # Weights relative to input
        self.Wh = np.random.randn(hidden_size, hidden_size)  # Weights relative to hidden states
        self.Wy = np.random.randn(output_size, hidden_size)  # Weights relative to output
        self.bh = np.zeros((hidden_size, 1)) #Bias relative to activation
        self.by = np.zeros((output_size, 1)) #Bias relative to output

    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_states = []
        
        h_old = np.zeros((self.hidden_size, 1))
        for x in inputs:
            h = np.tanh(np.dot(self.Wx, x) + np.dot(self.Wh, h_old) + self.bh) # Using tanh
            y = np.dot(self.Wy, h) + self.by       
            self.hidden_states.append(h)
            h_old = h
            
        return y[-1,-1] # Original y is a matrix (something like 1x10); We want last value from the vector

    def backward(self, output_grad, learning_rate):
        dWx, dWh, dWy, dbh, dby, dh_new = np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.Wy), np.zeros_like(self.bh), np.zeros_like(self.by), np.zeros_like(self.bh)
        
        for t in reversed(range(len(self.inputs))):
            x = self.inputs[t]
            h = self.hidden_states[t]
            dy = output_grad
            
            dWy += np.sum(np.dot(dy, h.T), axis = 0, keepdims=True) #Initially with different shapes; So needed to do this
            dby += np.sum(dy)
            
            dh = np.dot(self.Wy.T, dy) + dh_new
            dh_grad = (1 - h * h) * dh # derivative of tanh
            dbh += np.sum(dh_grad.T, axis=0, keepdims=True).T #Initially with different shapes; So needed to do this

            
            dWx += np.dot(dh_grad, x.T)
            dWh += np.dot(dh_grad, self.hidden_states[t - 1].T)
            dh_out = np.dot(self.Wh.T, dh_grad)
        
        
        self.Wx -= learning_rate * dWx
        self.Wh -= learning_rate * dWh
        self.Wy -= learning_rate * dWy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby
        
        return dh_out

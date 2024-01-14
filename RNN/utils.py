import numpy as np 
import matplotlib.pyplot as plt 

#Mean squared error
def mse(y_true, y_pred):
	return np.mean((y_true - y_pred)**2)

def mse_prime(y_true, y_pred):
	return 2 * (y_pred - y_true)/np.size(y_true)




def predict(network, input):
	output = input 
	for layer in network:
		output = layer.forward(output)

	return output


#For temperature time-series forecasting
def train(network, loss, loss_prime, x_train, y_train,
          epochs=10, learning_rate=0.01, verbose=True):


    predic_labels = []
    for e in range(epochs):
        training_loss = 0   
        
        # Forward pass
        output = predict(network, x_train)
        predic_labels.append(output)
        training_loss = loss(y_train, output)
        
        # Compute grad
        grad = loss_prime(y_train, output)
        
        # Backward pass
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
    

        if verbose:
            print(f"Epoch {e + 1}/{epochs}, Training Loss = {training_loss}")
            print(f"Predicted temperature: {output:.1f}, Real temperature: {y_train}")

    plt.plot(np.arange(1, epochs + 1), np.full(20, y_train), "k", label = "Temperature at $\\tau$")
    plt.plot(np.arange(1, epochs + 1), predic_labels, "r.", label = "Predicted temperature at $\\tau$")
    plt.xlabel("Epochs")
    plt.ylabel("Temperature value")
    plt.legend()
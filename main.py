# XOR neural network
import numpy as np
import time

#variables
n_hidden = 10
n_in = 10
n_out = 10
n_sample = 300

#hyperparameters
learning_rate = 0.01
momentum = 0.9

np.random.seed(0)

# activation function
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh_prime(x):
    return 1- np.tanh(x)**2

#x is input
#t is transpose for matrix multiplication, basicall the inverse of the input
#V and W are our layers
#bv and bw are our biases
def train(x, t, V, W, bv, bw):
    # forward propagation -- matrix multiply + biases
    A = np.dot(x, V) + bv
    Z = np.tanh(A)

    B = np.dot(Z, W) +bw
    Y = sigmoid(B)

    # Backward propogation
    Ew = Y - t
    Ev = tanh_prime(A) * np.dot(W, Ew)

    #predict our loss
    dW = np.outer(Z, Ew)
    dV = np.outer(x, Ev)
    
    # Cross Entropy because we are doing Classification (because it gives better results usually)
    loss = -np.mean( t * np.log(Y) + (1 - t) * np.log(1-Y))

    return loss, (dV, dW, Ev, Ew)

def predict(x, V, W, bv, bw):
    A = np.dot(x, V) + bv
    B = np.dot(np.tanh(A), W) + bw
    return (sigmoid(B) > 0.5).astype(int)

# Create layers
V = np.random.normal(scale=0.1, size=(n_in, n_hidden))
W = np.random.normal(scale=0.1, size=(n_in, n_hidden))

bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

# generate our data
X = np.random.binomial(1, 0.5, (n_sample, n_in))
T = X ^ 1
params = [V, W, bv, bw ]


# TIME TO START TRAINING!
for epoch in range(100):
    error = []
    update = [0]*len(params)

    t0 = time.clock()
    # For each datapoint, update the weights of the network
    for i in range(X.shape[0]):
        loss, grad = train(X[i], T[i], *params)
        # Update loss
        for j in range(len(params)):
            params[j] -= update[j]
            
        for j in range(len(params)):
            update[j] = learning_rate * grad[j] + momentum * update[j]

        error.append(loss)
    print('Epoch: %d, Loss: %.8f, Time: %fs') %(epoch, np.mean(error), time.clock()-t0)

x = np.random.binomial(1, 0.5, n_in)
print('XOR Prediction')
print(x)
print(predict(x, *params))

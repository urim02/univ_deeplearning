import numpy as np

class shallow_neural_network():
    #Model Initialization
    def __init__(self, num_input_features, num_hiddens):
        self.num_input_features = num_input_features
        self.num_hiddens = num_hiddens

        self.W1 = np.random.normal(size = (num_hiddens, num_input_features))
        self.b1 = np.random.normal(size = (num_hiddens, 1))
        self.W2 = np.random.normal(size = (1, num_hiddens))
        self.b2 = np.random.normal(size = (1, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # Forward Pass
    def predict(self, x):
        z1 = np.matmul(self.W1, x) + self.b1
        a1 = np.tanh(z1)
        z2 = np.matmul(self.W2, a1) + self.b2
        a2 = self.sigmoid(z2)

        return a2, (z1, a1, z2, a2)
    
    #Backpropagation
    def backward(self, x, y, cache):
        z1, a1, z2, a2 = cache
        m = x.shape[1]

        dz2 = a2 - y
        dW2 = np.dot(dz2, a1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        dz1 = np.dot(self.W2.T, dz2) * (1 - np.power(a1, 2)) 
        dW1 = np.dot(dz1, x.T) / m 
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        return dW1, db1, dW2, db2
    
    # Parameter Update
    def update_params(self, grads, lr=0.1):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def train(self, X, Y, epochs=1000, lr=0.1):
        for epoch in range(epochs):
            a2, cache = self.predict(X)  
            cost = -np.mean(Y * np.log(a2 + 1e-15) + (1 - Y) * np.log(1 - a2 + 1e-15)) 
            grads = self.backward(X, Y, cache) 
            self.update_params(grads, lr)  

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost={cost:.6f}")


X = np.array([[0,0],[0,1],[1,0],[1,1]]).T 
Y = np.array([[0,1,1,0]])

# 모델 생성 및 학습
model = shallow_neural_network(num_input_features=2, num_hiddens=2)
model.train(X, Y, epochs=1000, lr=0.1)

test_inputs = [(1,1), (1,0), (0,1), (0,0)]
for x in test_inputs:
    x_array = np.array(x).reshape(-1, 1)  
    pred, _ = model.predict(x_array)
    print(f"model.predict({x})[0].item() = {pred.item():.6f}")

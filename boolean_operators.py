import math
import random
import matplotlib.pyplot as plt

class LogisticRegressionModel:
    def __init__(self):
        self.w0 = random.random()
        self.w1 = random.random()
        self.b  = random.random()
    
    def sigmoid(self, z):
        return 1.0 / (1.0 + math.exp(-z))
    
    def predict(self, x):
        z = self.w0 * x[0] + self.w1 * x[1] + self.b
        return self.sigmoid(z)
    
    def train(self, X, Y, lr=0.1, epochs=2000):
        n = len(X)
        loss_history = []

        for epoch in range(epochs):
            dw0, dw1, db = 0.0, 0.0, 0.0
            cost = 0.0

            for (x0, x1), y in zip(X, Y):
                a = self.predict((x0, x1))
                cost -= (y * math.log(a) + (1 - y) * math.log(1 - a))

                dw0 += (a - y) * x0
                dw1 += (a - y) * x1
                db  += (a - y)

            cost /= n
            loss_history.append(cost)

            dw0 /= n
            dw1 /= n
            db  /= n

            self.w0 -= lr * dw0
            self.w1 -= lr * dw1
            self.b  -= lr * db
            
        return loss_history
    

def main():
    X = [(0,0), (0,1), (1,0), (1,1)]

    datasets = {
        "AND": [0, 0, 0, 1],
        "OR":  [0, 1, 1, 1],
        "XOR": [0, 1, 1, 0]
    }

    learning_rates = [0.1, 1e-3, 1e-5]
    epochs = 2000
    
    for operator, Y in datasets.items():
        print(f"Operator: {operator}")
        plt.figure(figsize=(8, 6))
        
        for lr in learning_rates:
            model = LogisticRegressionModel()
            loss_history = model.train(X, Y, lr=lr, epochs=epochs)

            predicted_probs = [round(model.predict(x), 4) for x in X]
            print(f"Learning Rate: {lr}")
            print("Predicted Probabilities:", predicted_probs)
            plt.plot(range(epochs), loss_history, label=f"lr={lr}")
            
        plt.title(f"Loss Plot for {operator}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()


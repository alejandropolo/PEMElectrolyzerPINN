import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Function to generate training data
def generate_data(n_samples=1000):
    x = np.linspace(-2, 2, n_samples).reshape(-1, 1)
    y1 = x ** 3  # f(x) = x^3
    y2 = x  # g(x) = x^3 - x
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y1, dtype=torch.float32), torch.tensor(y2, dtype=torch.float32)

# Neural Network Model
class DualOutputNN(nn.Module):
    def __init__(self):
        super(DualOutputNN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, 32),
            nn.Sigmoid(),
            nn.Linear(32, 32),
            nn.Sigmoid()
        )
        self.output1 = nn.Linear(32, 1)  # First output
        self.output2 = nn.Linear(32, 1)  # Second output

    def forward(self, x):
        features = self.hidden(x)
        y1 = self.output1(features)  # f(x) prediction
        y2 = self.output2(features)  # g(x) prediction
        return y1, y2

# Training function
def train(model, x_train, y1_train, y2_train, epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y1_pred, y2_pred = model(x_train)
        loss1 = criterion(y1_pred, y1_train)
        loss2 = criterion(y2_pred, y2_train)
        loss = loss1 + loss2  # Combined loss
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    print("Training complete!")

# Function to plot results
def plot_results(model, x_test):
    model.eval()
    with torch.no_grad():
        y1_pred, y2_pred = model(x_test)
    
    x_test_np = x_test.numpy()
    y1_pred_np = y1_pred.numpy()
    y2_pred_np = y2_pred.numpy()
    
    plt.figure(figsize=(12, 6))

    # Subplot for f(x) = x^3
    plt.subplot(1, 2, 1)
    plt.scatter(x_test_np, x_test_np ** 3, label="True f(x) = x^3", alpha=0.5)
    plt.scatter(x_test_np, y1_pred_np, label="Predicted f(x)", marker='x', s=10)
    plt.legend()
    plt.title("True vs Predicted f(x) = x^3")

    # Subplot for g(x) = x^3 - x
    plt.subplot(1, 2, 2)
    plt.scatter(x_test_np, x_test_np, label="True g(x) = x^3 - x",  alpha=0.5)
    plt.scatter(x_test_np, y2_pred_np, label="Predicted g(x)",  marker='x', s=10)
    plt.legend()
    plt.title("True vs Predicted g(x) = x^3 - x")

    plt.tight_layout()
    plt.show()



# Run the pipeline
def main():
    """
    DualOutputNN.py

    This script defines and trains a dual-output neural network model as an example for the case of 
    a dual-output neural network. It demonstrates how to handle multiple outputs in a single model 
    and train it effectively.

    Functions:
    - main(): The entry point of the script. It generates training data, initializes the dual-output 
        neural network model, trains it using the generated data, and visualizes the results.

    Modules:
    - generate_data: A function to generate input data (x_train) and two corresponding output datasets 
        (y1_train, y2_train) for training the model.
    - DualOutputNN: A class or function that defines the architecture of the dual-output neural network.
    - train: A function to train the neural network model using the provided training data.
    - plot_results: A function to visualize the model's performance on the training data.

    Usage:
    Run this script to train a dual-output neural network and visualize its performance.
    """
    x_train, y1_train, y2_train = generate_data()
    model = DualOutputNN()
    train(model, x_train, y1_train, y2_train)
    plot_results(model, x_train)

if __name__ == "__main__":
    main()

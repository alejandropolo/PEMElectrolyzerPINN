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
    x_train, y1_train, y2_train = generate_data()
    model = DualOutputNN()
    train(model, x_train, y1_train, y2_train)
    plot_results(model, x_train)

if __name__ == "__main__":
    main()

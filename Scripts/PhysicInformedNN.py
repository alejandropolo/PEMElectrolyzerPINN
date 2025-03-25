import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Function to generate data
def generate_data(function, n_samples=100):
    x = torch.linspace(0, 1, n_samples).reshape(-1, 1)
    u_real = function(x)  # Exact solution to PDE u'' = -u
    return x, u_real

# PINN Model Class
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers) - 2:
                layer_list.append(self.activation)
        self.net = nn.Sequential(*layer_list)
        self.apply(initialize_weights)

    def forward(self, x):
        return self.net(x)

    def loss(self, x_phys):
        x_phys.requires_grad = True
        u = self.forward(x_phys)
        
        du_dx = torch.autograd.grad(u, x_phys, torch.ones_like(u), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x_phys, torch.ones_like(du_dx), create_graph=True)[0]
        
        # f = d2u_dx2 + u  # PDE: u'' = -u
        f = du_dx -u # PDE u' = u
        return torch.mean(f**2)
    
    def mse_loss(self, x_data, u_data):
        u_pred = self.forward(x_data)
        return nn.MSELoss()(u_pred, u_data)

# Training Function
def train_pinn(model, optimizer, x_train_mse, y_train_mse, x_train_phys, epochs=1000, lambda_mse=1.0, lambda_phys=1.0):
    for epoch in range(epochs):
        loss_pde = model.loss(x_train_phys)
        loss_mse = model.mse_loss(x_train_mse, y_train_mse)
        
        loss = lambda_phys * loss_pde + lambda_mse * loss_mse
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: PDE Loss = {loss_pde.item():.6f}, MSE Loss = {loss_mse.item():.6f}, Total Loss = {loss.item():.6f}")
        # Print also last loss
        elif epoch == epochs - 1:
            print(f"Epoch {epoch}: PDE Loss = {loss_pde.item():.6f}, MSE Loss = {loss_mse.item():.6f}, Total Loss = {loss.item():.6f}")

# Function to Plot Results
def plot_solution(model, data, x_train_mse, y_train_mse):
    x_real, u_real = data
    
    with torch.no_grad():
        u_pred = model(x_real).numpy()
    
    x_real = x_real.detach().numpy()
    u_real = u_real.detach().numpy()
    x_train_mse = x_train_mse.numpy()
    y_train_mse = y_train_mse.numpy()
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_real, u_real, 'b-', label='Exact Solution')
    plt.plot(x_real, u_pred, 'r--', label='PINN Prediction')
    plt.scatter(x_train_mse, y_train_mse, color='g', label='Training Data (MSE)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    plt.title('Comparison of PINN Solution and Exact Solution')
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Function to generate data
    generating_function = torch.exp  # Exact solution to PDE u'' = -u
    
    # Generate Data
    x_train_mse, y_train_mse = generate_data(generating_function, n_samples=10)
    # Select just the first two points
    x_train_mse = x_train_mse[:2]
    y_train_mse = y_train_mse[:2]
    x_train_phys, _ = generate_data(generating_function, n_samples=1000)  # Larger set for physics loss
    
    # Define Model and Optimizer
    # layers = [1, 20, 20, 20, 1]  # 1D input, 3 hidden layers, 1D output
    layers = [1, 10, 20, 20, 1]  # 1D input, 3 hidden layers, 1D output
    pinn = PINN(layers)
    optimizer = optim.Adam(pinn.parameters(), lr=0.001)
    
    # Train PINN
    train_pinn(pinn, optimizer, x_train_mse, y_train_mse, x_train_phys, epochs=1000, lambda_phys=0)
    
    # Plot results
    plot_solution(pinn, (x_train_phys, generating_function(x_train_phys)), x_train_mse, y_train_mse)
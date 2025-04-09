import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from verifyODE import check_ode

# Function to generate training data
def generate_data(n_samples=1000, f_func=None, g_func=None):
    x = torch.linspace(-0.5, 0.5, n_samples).reshape(-1, 1)
    if f_func:
        y1 = f_func(x)
    else:
        raise ValueError("f_func is required")
    if g_func:
        y2 = g_func(x)
    else:
        raise ValueError("g_func is required")
    return x, y1, y2



# Neural Network Model
class DualOutputNN(nn.Module):
    def __init__(self, y01, y02):
        super(DualOutputNN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, 32),
            nn.Sigmoid(),
            nn.Linear(32, 32),
            nn.Sigmoid()
        )
        self.output1 = nn.Linear(32, 1)  # First output
        self.output2 = nn.Linear(32, 1)  # Second output
        self.y01 = y01
        self.y02 = y02

    def forward(self, x):
        features = self.hidden(x)
        y1 = torch.exp(self.output1(features))*x + self.y01   # f(x) prediction
        y2 = torch.exp(self.output2(features))*x +self.y02 # g(x) prediction
        # y1 = self.output1(features)  # f(x) prediction
        # y2 = self.output2(features)  # g(x) prediction
        return y1, y2
    
    def physics_loss(self, x_phys, ode_residual_f_func, ode_residual_g_func):
        """
        Computes the physics-based loss for the given physical inputs and ODE residual functions.
        Args:
            x_phys (torch.Tensor): The physical input tensor.
            f_values (torch.Tensor): The target values for the first function.
            g_values (torch.Tensor): The target values for the second function.
            ode_residual_f_func (callable): A function that computes the ODE residual for the first function.
            ode_residual_g_func (callable): A function that computes the ODE residual for the second function.
        Returns:
            torch.Tensor: The computed physics-based loss.
        Example of ode_residual_f_func:
            ode_residual_f_func = lambda df_dx, dg_dx, x_phys: df_dx + dg_dx - x_phys
        Example of ode_residual_g_func:
            ode_residual_g_func = lambda dg_dx, x_phys: dg_dx - x_phys
        """
        x_phys.requires_grad = True
        y1, y2 = self.forward(x_phys)
        f_pred, g_pred = y1, y2

        df_dx = torch.autograd.grad(f_pred, x_phys, torch.ones_like(f_pred), create_graph=True)[0]
        dg_dx = torch.autograd.grad(g_pred, x_phys, torch.ones_like(g_pred), create_graph=True)[0]
        
        ode_residual_f = ode_residual_f_func(f_pred, g_pred, df_dx, dg_dx, x_phys)
        ode_residual_g = ode_residual_g_func(dg_dx, x_phys)
        
        return torch.mean(ode_residual_f**2) + torch.mean(ode_residual_g**2)
    
    def mse_loss(self, x_data, f_data, g_data):
        f_pred, g_pred = self.forward(x_data)
        # f_pred, g_pred = u_pred[:, 0:1], u_pred[:, 1:2]
        loss_f = nn.MSELoss()(f_pred, f_data)
        loss_g = nn.MSELoss()(g_pred, g_data)
        return loss_f + loss_g

# Training function
def train(model, x_mse, x_phys, y1_train, y2_train, ode_residual_f_func, ode_residual_g_func, epochs=1000, lr=0.01,
          lambda_mse = 1.0, lambda_phys = 1.0):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        # y1_pred, y2_pred = model(x_train)
        # loss1 = criterion(y1_pred, y1_train)
        # loss2 = criterion(y2_pred, y2_train)
        # loss = loss1 + loss2  # Combined loss
        loss_mse = model.mse_loss(x_mse, y1_train, y2_train)
        physics_loss = model.physics_loss(x_phys, ode_residual_f_func, ode_residual_g_func)
        loss = lambda_phys * physics_loss + lambda_mse * loss_mse
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Total Loss = {loss.item():.6f}, MSE Loss = {loss_mse.item():.6f}, Physics Loss = {physics_loss.item():.6f}")
    print("Training complete!")

# Function to plot results
def plot_results(model, x_test, x_train_mse,f_train,g_train, f_test, g_test, f_func=None, g_func=None):
    model.eval()
    with torch.no_grad():
        y1_pred, y2_pred = model(x_test)
    
    x_test_np = x_test.detach().numpy()
    x_train_mse_np = x_train_mse.detach().numpy()
    y1_pred_np = y1_pred.detach().numpy()
    y2_pred_np = y2_pred.detach().numpy()
    
    plt.figure(figsize=(12, 6))

    # Subplot for f(x)
    plt.subplot(1, 2, 1)
    if f_func:
        plt.scatter(x_test_np, f_func(x_test_np), label="True f(x)", alpha=0.5)
        plt.scatter(x_train_mse_np, f_func(x_train_mse_np), label="Training f(x)", marker='o', s=30, edgecolor='k')
    else:
        plt.scatter(x_test_np, f_test, label="True f(x)", alpha=0.5)
        plt.scatter(x_train_mse_np, f_train, label="Training f(x)", marker='o', s=30, edgecolor='k')
    plt.scatter(x_test_np, y1_pred_np, label="Predicted f(x)", marker='x', s=10)
    plt.legend()
    plt.title("True vs Predicted f(x)")

    # Subplot for g(x)
    plt.subplot(1, 2, 2)
    if g_func:
        plt.scatter(x_test_np, g_func(x_test_np), label="True g(x)", alpha=0.5)
        plt.scatter(x_train_mse_np, g_func(x_train_mse_np), label="Training g(x)", marker='o', s=30, edgecolor='k')
    else:
        plt.scatter(x_test_np, g_test, label="True g(x)", alpha=0.5)
        plt.scatter(x_train_mse_np, g_train, label="Training g(x)", marker='o', s=30, edgecolor='k')
    plt.scatter(x_test_np, y2_pred_np, label="Predicted g(x)", marker='x', s=10)
    plt.legend()
    plt.title("True vs Predicted g(x)")

    plt.tight_layout()
    plt.show()



# Run the pipeline
def main():
    ## Example of use
    # # Define the functions using lambda
    # f_func = lambda x: x**2# - x
    # g_func = lambda x: x**2
    # ode_residual_f_func = lambda f_values, g_values, df_dx, dg_dx, x_phys: df_dx + f_values  - (dg_dx+g_values)
    # ode_residual_g_func = lambda dg_dx, x_phys: dg_dx - 2*x_phys
    # x_train_mse, y1_train, y2_train = generate_data(f_func=f_func, g_func=g_func, n_samples=10)
    # x_phys, _, _ = generate_data(f_func=f_func, g_func=g_func, n_samples=1000) 

    df = pd.read_csv('membrane_thinning_voltage_data.csv')
    # Rename columns to [time, membrane_thickness, voltage, power]
    df.columns = ['Time', 'memThickness', 'V','VCheck','I','ICheck','P','PCheck']

    df_constants = pd.read_csv('constants.csv')
    k1 = df_constants['k1'].mean()
    k2 = df_constants['k2'].mean()
    k3 = df_constants['k3'].mean()
    Area_cell = 680
    P = (df['P']/Area_cell).mean()


    ## TODO: Check that the values are the same as the ones used to generate the data
    final_time = 1e4
    k4 = 1e-6*final_time  # Thinning rate constant [cm/hour]


    x_train_mse = torch.tensor(df['Time'].values, dtype=torch.float32).reshape(-1, 1)
    x_phys = torch.tensor(df['Time'].values, dtype=torch.float32).reshape(-1, 1)


    y1_train = torch.tensor(df['VCheck'].values, dtype=torch.float32).reshape(-1, 1)
    y2_train = torch.tensor(df['memThickness'].values, dtype=torch.float32).reshape(-1, 1)

    def ode_equation(x, f_values, g_values, df_dx, dg_dx,constants):
        return 2 * f_values * df_dx - k1 * df_dx - k2 * df_dx * np.log(P/f_values) + k2 * df_dx + k3 * P * dg_dx/(g_values**2)

    constants = {'k1': k1, 'k2': k2, 'k3': k3, 'P': P}
    print(check_ode(x_train_mse, y1_train, y2_train, ode_equation, constants))


    ode_residual_f_func = lambda f_values, g_values, df_dx, dg_dx, x_phys: (
        2 * f_values * df_dx - k1 * df_dx - k2 * df_dx * torch.log(P / f_values) 
        + k2 * df_dx + k3 * P * dg_dx / (g_values**2)
    )

    ode_residual_g_func = lambda dg_dx, x_phys: k4 * x_phys


    # Select only the first n points
    n = 10
    # x_train_mse, y1_train, y2_train = x_train_mse[:n], y1_train[:n], y2_train[:n]
    # Select n equidistant points
    x_train_mse = torch.cat((x_train_mse[::len(x_train_mse)//n], x_train_mse[-1:]), dim=0)
    y1_train = torch.cat((y1_train[::len(y1_train)//n], y1_train[-1:]), dim=0)
    y2_train = torch.cat((y2_train[::len(y2_train)//n], y2_train[-1:]), dim=0)
    

    
    model = DualOutputNN(y01 = y1_train[0], y02 = y2_train[0])
    train(model=model, x_mse=x_train_mse, x_phys=x_phys, y1_train= y1_train,
           y2_train=y2_train,lambda_phys=1.0, lambda_mse=1.0, epochs=5000,
           ode_residual_f_func=ode_residual_f_func, ode_residual_g_func=ode_residual_g_func)
    f_test = torch.tensor(df['VCheck'].values, dtype=torch.float32).reshape(-1, 1)
    g_test = torch.tensor(df['memThickness'].values, dtype=torch.float32).reshape(-1, 1)
    plot_results(model,x_phys, x_train_mse, y1_train, y2_train, 
                 f_test=f_test,g_test=g_test, f_func=None, g_func=None)
    
if __name__ == "__main__":
    main()

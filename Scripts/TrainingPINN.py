# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Training function
def train(model, t_mse, t_phys, x_phys, y1_train, y2_train, ode_residual_f_func, ode_residual_g_func, 
          epochs=1000, lr=0.001, lambda_mse=1.0, lambda_phys=1.0, 
          lambda_phys_f=1.0, lambda_phys_g=1.0, lambda_mse_f=1.0, lambda_mse_g=1.0):
    
    # Convert all training data to float64
    t_mse = t_mse.to(torch.float64)
    t_phys = t_phys.to(torch.float64)
    x_phys = x_phys.to(torch.float64)
    y1_train = y1_train.to(torch.float64)
    y2_train = y2_train.to(torch.float64)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        loss_mse = model.mse_loss(t_mse, y1_train, y2_train, 
                                 lambda_mse_f=lambda_mse_f, 
                                 lambda_mse_g=lambda_mse_g)
        physics_loss = model.physics_loss(t_phys, x_phys, 
                                        ode_residual_f_func, ode_residual_g_func, 
                                        lambda_phys_f=lambda_phys_f, 
                                        lambda_phys_g=lambda_phys_g)
        
        loss = lambda_phys * physics_loss + lambda_mse * loss_mse
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Total Loss = {loss.item():.6f}, "
                  f"MSE Loss = {loss_mse.item():.6f}, "
                  f"Physics Loss = {physics_loss.item():.6f}")
    
    print("Training complete!")

# Function to plot results
def plot_results(model, t_test, t_train_mse, f_train, g_train, f_test, g_test, 
                 f_func=None, g_func=None, figsize=(12,6)):
    model.eval()
    with torch.no_grad():
        # Ensure test data is float64
        t_test = t_test.to(torch.float64)
        y1_pred, y2_pred = model(t_test)
    
    t_test_np = t_test.detach().cpu().numpy()
    t_train_mse_np = t_train_mse.detach().cpu().numpy()
    y1_pred_np = y1_pred.detach().cpu().numpy()
    y2_pred_np = y2_pred.detach().cpu().numpy()
    
    plt.figure(figsize=figsize)

    # Subplot for f(x)
    plt.subplot(1, 2, 1)
    if f_func:
        plt.scatter(t_test_np, f_func(t_test_np), label="True f(x)", alpha=0.5)
        plt.scatter(t_test_np, y1_pred_np, label="Predicted f(x)", marker='x', s=10)
        plt.scatter(t_train_mse_np, f_func(t_train_mse_np), label="Training f(x)", marker='o', s=30, edgecolor='k')
    else:
        plt.scatter(t_test_np, f_test, label="True f(x)", alpha=0.5)
        plt.scatter(t_test_np, y1_pred_np, label="Predicted f(x)", marker='x', s=10)
        plt.scatter(t_train_mse_np, f_train, label="Training f(x)", marker='o', s=30, edgecolor='k')
    plt.legend()
    plt.title("True vs Predicted f(x)")

    # Subplot for g(x)
    plt.subplot(1, 2, 2)
    if g_func:
        plt.scatter(t_test_np, g_func(t_test_np), label="True g(x)", alpha=0.5)
        plt.scatter(t_test_np, y2_pred_np, label="Predicted g(x)", marker='x', s=10)
        plt.scatter(t_train_mse_np, g_func(t_train_mse_np), label="Training g(x)", marker='o', s=30, edgecolor='k')
    else:
        plt.scatter(t_test_np, g_test, label="True g(x)", alpha=0.5)
        plt.scatter(t_test_np, y2_pred_np, label="Predicted g(x)", marker='x', s=10)
        plt.scatter(t_train_mse_np, g_train, label="Training g(x)", marker='o', s=30, edgecolor='k')
    
    plt.legend()
    plt.title("True vs Predicted g(x)")

    plt.tight_layout()
    plt.show()
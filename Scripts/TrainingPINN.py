# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time


# Training function
def train(model, t_mse, t_phys, x_phys, y1_train, y2_train, t_val, y1_val, y2_val, 
          ode_residual_f_func, ode_residual_g_func, 
          epochs=1000, lr=0.001, lambda_mse=1.0, lambda_phys=1.0, 
          lambda_phys_f=1.0, lambda_phys_g=1.0, lambda_mse_f=1.0, lambda_mse_g=1.0, 
          patience=10):
    
    # Convert all training and validation data to float64
    t_mse = t_mse.to(torch.float64)
    t_phys = t_phys.to(torch.float64)
    x_phys = x_phys.to(torch.float64)
    y1_train = y1_train.to(torch.float64)
    y2_train = y2_train.to(torch.float64)
    t_val = t_val.to(torch.float64)
    y1_val = y1_val.to(torch.float64)
    y2_val = y2_val.to(torch.float64)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute training losses
        loss_mse = model.mse_loss(t_mse, y1_train, y2_train, 
                                 lambda_mse_f=lambda_mse_f, 
                                 lambda_mse_g=lambda_mse_g)
        physics_loss = model.physics_loss(t_phys, x_phys, 
                                        ode_residual_f_func, ode_residual_g_func, 
                                        lambda_phys_f=lambda_phys_f, 
                                        lambda_phys_g=lambda_phys_g)
        loss = lambda_phys * physics_loss + lambda_mse * loss_mse
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        # Compute validation loss
        with torch.no_grad():
            val_loss_mse = model.mse_loss(t_val, y1_val, y2_val, 
                                          lambda_mse_f=lambda_mse_f, 
                                          lambda_mse_g=lambda_mse_g)
        
        # Check for improvement
        # FIXME: Use validation loss for early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % max(1, epochs // 50) == 0:
            print(f"Epoch {epoch}: Total Loss = {loss.item():.6f}, "
                  f"MSE Loss = {loss_mse.item():.9f}, "
                  f"Physics Loss = {physics_loss.item():.9f}, "
                  f"Validation Loss = {val_loss_mse.item():.9f}")
        
        # Early stopping
        if patience_counter >= patience:
            break
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Best model loaded.")
    
    best_mse_loss = model.mse_loss(t_mse, y1_train, y2_train,
                                   lambda_mse_f=lambda_mse_f, 
                                   lambda_mse_g=lambda_mse_g)
    best_val_loss_mse = model.mse_loss(t_val, y1_val, y2_val,
                                    lambda_mse_f=lambda_mse_f, 
                                    lambda_mse_g=lambda_mse_g)
    
    print("Training complete!")
    print(f"Early stopping at epoch {epoch}. Best training loss: {best_loss:.9f}. Best MSE loss: {best_mse_loss.item():.9f}. Best Val loss: {best_val_loss_mse.item():.9f}.")
    # Save the model with the best state and the timestamp in the folder ../Models
    model_dir = "../Models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = f"{model_dir}/BestModel_{int(time.time())}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

import os
import torch
import matplotlib.pyplot as plt

# Function to plot results and save the figure using a provided file path.
def plot_results(model, t_test, t_train_mse, f_train, g_train, f_test, g_test, 
                 f_func=None, g_func=None, figsize=(12,6),plot=False, filepath=None):
    model.eval()
    with torch.no_grad():
        # Ensure test data is float64
        t_test = t_test.to(torch.float64)
        y1_pred, y2_pred = model(t_test)
    
    t_test_np = t_test.detach().cpu().numpy()
    t_train_mse_np = t_train_mse.detach().cpu().numpy()
    y1_pred_np = y1_pred.detach().cpu().numpy()
    y2_pred_np = y2_pred.detach().cpu().numpy()
    
    if plot or filepath:
        plt.figure(figsize=figsize)

        # Subplot for Voltage
        plt.subplot(1, 2, 1)
        if f_func:
            plt.scatter(t_test_np, f_func(t_test_np), label="True Voltage", alpha=0.5)
            plt.scatter(t_test_np, y1_pred_np, label="Predicted Voltage", marker='x', s=10)
            plt.scatter(t_train_mse_np, f_func(t_train_mse_np), label="Training Voltage", marker='o', s=30, edgecolor='k')
        else:
            plt.scatter(t_test_np, f_test, label="True Voltage", alpha=0.5)
            plt.scatter(t_test_np, y1_pred_np, label="Predicted Voltage", marker='x', s=10)
            plt.scatter(t_train_mse_np, f_train, label="Training Voltage", marker='o', s=30, edgecolor='k')
        plt.legend()
        plt.title("True vs Predicted Voltage")
        # plt.ylim(1.8, 2.5)  # Set y-axis limits between 1.5 and 3

        # Subplot for Membrane Thickness
        plt.subplot(1, 2, 2)
        if g_func:
            plt.scatter(t_test_np, g_func(t_test_np), label="True Membrane Thickness", alpha=0.5)
            plt.scatter(t_test_np, y2_pred_np, label="Predicted Membrane Thickness", marker='x', s=10)
            plt.scatter(t_train_mse_np, g_func(t_train_mse_np), label="Training Membrane Thickness", marker='o', s=30, edgecolor='k')
        else:
            plt.scatter(t_test_np, g_test, label="True Membrane Thickness", alpha=0.5)
            plt.scatter(t_test_np, y2_pred_np, label="Predicted Membrane Thickness", marker='x', s=10)
            plt.scatter(t_train_mse_np, g_train, label="Training Membrane Thickness", marker='o', s=30, edgecolor='k')
        
        plt.legend()
        plt.title("True vs Predicted Membrane Thickness")
        plt.ylim(0, 2)  # Set y-axis limits between 1.5 and 3
        plt.tight_layout()

        # If a filepath is provided, ensure the directory exists and save the figure.
        if filepath:
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
        
        if plot:
            plt.show()
        else:
            plt.close()

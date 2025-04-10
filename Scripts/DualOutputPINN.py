# Neural Network Model

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


# TODO: Check why is it neccesary to use float64
class DualOutputPINN(nn.Module):
    def __init__(self, t0, y01, y02):
        super(DualOutputPINN, self).__init__()
        # All layers use torch.float64
        self.hidden = nn.Sequential(
            nn.Linear(1, 32, dtype=torch.float64),
            nn.Sigmoid(),
            nn.Linear(32, 32, dtype=torch.float64),
            nn.Sigmoid()
        )
        self.output1 = nn.Linear(32, 1, dtype=torch.float64)  # First output
        self.output2 = nn.Linear(32, 1, dtype=torch.float64)  # Second output
        
        # Ensure initial conditions are float64
        self.t0 = t0.to(torch.float64) if isinstance(t0, torch.Tensor) else torch.tensor(t0, dtype=torch.float64)
        self.y01 = y01.to(torch.float64) if isinstance(y01, torch.Tensor) else torch.tensor(y01, dtype=torch.float64)
        self.y02 = y02.to(torch.float64) if isinstance(y02, torch.Tensor) else torch.tensor(y02, dtype=torch.float64)

    def forward(self, x):
        # Convert input to float64 if needed
        x = x.to(torch.float64) if x.dtype != torch.float64 else x
        
        features = self.hidden(x)
        # Outputs will automatically be float64 due to layer definitions
        y1 = torch.exp(self.output1(features))
        y2 = torch.exp(self.output2(features))
        return y1, y2
    
    def physics_loss(self, t_phys, x_phys, ode_residual_f_func, ode_residual_g_func, lambda_phys_f, lambda_phys_g):
        # Ensure all inputs are float64
        t_phys = t_phys.to(torch.float64).requires_grad_(True)
        x_phys = x_phys.to(torch.float64)
        
        y1, y2 = self.forward(t_phys)
        f_pred, g_pred = y1, y2

        # Create gradient tensors with float64
        df_dx = torch.autograd.grad(f_pred, t_phys, 
                                   grad_outputs=torch.ones_like(f_pred, dtype=torch.float64), 
                                   create_graph=True)[0]
        dg_dx = torch.autograd.grad(g_pred, t_phys, 
                                   grad_outputs=torch.ones_like(g_pred, dtype=torch.float64), 
                                   create_graph=True)[0]
        
        ode_residual_f = ode_residual_f_func(f_pred, g_pred, df_dx, dg_dx, t_phys, x_phys)
        ode_residual_g = ode_residual_g_func(f_pred, g_pred, dg_dx, t_phys)
        
        # Return float64 loss
        return lambda_phys_f*torch.mean(ode_residual_f**2) + lambda_phys_g*torch.mean(ode_residual_g**2)
    
    def mse_loss(self, t_data, f_data, g_data, lambda_mse_f, lambda_mse_g):
        # Ensure all inputs are float64
        t_data = t_data.to(torch.float64)
        f_data = f_data.to(torch.float64)
        g_data = g_data.to(torch.float64)
        
        f_pred, g_pred = self.forward(t_data)
        
        # Use float64 for MSELoss
        loss_f = nn.MSELoss()(f_pred, f_data)
        loss_g = nn.MSELoss()(g_pred, g_data)
        return lambda_mse_f*loss_f + lambda_mse_g*loss_g
    

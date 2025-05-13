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
class PEMElectrolyzerPINN(nn.Module):
    """
    A PyTorch implementation of a Physics-Informed Neural Network (PINN) with dual outputs.
    This class is designed to solve systems of ordinary differential equations (ODEs) 
    with two dependent variables. It includes methods for forward propagation, 
    physics-based loss computation, and data-based loss computation.
    Attributes:
        hidden (nn.Sequential): A sequential model representing the hidden layers of the network.
        output1 (nn.Linear): A linear layer producing the first output of the network.
        output2 (nn.Linear): A linear layer producing the second output of the network.
        t0 (torch.Tensor): Initial time value, converted to torch.float64.
        y01 (torch.Tensor): Initial condition for the first dependent variable, converted to torch.float64.
        y02 (torch.Tensor): Initial condition for the second dependent variable, converted to torch.float64.
    Methods:
        forward(x):
            Performs forward propagation through the network.
            Args:
                x (torch.Tensor): Input tensor of shape (N, 1), where N is the batch size.
            Returns:
                tuple: Two tensors representing the predicted outputs y1 and y2.
        physics_loss(t_phys, x_phys, ode_residual_f_func, ode_residual_g_func, lambda_phys_f, lambda_phys_g):
            Computes the physics-based loss using the residuals of the ODEs.
            Args:
                t_phys (torch.Tensor): Time values for the physics-based loss computation.
                x_phys (torch.Tensor): Additional input data for the physics-based loss computation.
                ode_residual_f_func (callable): Function defining the residual of the first ODE.
                ode_residual_g_func (callable): Function defining the residual of the second ODE.
                lambda_phys_f (float): Weight for the first ODE residual in the loss.
                lambda_phys_g (float): Weight for the second ODE residual in the loss.
            Returns:
                torch.Tensor: The computed physics-based loss.
        mse_loss(t_data, f_data, g_data, lambda_mse_f, lambda_mse_g):
            Computes the mean squared error (MSE) loss using data.
            Args:
                t_data (torch.Tensor): Time values for the data-based loss computation.
                f_data (torch.Tensor): Ground truth data for the first dependent variable.
                g_data (torch.Tensor): Ground truth data for the second dependent variable.
                lambda_mse_f (float): Weight for the MSE loss of the first dependent variable.
                lambda_mse_g (float): Weight for the MSE loss of the second dependent variable.
            Returns:
                torch.Tensor: The computed data-based MSE loss.
        boundary_loss(t_boundary, f_boundary, g_boundary, lambda_boundary_f, lambda_boundary_g):
            Computes the boundary loss error.
            Args:
                t_boundary (torch.Tensor): Boundary time values.
                f_boundary (torch.Tensor): Boundary values for the first dependent variable.
                g_boundary (torch.Tensor): Boundary values for the second dependent variable.
                lambda_boundary_f (float): Weight for the boundary loss of the first dependent variable.
                lambda_boundary_g (float): Weight for the boundary loss of the second dependent variable.
            Returns:
                torch.Tensor: The computed boundary loss.
    """
    def __init__(self, t0, y01, y02):
        super(PEMElectrolyzerPINN, self).__init__()
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

        # Add learnable parameter k for parameter inference
        self.k = nn.Parameter(torch.tensor(100.0, dtype=torch.float64))  # Initialize k to 1.0


    def forward(self, x):
        # Convert input to float64 if needed
        x = x.to(torch.float64) if x.dtype != torch.float64 else x
        
        features = self.hidden(x)
        # Outputs will automatically be float64 due to layer definitions
        # Apply exponential activation to ensure positive outputs
        y1 = torch.exp(self.output1(features))  # First output, enforced to be positive
        # y2 = torch.exp(self.output2(features))  # Second output, enforced to be positive
        y2 = self.output2(features)  # Second output, enforced to be positive

        # y1 = self.y01 + (x - self.t0) * torch.exp(self.output1(features))
        # y2 = self.y02 + (x - self.t0) * self.output2(features)
        return y1, y2
    
    def physics_loss(self, t_phys, x_phys, ode_residual_f_func, ode_residual_g_func, lambda_phys_f, lambda_phys_g):
        # Ensure all inputs are float64 and set t_phys to require gradients for autograd
        t_phys = t_phys.to(torch.float64).requires_grad_(True)
        x_phys = x_phys.to(torch.float64)
        
        # Perform forward pass to get predictions for y1 and y2
        y1, y2 = self.forward(t_phys)
        f_pred, g_pred = y1, y2

        # Compute the gradient of f_pred with respect to t_phys
        df_dx = torch.autograd.grad(f_pred, t_phys, 
                       grad_outputs=torch.ones_like(f_pred, dtype=torch.float64), 
                       create_graph=True)[0]
        
        # Compute the gradient of g_pred with respect to t_phys
        dg_dx = torch.autograd.grad(g_pred, t_phys, 
                       grad_outputs=torch.ones_like(g_pred, dtype=torch.float64), 
                       create_graph=True)[0]
        
        # Compute the residuals of the ODEs using the provided residual functions
        ode_residual_f = ode_residual_f_func(f_pred, g_pred, df_dx, dg_dx, t_phys, x_phys)
        ode_residual_g = ode_residual_g_func(f_pred, g_pred, dg_dx, t_phys, self.k)
        
        # Compute and return the weighted mean squared residuals as the physics-based loss
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

    def boundary_loss(self,lambda_boundary_f, lambda_boundary_g):
        """
        Computes the boundary loss error.
        Args:
            t_boundary (torch.Tensor): Boundary time values.
            f_boundary (torch.Tensor): Boundary values for the first dependent variable.
            g_boundary (torch.Tensor): Boundary values for the second dependent variable.
            lambda_boundary_f (float): Weight for the boundary loss of the first dependent variable.
            lambda_boundary_g (float): Weight for the boundary loss of the second dependent variable.
        Returns:
            torch.Tensor: The computed boundary loss.
        """
        # Use initial conditions y01 and y02 from self
        f_boundary = self.y01
        g_boundary = self.y02
        
        # Perform forward pass to get predictions at initial time t0
        f_pred, g_pred = self.forward(self.t0)
        
        # Compute boundary loss using MSE
        loss_f = nn.MSELoss()(f_pred, f_boundary)
        loss_g = nn.MSELoss()(g_pred, g_boundary)
        return lambda_boundary_f * loss_f + lambda_boundary_g * loss_g


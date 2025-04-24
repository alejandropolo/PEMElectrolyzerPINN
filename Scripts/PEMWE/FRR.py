import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory of 'Scripts' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PEMWE.PEMModel import ELCellStack 
import plotly.graph_objects as go
import plotly.io as pio


"""
FRR.py

This script calculates and visualizes the Fluoride Release Rate (FRR) of a PEM (Proton Exchange Membrane) electrolyzer 
cell stack as a function of temperature and current density. The FRR is computed using the `ELCellStack` class, which 
provides methods for determining the concentration of chemical species and the fluoride release rate.

The script performs the following steps:
1. Defines a range of operating temperatures and current densities.
2. Computes the fluoride release rate (FRR) for each combination of temperature and current density.
3. Stores the computed FRR values in a matrix.
4. Generates a 3D surface plot to visualize the relationship between FRR, temperature, and current density.

Dependencies:
- numpy
- matplotlib
- PEMModel (assumed to contain the `ELCellStack` class)

Usage:
Run this script to generate a 3D plot of FRR vs. temperature and current density for a PEM electrolyzer cell stack.
"""

# Create an instance of the ELCellStack class
cell_stack = ELCellStack()

# Define parameters
t = 0  # Time of operation [seconds] (1 hour for example)
Tk_range = np.linspace(300, 400, 100)  # Temperature range from 300 K to 400 K
I_cell_range = np.linspace(0.0001, 3, 100)  # Current density range from 0.01 to 5 A/cm^2
pres = 30  # Pressure [bars]

# Initialize a matrix to store FRR values
FRR_matrix = np.zeros((len(Tk_range), len(I_cell_range)))

# Loop over the temperature and current density ranges to calculate FRR
for i, Tk in enumerate(Tk_range):
    for j, i_cell in enumerate(I_cell_range):
        # Calculate the concentration of CHO using the conc function
        CH2O2, CHO = cell_stack.conc(Tk, i_cell, pres)
        
        # Calculate FRR using the FRRlm function
        # FRR_value, _ = cell_stack.FRRlm(CHO, t)  # Renamed FRR to FRR_value
        FRR_value, _ = cell_stack.FRRlm_mod(CHO, t)  # Renamed FRR to FRR_value
        
        # Store the FRR value in the matrix (convert to microgram)
        FRR_matrix[i, j] = FRR_value * 1e6

# Create the 3D plot using Plotly


# Create a meshgrid for the X and Y axes
X, Y = np.meshgrid(I_cell_range / cell_stack.A, Tk_range)

# Create the surface plot
fig = go.Figure(data=[go.Surface(z=FRR_matrix, x=X, y=Y, colorscale='Viridis')])

# Update layout for labels and title
fig.update_layout(
    scene=dict(
        xaxis_title='Current Density (A/cm^2)',
        yaxis_title='Temperature (K)',
        zaxis_title='Fluoride Release Rate (ug/cm^2/h)',
    ),
    title='Fluoride Release Rate (FRR) vs. Current Density and Temperature',
)

# Show the plot
fig.show()

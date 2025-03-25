import numpy as np
import matplotlib.pyplot as plt
from PEMModel import ELCellStack  # Assuming ELCellStack is implemented in a module

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
        FRR_value, _ = cell_stack.FRRlm(CHO, t)  # Renamed FRR to FRR_value
        
        # Store the FRR value in the matrix (convert to microgram)
        FRR_matrix[i, j] = FRR_value * 1e6

# Create the 2D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(I_cell_range / cell_stack.A, Tk_range)
ax.plot_surface(X, Y, FRR_matrix, cmap='viridis')

# Labels and title
ax.set_xlabel('Current Density (A/cm^2)')
ax.set_ylabel('Temperature (K)')
ax.set_zlabel('Fluoride Release Rate (ug/cm^2/h)')
ax.set_title('Fluoride Release Rate (FRR) vs. Current Density and Temperature')

# Add color bar
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(FRR_matrix)
plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)

# Adjust view angle
ax.view_init(elev=30, azim=135)  # Adjust angle for better visualization

# Show the plot
plt.show()

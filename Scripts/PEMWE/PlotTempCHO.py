import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory of 'Scripts' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PEMWE.PEMModel import ELCellStack

def plotTempCHO():
    # Create an instance of the ELCellStack class
    cell_stack = ELCellStack()

    # Define parameters
    pres = 30  # Assume pressure in bars (standard pressure)
    i_cell = 1.47 # Assume a constant current (A/cm^2)
    Tk_range = np.linspace(300, 400, 100)  # Temperature range from 300 K to 400 K

    # Initialize an array to store CHO concentrations
    CHO_conc = np.zeros_like(Tk_range)

    # Loop over the temperature range and calculate CHO concentration
    for i, Tk in enumerate(Tk_range):
        _, CHO = cell_stack.conc(Tk, i_cell, pres)
        CHO_conc[i] = CHO

    # Plot CHO concentration vs. temperature
    plt.figure()
    plt.plot(Tk_range, CHO_conc, linewidth=2, color='b')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Hydroxyl Concentration (mol/m^3)')
    plt.title('Hydroxyl Concentration vs. Temperature')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plotTempCHO()
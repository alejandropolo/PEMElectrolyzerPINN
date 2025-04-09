import numpy as np
import matplotlib.pyplot as plt
from PEMModel import ELCellStack

# Define constants and the cell stack object
Tk = 353  # Temperature in Kelvin (standard room temperature)
pres = 10  # Assume pressure in bars (standard pressure)

# Create an instance of the ELCellStack class
cell_stack = ELCellStack()

lm = 1.35e-2  # Membrane thickness [cm]
cell_stack.lm = lm

# Define the current range from 0 to 6 (units of A/cm^2)
I_range = np.linspace(0, 5.9, 1000)   # Current Density in A/cm^2

# Initialize arrays to store voltages, overpotentials, and total potentials
voltages = np.zeros_like(I_range)
activation_overpotentials = np.zeros_like(I_range)
activation_overpotentials_log = np.zeros_like(I_range)
ohmic_voltages = np.zeros_like(I_range)
concentration_voltages = np.zeros_like(I_range)

# Loop over the current range to compute total potential and individual components
for i, i_cell in enumerate(I_range):
    # Calculate individual potentials
    ENernst_potential = cell_stack.ENernst(Tk, pres)
    VAct_potential = cell_stack.VAct(Tk, i_cell)
    VAct_potential_log = cell_stack.VAct_log(Tk, i_cell)
    VOhm_potential = cell_stack.VOhm(Tk, i_cell)
    VConc_potential = cell_stack.VConc(Tk, i_cell)
    
    # Sum the potentials for the total voltage
    total_voltage = ENernst_potential + VAct_potential + VOhm_potential + VConc_potential
    
    # Store the total voltage and individual components
    voltages[i] = total_voltage
    activation_overpotentials[i] = VAct_potential
    activation_overpotentials_log[i] = VAct_potential_log
    ohmic_voltages[i] = VOhm_potential
    concentration_voltages[i] = VConc_potential

# Plot all in the same plot
plt.figure()
plt.plot(I_range, voltages, linewidth=2, label='Total Voltage')
plt.plot(I_range, activation_overpotentials, '--', linewidth=2, label='Activation Overpotential')
# plt.plot(I_range, activation_overpotentials_log, '-.', linewidth=2, label='Activation Overpotential (Log)')
plt.plot(I_range, ohmic_voltages, ':', linewidth=2, label='Ohmic Voltage')
plt.plot(I_range, concentration_voltages, '-.', linewidth=2, label='Concentration Voltage')

# Labels and title
plt.xlabel('Current (A/cm^2)')
plt.ylabel('Voltage (V)')
plt.title('IV Curve and Components')
plt.legend(loc='best')

plt.grid(True)
plt.show()
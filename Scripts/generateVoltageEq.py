import numpy as np
import matplotlib.pyplot as plt
from PEMModel import ELCellStack

# Define constants and the cell stack object
Tk = 353  # Temperature in Kelvin (standard room temperature)
pres = 1  # Assume pressure in bars (standard pressure)

# Create an instance of the ELCellStack class
cell_stack = ELCellStack()

lm = 1.78e-2  # Membrane thickness [cm]
cell_stack.lm = lm

# Define the current range from 0 to 6 (units of A/cm^2)
I_range = np.linspace(0, 5.9, 1000)   # Current Density in A/cm^2

# Initialize arrays to store voltages, overpotentials, and total potentials
voltages = np.zeros_like(I_range)
activation_overpotentials = np.zeros_like(I_range)
activation_overpotentials_log = np.zeros_like(I_range)
ohmic_voltages = np.zeros_like(I_range)

voltage_check = np.zeros_like(I_range)
voltage_check_deg = np.zeros_like(I_range)


# Loop over the current range to compute total potential and individual components
for i, i_cell in enumerate(I_range):

    # Calculate individual potentials
    ENernst_potential = cell_stack.ENernst(Tk, pres)
    # VAct_potential = cell_stack.VAct(Tk, i_cell)
    VAct_potential = cell_stack.Vact_deg(Tk, i_cell,pres,lm,exact=False)
    # VAct_potential_log = cell_stack.VAct_log(Tk, i_cell)
    # VOhm_potential = cell_stack.VOhm(Tk, i_cell)
    VOhm_potential = cell_stack.VOhm_deg(Tk, i_cell,lm)

    # Vcheck = cell_stack.VCell(Tk, i_cell, pres)[0]
    Vcheck_deg = cell_stack.VCell_deg(Tk, i_cell, pres,lm)[0]
    
    # Sum the potentials for the total voltage
    total_voltage = ENernst_potential + VAct_potential + VOhm_potential
    # total_voltage = ENernst_potential + VAct_potential_log + VOhm_potential
    
    # Store the total voltage and individual components
    voltages[i] = total_voltage
    activation_overpotentials[i] = VAct_potential
    # activation_overpotentials_log[i] = VAct_potential_log
    ohmic_voltages[i] = VOhm_potential
    # voltage_check[i] = Vcheck
    voltage_check_deg[i] = Vcheck_deg


# Plot all in the same plot
plt.figure()
plt.plot(I_range, voltages, linewidth=2, label='Total Voltage')
plt.plot(I_range, activation_overpotentials, '--', linewidth=2, label='Activation Overpotential')
# plt.plot(I_range, activation_overpotentials_log, '-.', linewidth=2, label='Activation Overpotential (Log)')
plt.plot(I_range, ohmic_voltages, ':', linewidth=2, label='Ohmic Voltage')
# plt.plot(I_range, voltage_check, '-.', linewidth=2, label='Voltage Check')
plt.plot(I_range, voltage_check_deg, '-.', linewidth=2, label='Voltage Check')

# Labels and title
plt.xlabel('Current (A/cm^2)')
plt.ylabel('Voltage (V)')
plt.title('IV Curve and Components')
plt.legend(loc='best')

plt.grid(True)
plt.show()
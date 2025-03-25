import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from PEMModel import ELCellStack

def generateData():
    # Define an instance of the ELCellStack class
    cell_stack = ELCellStack()

    # Constants
    k = 1e-6  # Thinning rate constant [cm/hour]
    Tk = 353  # Temperature [K]
    pres = 30  # Pressure [bars]
    power = 1000  # Power [W]
    initial_thickness = 1.78e-2  # Initial membrane thickness [cm]

    # Time range from 0 to 10000 hours
    t_range = np.linspace(0, 1e4, 1000)

    # Initialize arrays to store results
    membrane_thickness = np.zeros_like(t_range)
    voltage_data = np.zeros_like(t_range)
    ohmic_data = np.zeros_like(t_range)
    activation_data = np.zeros_like(t_range)
    concentration_data = np.zeros_like(t_range)
    intensity_data = np.zeros_like(t_range)
    power_data = np.zeros_like(t_range)

    # Set initial membrane thickness
    membrane_thickness[0] = initial_thickness

    # Function to solve for intensity
    def intensity_solver(i_cell, Tk, pres, lm, power):
        ENernst_potential = cell_stack.ENernst(Tk, pres)
        VAct_potential = cell_stack.Vact_deg(Tk, i_cell, pres, lm,exact=False)
        VOhm_potential = cell_stack.VOhm_deg(Tk, i_cell, lm)
        VConc_potential = cell_stack.VConc(Tk, i_cell)
        total_voltage = ENernst_potential + VAct_potential + VOhm_potential + VConc_potential
        I = i_cell * cell_stack.A
        return power - (I * total_voltage)  # Solve for I where Power = I * Voltage

    # Loop over the time range
    for i, t in enumerate(t_range):
        if i == 0:
            continue  # Skip the first point (initial condition)
        
        # Calculate membrane thinning
        membrane_thickness[i] = initial_thickness - k * t
        lm = membrane_thickness[i]
        
        if lm <= 0:
            membrane_thickness[i] = 0
            break  # Stop if membrane is completely degraded

        # Solve for intensity using fsolve
        I_cell_initial_guess = 1.0 # A/cm^2
        i_cell_solution = fsolve(intensity_solver, I_cell_initial_guess, args=(Tk, pres, lm, power))[0]
        
        # Compute voltage
        ENernst_potential = cell_stack.ENernst(Tk, pres)
        VAct_potential = cell_stack.Vact_deg(Tk, i_cell_solution, pres, lm,exact=False)
        VOhm_potential = cell_stack.VOhm_deg(Tk, i_cell_solution, lm)
        VConc_potential = cell_stack.VConc(Tk, i_cell_solution)
        total_voltage = ENernst_potential + VAct_potential + VOhm_potential + VConc_potential

        ohmic_data[i] = VOhm_potential
        activation_data[i] = VAct_potential
        concentration_data[i] = VConc_potential

        # Store values
        intensity_data[i] = i_cell_solution
        voltage_data[i] = total_voltage
        power_data[i] = i_cell_solution*cell_stack.A * total_voltage

    # Plot results
    plt.figure()
    plt.plot(t_range, membrane_thickness, linewidth=2, label='Membrane Thickness')
    plt.xlabel('Operating Time (hours)')
    plt.ylabel('Membrane Thickness (cm)')
    plt.title('Membrane Thinning Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t_range, voltage_data, linewidth=2, label='Voltage')
    plt.plot(t_range, activation_data, '--', linewidth=2, label='Activation Overpotential')
    plt.plot(t_range, ohmic_data, ':', linewidth=2, label='Ohmic Voltage')
    plt.plot(t_range, concentration_data, '-.', linewidth=2, label='Concentration Voltage')
    plt.xlabel('Operating Time (hours)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t_range, power_data, linewidth=2, label='Power (Intensity * Voltage)')
    plt.xlabel('Operating Time (hours)')
    plt.ylabel('Power (W)')
    plt.title('Power Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Save data to a CSV file
    data = np.column_stack((t_range, membrane_thickness, voltage_data, intensity_data, power_data))
    np.savetxt('membrane_thinning_voltage_data.csv', data, 
               header='Time (hours),Membrane Thickness (cm),Voltage (V),Current Density (A/cm^2),Power (W)', 
               fmt='%.6f', delimiter=',')

if __name__ == "__main__":
    generateData()

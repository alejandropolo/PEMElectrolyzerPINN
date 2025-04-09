import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from PEMModel import ELCellStack
import os

def generateData(decreaseType = 'linear'):
    # Define an instance of the ELCellStack class
    cell_stack = ELCellStack()

    # Constants
    k = 1e-6  # Thinning rate constant [cm/hour]
    Tk = 353  # Temperature [K]
    pres = 30  # Pressure [bars]
    power = 1000  # Power [W]
    initial_thickness = 1.78e-2  # Initial membrane thickness [cm]

    # Time range from 0 to 10000 hours
    final_time = 1e4
    t_range = np.linspace(0, 1, 1000)

    ## TODO: Delete this line and establish as before
    # final_time = 1
    # t_range = np.linspace(0, 1e4, 1000)

    # Initialize arrays to store results
    membrane_thickness = np.zeros_like(t_range)
    voltage_data = np.zeros_like(t_range)
    voltage_data_check = np.zeros_like(t_range)
    ohmic_data = np.zeros_like(t_range)
    activation_data = np.zeros_like(t_range)
    # concentration_data = np.zeros_like(t_range)
    intensity_data = np.zeros_like(t_range)
    intensity_data_check = np.zeros_like(t_range)
    power_data = np.zeros_like(t_range)
    power_data_check = np.zeros_like(t_range)

    ## Save constants 
    k1 = np.zeros_like(t_range)
    k2 = np.zeros_like(t_range)
    k3 = np.zeros_like(t_range)

    # Set initial membrane thickness
    membrane_thickness[0] = initial_thickness

    # Function to solve for intensity
    def intensity_solver(i_cell, Tk, pres, lm, power):
        ENernst_potential = cell_stack.ENernst(Tk, pres)
        VAct_potential = cell_stack.Vact_deg(Tk, i_cell, pres, lm, exact=False)
        VOhm_potential = cell_stack.VOhm_deg(Tk, i_cell, lm)
        total_voltage = ENernst_potential + VAct_potential + VOhm_potential
        I = i_cell * cell_stack.A
        return power - (I * total_voltage)

    def intensity_solver_check(i_cell, Tk, pres, lm, power):
        total_voltage = cell_stack.VCell_deg(Tk, i_cell, pres, lm)[0]
        I = i_cell * cell_stack.A
        return power - (I * total_voltage)

    # Loop over the time range
    for i, t in enumerate(t_range):
        # if i == 0:
        #     continue  # Skip the first point (initial condition)
        
        # # Calculate membrane thinning
        if decreaseType == 'linear':
            membrane_thickness[i] = initial_thickness - (k*final_time) * t
        elif decreaseType == 'quadratic':
            # Option for quadratic decrease
            membrane_thickness[i] = initial_thickness - (k*final_time) * t**2
        elif decreaseType == 'exponential':
            # Option for exponential decrease (dt_mem/dt = -k*mem)
            membrane_thickness[i] = initial_thickness * np.exp(-(k*final_time)*3e2 * t)
        else:
            raise ValueError("Invalid decrease type. Choose 'linear', 'quadratic', or 'exponential'.")
        lm = membrane_thickness[i]
        
        if lm <= 0:
            membrane_thickness[i] = 0
            break  # Stop if membrane is completely degraded

        # Solve for intensity using fsolve
        I_cell_initial_guess = 1.0 # A/cm^2
        i_cell_solution = fsolve(intensity_solver, I_cell_initial_guess, args=(Tk, pres, lm, power))[0]
        i_cell_solution_check = fsolve(intensity_solver_check, I_cell_initial_guess, args=(Tk, pres, lm, power))[0]
        
        # Compute voltage
        ENernst_potential = cell_stack.ENernst(Tk, pres)
        VAct_potential = cell_stack.Vact_deg(Tk, i_cell_solution, pres, lm,exact=False)
        VOhm_potential = cell_stack.VOhm_deg(Tk, i_cell_solution, lm)
        ## TODO: Modify the VCell_deg function to include the concentration overpotential
        # VConc_potential = cell_stack.VConc(Tk, i_cell_solution)
        total_voltage = ENernst_potential + VAct_potential + VOhm_potential #+ VConc_potential
        total_voltage_check, k1_val, k2_val, k3_val = cell_stack.VCell_deg(Tk, i_cell_solution, pres,lm)
         


        ohmic_data[i] = VOhm_potential
        activation_data[i] = VAct_potential
        # concentration_data[i] = VConc_potential

        # Store values
        intensity_data[i] = i_cell_solution
        intensity_data_check[i] = i_cell_solution_check
        voltage_data[i] = total_voltage
        voltage_data_check[i] = total_voltage_check
        power_data[i] = i_cell_solution*cell_stack.A * total_voltage
        power_data_check[i] = i_cell_solution_check*cell_stack.A * total_voltage_check
        k1[i] = k1_val
        k2[i] = k2_val
        k3[i] = k3_val

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
    plt.plot(t_range, voltage_data_check, '--', linewidth=2, label='Voltage Check')
    plt.plot(t_range, activation_data, '--', linewidth=2, label='Activation Overpotential')
    plt.plot(t_range, ohmic_data, ':', linewidth=2, label='Ohmic Voltage')
    # plt.plot(t_range, concentration_data, '-.', linewidth=2, label='Concentration Voltage')
    plt.xlabel('Operating Time (hours)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    # plt.figure()
    # plt.plot(t_range, power_data, linewidth=2, label='Power (Intensity * Voltage)')
    # plt.xlabel('Operating Time (hours)')
    # plt.ylabel('Power (W)')
    # plt.title('Power Over Time')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # Check if the folder ./Data exists, if not create it
    if not os.path.exists('./Data'):
        os.makedirs('./Data')

    # Save data to a CSV file
    data = np.column_stack((t_range, membrane_thickness, voltage_data, voltage_data_check,
                             intensity_data,intensity_data_check, power_data, power_data_check))
    np.savetxt('./Data/membrane_thinning_voltage_data.csv', data, 
               header='Time (hours),Membrane Thickness (cm),Voltage (V),Voltage Check(V),Current Density (A/cm^2),Current Density Check(A/cm^2),Power (W),Power Check(W)', 
               fmt='%.6f', delimiter=',')
    data = np.column_stack((t_range, k1, k2, k3))
    np.savetxt('./Data/constants.csv', data, 
               header='Time (hours),k1,k2,k3', 
               fmt='%.6f', delimiter=',')
    

if __name__ == "__main__":
    generateData(decreaseType='exponential')

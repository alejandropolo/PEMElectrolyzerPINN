import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from PEMModel import ELCellStack
import os
import csv

def generateData(decreaseType='linear'):
    # Define an instance of the ELCellStack class
    cell_stack = ELCellStack()

    # Constants
    k = 1e-6  # Thinning rate constant [cm/hour]
    Tk = 353  # Temperature [K]
    pres = 30  # Pressure [bars]
    power = 1000  # Power [W]
    initial_thickness = 1.78e-2  # Initial membrane thickness [cm]
    final_time = 8e5  # Total simulation time in hours
    

    # Time range from 0 to 1 (normalized)
    n_steps = 1000
    t_range = np.linspace(0, 1, n_steps)
    final_time_np = np.ones(n_steps) * final_time
    dt = t_range[1] - t_range[0] if len(t_range) > 1 else 0

    # Initialize arrays to store results
    membrane_thickness = np.zeros_like(t_range)
    voltage_data = np.zeros_like(t_range)
    voltage_data_check = np.zeros_like(t_range)
    ohmic_data = np.zeros_like(t_range)
    activation_data = np.zeros_like(t_range)
    intensity_data = np.zeros_like(t_range)
    intensity_data_check = np.zeros_like(t_range)
    power_data = np.zeros_like(t_range)
    power_data_check = np.zeros_like(t_range)
    k1 = np.zeros_like(t_range)
    k2 = np.zeros_like(t_range)
    k3 = np.zeros_like(t_range)

    # Chemical degradation specific arrays
    if decreaseType == 'chemical':
        CHO_data = np.zeros_like(t_range)
        CH2O2_data = np.zeros_like(t_range)
        FRR_data = np.zeros_like(t_range)
        cell_stack.lm = initial_thickness  # Initialize membrane thickness

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

    # Main simulation loop
    for i, t in enumerate(t_range):
        if decreaseType == 'chemical':
            if i == 0:
                current_lm = initial_thickness
                dt_actual = 0.0
            else:
                dt_actual = dt * final_time
                current_lm = cell_stack.lm

            # Solve for current density
            I_cell_initial_guess = 1.0
            i_cell_solution = fsolve(
                intensity_solver, I_cell_initial_guess,
                args=(Tk, pres, current_lm, power)
            )[0]
            
            # Get concentrations and update membrane thickness
            CH2O2, CHO = cell_stack.conc(Tk, i_cell_solution, pres)
            FRR_value, new_lm = cell_stack.FRRlm_mod(CHO, dt_actual)
            
            # Store chemical data
            membrane_thickness[i] = new_lm
            CHO_data[i] = CHO
            CH2O2_data[i] = CH2O2
            FRR_data[i] = FRR_value

            if new_lm <= 0:
                membrane_thickness[i] = 0
                break

        else:  # Existing degradation models
            if decreaseType == 'linear':
                membrane_thickness[i] = initial_thickness - (k*final_time) * t
            elif decreaseType == 'quadratic':
                membrane_thickness[i] = initial_thickness - (k*final_time) * t**2
            elif decreaseType == 'exponential':
                membrane_thickness[i] = initial_thickness * np.exp(-(k*final_time)*3e2 * t)
            else:
                raise ValueError("Invalid decrease type. Choose 'linear', 'quadratic', 'exponential' or 'chemical'.")

            if membrane_thickness[i] <= 0:
                membrane_thickness[i] = 0
                break

            current_lm = membrane_thickness[i]

        # Solve for current density for non-chemical cases
        if decreaseType != 'chemical':
            I_cell_initial_guess = 1.0
            i_cell_solution = fsolve(
                intensity_solver, I_cell_initial_guess,
                args=(Tk, pres, current_lm, power)
            )[0]
            i_cell_solution_check = fsolve(
                intensity_solver_check, I_cell_initial_guess,
                args=(Tk, pres, current_lm, power)
            )[0]

        # Calculate voltage components
        ENernst_potential = cell_stack.ENernst(Tk, pres)
        VAct_potential = cell_stack.Vact_deg(Tk, i_cell_solution, pres, current_lm, exact=False)
        VOhm_potential = cell_stack.VOhm_deg(Tk, i_cell_solution, current_lm)
        total_voltage = ENernst_potential + VAct_potential + VOhm_potential
        total_voltage_check, k1_val, k2_val, k3_val = cell_stack.VCell_deg(Tk, i_cell_solution, pres,current_lm)
        

        # Store results
        ohmic_data[i] = VOhm_potential
        activation_data[i] = VAct_potential
        voltage_data[i] = total_voltage
        intensity_data[i] = i_cell_solution
        power_data[i] = i_cell_solution * cell_stack.A * total_voltage
        # power_data_check[i] = i_cell_solution_check*cell_stack.A * total_voltage_check
        

        # Additional calculations for check methods
        if decreaseType != 'chemical':
            total_voltage_check = cell_stack.VCell_deg(Tk, i_cell_solution, pres, current_lm)[0]
            voltage_data_check[i] = total_voltage_check
            intensity_data_check[i] = i_cell_solution_check
            power_data_check[i] = i_cell_solution_check * cell_stack.A * total_voltage_check

        # Store constants
        _, k1_val, k2_val, k3_val = cell_stack.VCell_deg(Tk, i_cell_solution, pres, current_lm)
        k1[i] = k1_val
        k2[i] = k2_val
        k3[i] = k3_val

    # Create Data folder if needed
    if not os.path.exists('./Data'):
        os.makedirs('./Data')

    # Save main data
    if decreaseType == 'chemical':
        data = np.column_stack((
            t_range,
            membrane_thickness,
            voltage_data,
            intensity_data,
            power_data,
            CHO_data,
            CH2O2_data,
            FRR_data
        ))
        header = ('Time (hours),Membrane Thickness (cm),Voltage (V),'
                  'Current Density (A/cm2),Power (W),CHO (mol/m3),'
                  'CH2O2 (mol/m3),FRR (g/cm2/h)')
    else:
        data = np.column_stack((
            t_range,
            membrane_thickness,
            voltage_data,
            voltage_data_check,
            intensity_data,
            intensity_data_check,
            power_data,
            power_data_check
        ))
        header = ('Time (hours),Membrane Thickness (cm),Voltage (V),Voltage Check (V),'
                  'Current Density (A/cm2),Current Density Check (A/cm2),'
                  'Power (W),Power Check (W)')

    np.savetxt('./Data/membrane_thinning_voltage_data.csv', data, header=header, 
               fmt='%.6g', delimiter=',')

    # Save constants
    constants_data = np.column_stack((t_range * final_time, k1, k2, k3,final_time_np))
    np.savetxt('./Data/constants.csv', constants_data, 
               header='Time (hours),k1,k2,k3,final_time', fmt='%.6g', delimiter=',')

    # Plotting
    plt.figure()
    plt.plot(t_range * final_time, membrane_thickness, linewidth=2)
    plt.xlabel('Operating Time (hours)')
    plt.ylabel('Membrane Thickness (cm)')
    plt.title('Membrane Thinning Over Time')
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(t_range * final_time, voltage_data, label='Voltage')
    plt.plot(t_range * final_time, activation_data, '--', label='Activation')
    plt.plot(t_range * final_time, ohmic_data, ':', label='Ohmic')
    plt.xlabel('Operating Time (hours)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage Components Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    generateData(decreaseType='chemical')  # Change degradation type as needed
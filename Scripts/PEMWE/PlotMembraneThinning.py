import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys

# Add the parent directory of 'Scripts' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PEMWE.PEMModel import ELCellStack

def plot_membrane_thinning():
    """
    Simulates and visualizes the degradation of a membrane in an electrochemical cell stack
    over operating time due to thinning effects. The function calculates the percentage 
    reduction in membrane thickness and saves the data to a CSV file for further analysis.
    The degradation is modeled using the ELCellStack class, which provides methods to 
    calculate concentrations of chemical species and the fluoride release rate (FRR) 
    affecting the membrane thickness.
    Parameters:
        None
    Returns:
        None
    Functionality:
        - Initializes an electrochemical cell stack using the ELCellStack class.
        - Defines operating parameters such as temperature, current density, pressure, 
          and simulation time.
        - Calculates the membrane thickness reduction over time using a differential 
          approach.
        - Writes the time, chemical concentrations, FRR, and membrane thickness data 
          to a CSV file.
        - Plots the percentage of thickness reduction and the membrane thickness over 
          operating time.
    Notes:
        - The function uses two approaches for calculating the fluoride release rate (FRR):
          one based on Sorace's paper and another modified differential approach.
        - The CSV file is saved in the './Data/' directory with the name 
          'membrane_thinning_data.csv'.
        - Ensure that the ELCellStack class and its methods (conc, FRRlm, FRRlm_mod) 
          are properly implemented and imported before using this function.
        - The function generates two plots:
          1. Percentage of thickness reduction over time.
          2. Membrane thickness over time.
    Example Usage:
        plot_membrane_thinning()
    """
    # Create an instance of the ELCellStack class
    cell_stack = ELCellStack()

    # Define parameters
    Tk = 353  # Temperature [K]
    i_cell = 1.47 # Current density [A/cm^2]
    pres = 30  # Pressure [bars]
    final_time = 5e6
    t_range = np.linspace(0, 1, 1000)  # Operating time range from 0 to 10000 hours
    dt = t_range[1] - t_range[0]
    # Initialize an array to store membrane thickness reduction percentage
    thickness_reduction_percentage = np.zeros(len(t_range))

    # Initial membrane thickness (in cm)
    initial_thickness = cell_stack.lm
    membrane_thickness = np.zeros_like(t_range)
    membrane_thickness[0] = initial_thickness

    # Open a CSV file to write the data
    with open('./Data/membrane_thinning_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Time', 'CHO', 'CH2O2', 'FRR_value', 'lm'])
        

        # #Loop over the time range to calculate thickness reduction
        # for i, t in enumerate(t_range):
        #     # Get CHO concentration
        #     CH2O2, CHO = cell_stack.conc(Tk, i_cell, pres)
        #     # Calculate FRR and new membrane thickness
        #     FRR_value, lm = cell_stack.FRRlm(CHO, dt*final_time)
        ########## WITH THE FOLLOWING LINE, THE THICKNESS IS REDUCED AS IN SORACE'S PAPER BUT I SUSPECT IT IS AN ERROR ##########
        #     FRR_value, lm = cell_stack.FRRlm(CHO, t*final_time) 
        #     # Calculate the percentage of thickness reduction
        #     thickness_reduction_percentage[i] = (lm / initial_thickness) * 100
        #     # Write the data to the CSV file
        #     writer.writerow([t, CHO, CH2O2, FRR_value, cell_stack.lm ])
        #     # Update the membrane thickness in the cell_stack object
        #     cell_stack.lm = lm
        #     membrane_thickness[i] = cell_stack.lm

        ##################### NEW -> Differential form #####################
        #Loop over the time range to calculate thickness reduction
        for i, t in enumerate(t_range):
            if i > 0:
                # Get CHO concentration
                CH2O2, CHO = cell_stack.conc(Tk, i_cell, pres)
                # Calculate FRR and TR
                FRR_value, lm = cell_stack.FRRlm_mod(CHO, dt*final_time)
                # Calculate the percentage of thickness reduction
                thickness_reduction_percentage[i] = (cell_stack.lm / initial_thickness) * 100
                membrane_thickness[i] = cell_stack.lm
                # Write the data to the CSV file
                writer.writerow([t, CHO, CH2O2, FRR_value, cell_stack.lm])
            else:
                # Get CHO concentration
                CH2O2, CHO = cell_stack.conc(Tk, i_cell, pres)
                # Calculate FRR and TR
                FRR_value, lm = cell_stack.FRRlm_mod(CHO, 0)
                # Calculate the percentage of thickness reduction
                thickness_reduction_percentage[i] = (cell_stack.lm / initial_thickness) * 100
                membrane_thickness[i] = cell_stack.lm
                # Write the data to the CSV file
                writer.writerow([t, CHO, CH2O2, FRR_value, cell_stack.lm])


    # Plot the membrane degradation
    plt.figure()
    plt.plot(t_range, thickness_reduction_percentage, linewidth=2)


    # Labels and title
    plt.xlabel('Operating Time (hours)')
    plt.ylabel('Percentage of Thickness Reduction (%)')
    plt.title('Membrane Degradation over Operating Time')
    plt.grid(True)

    # Show the plot
    plt.show()

    plt.figure()
    plt.plot(t_range, membrane_thickness, linewidth=2)
    # Labels and title
    plt.xlabel('Operating Time (hours)')
    plt.ylabel('Membrane Thickness (cm)')
    plt.title('Membrane Thickness over Operating Time')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_membrane_thinning()
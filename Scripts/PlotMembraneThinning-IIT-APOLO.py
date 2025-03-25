import numpy as np
import matplotlib.pyplot as plt
import csv
from PEMModel import ELCellStack  # Assuming ELCellStack is implemented in a module

def plot_membrane_thinning():
    # Create an instance of the ELCellStack class
    cell_stack = ELCellStack()

    # Define parameters
    Tk = 353  # Temperature [K]
    i_cell = 1 # Current density [A/cm^2]
    pres = 30  # Pressure [bars]
    final_time = 5e4
    t_range = np.linspace(0, 1, 100)  # Operating time range from 0 to 10000 hours

    # Initialize an array to store membrane thickness reduction percentage
    thickness_reduction_percentage = np.zeros(len(t_range))

    # Initial membrane thickness (in cm)
    initial_thickness = cell_stack.lm

    # Open a CSV file to write the data
    with open('./Data/membrane_thinning_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Time', 'CHO', 'CH2O2', 'FRR_value', 'lm'])

        # Loop over the time range to calculate thickness reduction
        for i, t in enumerate(t_range):
            # Get CHO concentration
            CH2O2, CHO = cell_stack.conc(Tk, i_cell, pres)
            
            # Calculate FRR and new membrane thickness
            FRR_value, lm = cell_stack.FRRlm(CHO, t*final_time)
            
            
            # Calculate the percentage of thickness reduction
            thickness_reduction_percentage[i] = (lm / initial_thickness) * 100

            # Write the data to the CSV file
            writer.writerow([t, CHO, CH2O2, FRR_value, cell_stack.lm ])

            # Update the membrane thickness in the cell_stack object
            cell_stack.lm = lm

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

if __name__ == "__main__":
    plot_membrane_thinning()
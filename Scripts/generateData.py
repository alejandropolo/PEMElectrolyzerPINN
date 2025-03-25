import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PEMModel import ELCellStack

def generateData():
    # Define an instance of the ELCellStack class
    cell_stack = ELCellStack()

    # Constants
    k = 1e-6  # Thinning rate constant [cm/hour]
    Tk = 353  # Temperature [K]
    pres = 30  # Pressure [bars]
    power = 100  # Power [W] (Intensity * Voltage)
    initial_thickness = 1.78e-2  # Initial membrane thickness [cm]

    # Time range from 0 to 10000 hours
    t_range = np.linspace(0, 1e4, 1000)  # 100 points between 0 and 10000 hours

    # Define constant intensity 
    I = 1 * cell_stack.A  # Current density [A/cm^2]

    # Initialize arrays to store membrane thickness, voltage, and power data
    membrane_thickness = np.zeros_like(t_range)
    voltage_data = np.zeros_like(t_range)
    power_data = np.zeros_like(t_range)

    # Set initial membrane thickness
    membrane_thickness[0] = initial_thickness


    # Loop over the time range to calculate membrane thinning and power
    for i, t in enumerate(t_range):
        if i == 0:
            continue  # Skip the first point (initial condition)
        
        # Calculate membrane thinning: lm(t) = lm(0) - k * t
        membrane_thickness[i] = initial_thickness - k * t
        lm = membrane_thickness[i]
        
        # Ensure membrane thickness does not go below zero
        if membrane_thickness[i] < 0:
            membrane_thickness[i] = 0
            break  # Stop the simulation if the membrane is completely thinned
        
        # Compute voltage based on electrochemical equations
        ENernst_potential = cell_stack.ENernst(Tk, pres)
        VAct_potential = cell_stack.Vact_deg(Tk, I, pres, lm)  # Assuming constant current I = power / voltage
        VOhm_potential = cell_stack.VOhm_deg(Tk, I, lm)

        VConc_potential = cell_stack.VConc(Tk, I)
        
        voltage = ENernst_potential + VAct_potential + VOhm_potential + VConc_potential
        voltage_data[i] = voltage
        
        # Calculate power (Intensity * Voltage)
        intensity = power / voltage if voltage > 0 else 0  # Avoid division by zero
        power_data[i] = intensity * voltage

    # Plot membrane thickness over time
    plt.figure()
    plt.plot(t_range, membrane_thickness, linewidth=2, label='Membrane Thickness')
    plt.xlabel('Operating Time (hours)')
    plt.ylabel('Membrane Thickness (cm)')
    plt.title('Membrane Thinning Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot voltage over time
    plt.figure()
    plt.plot(t_range, voltage_data, linewidth=2, label='Voltage')
    plt.xlabel('Operating Time (hours)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot power over time
    plt.figure()
    plt.plot(t_range, power_data, linewidth=2, label='Power (Intensity * Voltage)')
    plt.xlabel('Operating Time (hours)')
    plt.ylabel('Power (W)')
    plt.title('Power Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Save data to a CSV file
    data = np.column_stack((t_range, membrane_thickness, voltage_data, power_data))
    np.savetxt('membrane_thinning_voltage_data.csv', data, header='Time (hours),Membrane Thickness (cm),Voltage (V),Power (W)', fmt='%.6f', delimiter=',')

if __name__ == "__main__":
    generateData()
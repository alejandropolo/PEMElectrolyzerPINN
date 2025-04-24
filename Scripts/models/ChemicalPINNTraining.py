#!/usr/bin/env python3
"""
Dual-Output PINN Model for Chemical Decrease Simulation
=========================================================
This script implements a dual-output Physics-Informed Neural Network (PINN)
to simulate the chemical degradation of a membrane based on experimental data.
For each combination of temperature and pressure, synthetic data is generated,
the PINN is trained, the MSE losses are computed on both training and test data,
and the results are plotted and logged in a CSV file.
"""

### IMPORTS
import sys
import os
import warnings
import logging
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configure logging to print to the command prompt.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Append custom scripts directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PEMElectrolyzerPINN import PEMElectrolyzerPINN
from TrainingPINN import train, plot_results
from data.generateData import generateData

### GLOBAL CONSTANTS & CONFIGURATION
# Physical constants and parameters
MM_H2O = torch.tensor(18.0, dtype=torch.float64)   # Water molar mass [g/mol]
CELL_AREA = torch.tensor(680.0, dtype=torch.float64)  # Cell area [cm^2]
FARADAY = torch.tensor(96485.0, dtype=torch.float64)  # Faraday's constant [C/mol]
RHO_H2O = torch.tensor(997.0, dtype=torch.float64)    # Water density [kg/m^3]
LAMBDA_M = torch.tensor(20.0, dtype=torch.float64)    # Membrane hydration parameter
R_CONSTANT = torch.tensor(8.314, dtype=torch.float64)   # Universal gas constant [J/mol/K]
T_REF = torch.tensor(298.0, dtype=torch.float64)        # Reference temperature [K]

### UTILITY FUNCTIONS

def PsatH2O(Tk: torch.Tensor) -> torch.Tensor:
    """
    Compute the saturated water vapor pressure (in bar).

    Parameters:
        Tk (torch.Tensor): Temperature in Kelvin.

    Returns:
        torch.Tensor: Saturated water vapor pressure in bar.
    """
    Tc = Tk - 273.15  # Convert to Celsius
    return 0.0061 * torch.exp((Tc / (Tc + 238.3)) * 17.2694)


def compute_CH2O2_CHO(Tk: torch.Tensor, i_cell: torch.Tensor, pres: float) -> torch.Tensor:
    """
    Compute the hydroxyl concentration (CHO) from the current density.

    Parameters:
        Tk (torch.Tensor): Temperature in Kelvin (dtype=torch.float64).
        i_cell (torch.Tensor): Current density (dtype=torch.float64).
        pres (float): Pressure (bar).

    Returns:
        torch.Tensor: Hydroxyl concentration [mol/m^3].
    """
    # Input validations
    assert Tk.dtype == torch.float64, "Tk must be of type torch.float64"
    assert i_cell.dtype == torch.float64, "i_cell must be of type torch.float64"
    assert isinstance(pres, (float, int)), "pres must be a float or int"

    # Convert current density to A/cm^2
    i_cell = i_cell / (1e-4)
    eta = 0.695  # Equilibrium overpotential [V]
    pO2 = pres - PsatH2O(Tk)  # Oxygen partial pressure [bar]
    sO2 = 1.62e-6 * torch.exp(603 / Tk) * 1e5  # Oxygen solubility [mol/m^3/bar]
    cO2 = sO2 * pO2  # Oxygen concentration [mol/m^3]

    # Use global constants defined above
    Qc = MM_H2O * i_cell * (CELL_AREA * 1e-4) / (2 * FARADAY * RHO_H2O * 1e3)  # Consumed water flow [m^3/s]
    Qt = (-0.332 * torch.log(i_cell) + 5.59) * Qc  # Transferred water flow [m^3/s]
    vH2O = Qt / (CELL_AREA * 1e-4)  # Water velocity [m/s]
    EW = 1.100  # Nafion equivalent weight [kg/mol]
    rhonaf = 1980  # Nafion dry membrane density [kg/m^3]
    Cmemb = rhonaf / EW  # Membrane concentration [mol/m^3]
    eclc = 1e-5  # Thickness of cathode catalyst layer [m]
    gammac = 150  # Rugosity of cathode [m^2/m^2]
    k1o = 7.068e2  # Kinetic constant [m^7/mol^2/s]
    AH2O2 = 42450  # Activation energy [J/mol]
    alfa = 0.5   # Transfer coefficient of the reaction [-]

    # Concentration of H+
    cH = (1980 + 32.4 * LAMBDA_M) / ((1 + 0.0648 * LAMBDA_M) * EW)

    # Kinetic constant k1
    k1 = k1o * torch.exp(-AH2O2 / (R_CONSTANT * Tk)) * torch.exp(-alfa * FARADAY * eta / (R_CONSTANT * T_REF))
    R1 = k1 * cO2 * cH ** 2  # Reaction rate [mol/m^2/s]
    v1 = gammac * R1 / eclc  # Formation rate [mol/m^3/s]

    # Additional kinetic parameters
    k2 = 1.2e-7      # [s^(-1)]
    k6 = 2.7e4       # [m^3/mol/s]
    k7 = 1.2e7       # [m^3/mol/s]
    k10 = 1e3        # [m^3/mol/s]

    e = k7 * cO2 + k10 * Cmemb - vH2O / eclc
    A2 = -3 * k2 + vH2O / eclc
    B  = e * vH2O / (eclc * k6) - v1 - e * k2 / k6
    C  = -e * v1 / k6
    CH2O2 = (-B + torch.sqrt(B ** 2 - 4 * A2 * C)) / (2 * A2)  # Hydrogen peroxide concentration [mol/m^3]
    CHO = vH2O / (eclc * k6) - k2 / k6 - v1 / (k6 * CH2O2)  # Hydroxyl concentration [mol/m^3]
    return CHO


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load experimental data from a given file path.

    Parameters:
        file_path (str): Path to the CSV file containing the data.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    logging.info(f'Loading data from {file_path}.')
    try:
        df = pd.read_csv(file_path)
        df.columns = ['Time', 'memThickness', 'V', 'I', 'P', 'CHO', 'CH2O2', 'FRR_value']
        df['TR'] = df['FRR_value'] / (0.82 * 2)
        # if 'chemical' in file_path:
        #     df.columns = ['Time', 'memThickness', 'V', 'I', 'P', 'CHO', 'CH2O2', 'FRR_value']
        #     df['TR'] = df['FRR_value'] / (0.82 * 2)
        # elif 'linear' in file_path or 'exponential' in file_path or 'quadratic' in file_path:
        #     df.columns = ['Time', 'memThickness', 'V', 'VCheck', 'I', 'ICheck', 'P', 'PCheck']
        # else:
        #     logging.warning("Unknown file type. Ensure the file has the correct format.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise ValueError("Failed to load data. Check the file path and format.")

    logging.info("Data loaded successfully.")
    return df


def data_preprocessing(df: pd.DataFrame, plot: bool = False):
    """
    Preprocess the data including converting columns to torch tensors and reading constants.

    Parameters:
        df (pd.DataFrame): DataFrame containing the experimental data.
        plot (bool): Flag to plot membrane thickness data.

    Returns:
        tuple: Preprocessed tensors and constants needed for training.
    """
    logging.info("Starting data preprocessing...")
    x_values = torch.tensor(df['Time'].values, dtype=torch.float32)
    f_values = torch.tensor(df['V'].values, dtype=torch.float32)
    g_values = torch.tensor(df['memThickness'].values, dtype=torch.float32)
    i_cell = torch.tensor(df['I'].values, dtype=torch.float32)

    # Load constants from an external file
    constants_df = pd.read_csv('../Data/constants.csv')
    Area_cell = 680

    # Constants tensors
    k1 = torch.tensor(constants_df['k1'], dtype=torch.float32)
    k2 = torch.tensor(constants_df['k2'], dtype=torch.float32)
    k3 = torch.tensor(constants_df['k3'], dtype=torch.float32)
    final_time = torch.tensor(constants_df['final_time'], dtype=torch.float32)[0]
    P = torch.tensor((df['P'] / Area_cell).values, dtype=torch.float32)

    # Calculate mean values for later use
    k1_mean = torch.tensor(constants_df['k1'].mean(), dtype=torch.float32)
    k2_mean = torch.tensor(constants_df['k2'].mean(), dtype=torch.float32)
    k3_mean = torch.tensor(constants_df['k3'].mean(), dtype=torch.float32)
    P_mean = torch.tensor((df['P'] / Area_cell).mean(), dtype=torch.float32)

    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(x_values, g_values, label='Membrane Thickness')
        plt.xlabel('Time (s)')
        plt.ylabel('Membrane Thickness (cm)')
        plt.title('Membrane Thickness')
        plt.legend()
        plt.show()

    logging.info("Data preprocessing complete.")
    return (x_values, f_values, g_values, i_cell, k1_mean, k2_mean,
            k3_mean, P_mean, final_time, g_values, P)


def compute_mse_loss(model: nn.Module, t_data: torch.Tensor,
                     y1_true: torch.Tensor, y2_true: torch.Tensor) -> float:
    """
    Compute the MSE loss for the PINN model predictions versus the true values.
    Combines the MSE for both outputs by taking their average.

    Parameters:
        model (nn.Module): The trained PINN model.
        t_data (torch.Tensor): The input time data.
        y1_true (torch.Tensor): True values for the first output.
        y2_true (torch.Tensor): True values for the second output.

    Returns:
        float: The computed MSE loss.
    """
    model.eval()
    with torch.no_grad():
        y1_pred, y2_pred = model(t_data)
    loss_fn = nn.MSELoss()
    loss1 = loss_fn(y1_pred, y1_true)
    loss2 = loss_fn(y2_pred, y2_true)
    total_loss = (loss1 + loss2) / 2
    return total_loss.item()


def simulate_and_evaluate(temp_c, press, initial_thickness, power, k, 
                          Area_cell=680, decrease_type='chemical', final_time=8e5,
                          n_steps=1000, n_train=20, data_percentage=3, noise=0.0,
                          lambda_phys=1.0, lambda_mse=1.0, epochs=5000, lr=0.01,
                          patience=2000, lambda_boundary=10.0, factor=1e2):
    """
    For a given temperature (in Celsius) and pressure (in bar), generate synthetic data,
    train the PINN, compute the MSE loss on both the training subset and the full test set,
    plot the results, and return the computed losses.

    Parameters:
        temp_c (float): Temperature in Celsius.
        press (float): Pressure in bar.
        initial_thickness (float): Initial membrane thickness [cm].
        power (float): Power [W].
        k (float): Constant parameter for data generation.
        Area_cell (float): Cell area [cm^2]. Default is 680.
        decrease_type (str): Type of decrease ('chemical', etc.). Default is 'chemical'.
        final_time (float): Final simulation time [s]. Default is 8e5.
        n_steps (int): Number of simulation steps for data generation. Default is 1000.
        n_train (int): Number of training points. Default is 20.
        data_percentage (int): Percentage of data to use for training. Default is 3.
        noise (float): Noise level to add to training data. Default is 0.0.
        lambda_phys (float): Weight for physics-informed loss. Default is 1.0.
        lambda_mse (float): Weight for MSE loss. Default is 1.0.
        epochs (int): Number of training epochs. Default is 5000.
        lr (float): Learning rate. Default is 0.01.
        patience (int): Early stopping patience. Default is 2000.
        lambda_boundary (float): Weight for boundary loss. Default is 10.0.
        factor (float): Scale factor for thickness training data. Default is 1e2.

    Returns:
        dict: A dictionary with training and test MSE losses.
    """
    logging.info(f"Simulation start: Temperature = {temp_c}°C, Pressure = {press} bar")
    # ------------------------- Define Simulation Parameters -------------------------

    # Convert temperature from Celsius to Kelvin
    Tk = torch.tensor(temp_c + 273, dtype=torch.float64)
    n_steps = 1000                   # Number of simulation steps for data generation

    # Check if the file already exists and delete it if necessary
    orig_file = os.path.join(os.path.dirname(__file__),'..','..', 'Data', 'membrane_thinning_voltage_data.csv')
    if os.path.exists(orig_file):
        os.remove(orig_file)
        logging.info(f"Existing file {orig_file} deleted.")
    
    # ------------------------- Generate Data -------------------------
    # Call the generateData function.
    # It is assumed that generateData creates a CSV file named "membrane_thinning_voltage_data.csv"
    # in the ../Data directory. To avoid overwriting data between runs, we rename the file.

    # The function is assumed to create a CSV file named "membrane_thinning_voltage_data.csv"
    # in the ../Data directory.
    # and a CSV file names constants.csv
    generateData(decreaseType=decrease_type,
                 k=k,
                 Tk=Tk.item(),
                 pres=press,
                 power=power,
                 initial_thickness=initial_thickness,
                 final_time=final_time,
                 n_steps=n_steps,
                 save_path='../../Data')

    # Read final_time from the constants file
    constants_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'constants.csv'))
    # Assert final_time is equal to predefined final_time
    assert np.isclose(constants_df['final_time'].values[0], final_time), \
        f"Final time in constants.csv ({constants_df['final_time'].values[0]}) does not match the predefined final_time ({final_time})."
    # Convert final_time to tensor
    final_time = torch.tensor(constants_df['final_time'], dtype=torch.float64)[0]
    
    # Rename the generated file to include the combination parameters.
    orig_file = os.path.join('..','..', 'Data', 'membrane_thinning_voltage_data.csv')
    new_filename = f"membrane_thinning_voltage_data_{int(temp_c)}_{int(press)}.csv"
    new_file = os.path.join('..','..', 'Data', new_filename)
    if os.path.exists(new_file):
        os.remove(new_file)
        logging.info(f"Existing file {new_file} deleted.")
    os.rename(orig_file, new_file)
    logging.info(f"Data file renamed to {new_file}")

    # ------------------------- Data Loading -------------------------

    df = load_data(new_file)

    # ------------------------- Set Up Data for Training & Testing -------------------------

    torch.manual_seed(0)
    logging.info("Preparing training data for the PINN model...")

    factor = factor  # Scale factor for thickness training data (if not magnitudes are too different)

    # Define t_phys as full data (for test evaluation and plotting)
    t_phys = torch.tensor(df['Time'].values, dtype=torch.float64).reshape(-1, 1)
    f_values_full = torch.tensor(df['V'].values, dtype=torch.float64).reshape(-1, 1)
    g_values_full = factor * torch.tensor(df['memThickness'].values, dtype=torch.float64).reshape(-1, 1)
    P_area = torch.tensor((df['P'].values / Area_cell), dtype=torch.float64).reshape(-1, 1)


    # ------------------------- Prepare Training Data -------------------------

    t_train = torch.tensor(df['Time'].values, dtype=torch.float64).reshape(-1, 1)
    x_train = f_values_full.clone()
    y1_train = torch.tensor(df['V'].values, dtype=torch.float64).reshape(-1, 1)
    y2_train = factor * torch.tensor(df['memThickness'].values, dtype=torch.float64).reshape(-1, 1)

    # Use a small subset for training (e.g. first 1/8th of the points)
    n = n_train
    half_index = len(t_train) // data_percentage
    indices = torch.linspace(0, half_index - 1, n).long()
    t_train, x_train = t_train[indices], x_train[indices]
    y1_train, y2_train = y1_train[indices], y2_train[indices]

    # Add noise to the training data (excluding the first point)
    logging.info("Adding noise to training data...")
    noise_factor_y1 = noise * (y1_train.max() - y1_train.min())
    noise_factor_y2 = noise * (y2_train.max() - y2_train.min())
    y1_train[1:] += noise_factor_y1 * torch.randn_like(y1_train[1:])
    y2_train[1:] += noise_factor_y2 * torch.randn_like(y2_train[1:])
    logging.info("Training data prepared.")

    # ------------------------- Define ODE Residuals -------------------------
    constants_df = pd.read_csv('../../Data/constants.csv')
    k1_mean = torch.tensor(constants_df['k1'].mean(), dtype=torch.float32)
    k2_mean = torch.tensor(constants_df['k2'].mean(), dtype=torch.float32)
    k3_mean = torch.tensor(constants_df['k3'].mean(), dtype=torch.float32)

    f_func = lambda x, y: k1_mean * torch.ones_like(x) + \
                            k2_mean * torch.log(P_area / x) + \
                            k3_mean * (factor / y) * P_area / x

    ode_residual_f_func = lambda f_pred, g_pred, df_dx, dg_dx, t, x: \
        df_dx - (-k2_mean * df_dx / f_pred + k3_mean * factor * P_area *
                 (-(dg_dx / (f_pred * g_pred**2)) - (df_dx / (g_pred * f_pred**2))))

    k10 = 1e3
    EW = 1.1
    rhonaf = 1980
    Cmemb = rhonaf / EW
    MMF = 18.998403
    A_const = 3.6 * k10 * Cmemb * MMF * 3600 / 1e4
    lam = A_const / 164

    ode_residual_g_func = lambda f_pred, g_pred, dg_dx, t: \
        dg_dx + lam * compute_CH2O2_CHO(Tk, P_area / f_pred, press) * g_pred * final_time

    # ------------------------- Build and Train the Model -------------------------
    logging.info("Initializing and training the PINN model...")
    model = PEMElectrolyzerPINN(t0=t_train[0], y01=y1_train[0], y02=y2_train[0])
    train(model=model,
          t_mse=t_train,
          t_phys=t_phys,
          x_phys=f_values_full,
          y1_train=y1_train,
          y2_train=y2_train,
          t_val=t_phys,
          y1_val=f_values_full,
          y2_val=g_values_full,
          lambda_phys=lambda_phys,
          lambda_mse=lambda_mse,
          epochs=epochs,
          lr=lr,
          patience=patience,
          lambda_phys_f=lambda_phys,
          lambda_phys_g=lambda_phys,
          lambda_mse_f=lambda_mse,
          lambda_mse_g=lambda_mse,
          lambda_boundary=lambda_boundary,
          ode_residual_f_func=ode_residual_f_func,
          ode_residual_g_func=ode_residual_g_func,
          model_dir="../../Models")
    logging.info("Model training complete.")

    # ------------------------- Compute MSE Loss on Training and Test Data -------------------------
    # Loss on training subset
    # train_mse = compute_mse_loss(model, t_train, y1_train, y2_train)
    train_mse = model.mse_loss(t_train, y1_train, y2_train, 
                                          lambda_mse_f=1.0, 
                                          lambda_mse_g=1.0).item()
    # Loss on the full test set (all data points)
    # test_mse = compute_mse_loss(model, t_phys, f_values_full, g_values_full)
    test_mse = model.mse_loss(t_phys, f_values_full, g_values_full,
                                          lambda_mse_f=1.0, 
                                          lambda_mse_g=1.0).item()
    logging.info(f"Train MSE Loss: {train_mse:.8f}")
    logging.info(f"Test MSE Loss: {test_mse:.8f}")

    # ------------------------- Plot the Results -------------------------
    # The plot_results function should handle plotting the predictions vs. the full dataset.
    g_test = g_values_full.detach().numpy()
    f_test = f_values_full.detach().numpy()
    # Generate a filepath with the specific temperature and pressure
    filepath = f"../../Results/Results_{int(temp_c)}_{int(press)}_{int(power)}_{initial_thickness:.2e}_{noise:.2f}_{n}.png"
    plot_results(model, t_phys, t_train, y1_train, y2_train,
                 f_test=f_test, g_test=g_test, figsize=(18, 6), 
                 plot=False, filepath=filepath)

    # Return both training and test losses
    return {
        "Temperature_C": temp_c,
        "Pressure_bar": press,
        "Power_W": power,
        "Initial_Thickness_cm": initial_thickness,
        "Noise": noise,
        "N": n,
        "Train_MSE": train_mse,
        "Test_MSE": test_mse
    }


def main():
    """"
    ChemicalPINNTraining.py
    This script is designed to train and evaluate a Physics-Informed Neural Network (PINN) 
    model for simulating chemical degradation processes in Proton Exchange Membrane Water 
    Electrolysis (PEMWE) systems. The script performs simulations across various combinations 
    of temperature, pressure, power, and initial membrane thickness, computes training and 
    test Mean Squared Error (MSE) losses, and logs the results into a CSV file.
    Functions:
        main():
            Executes the simulation for all combinations of input parameters, evaluates 
            the model's performance, and saves the results.
    Usage:
        Run this script to generate and evaluate PINN models for chemical degradation 
        under different operating conditions. The results are stored in a CSV file 
        for further analysis.
    """
    # Define the temperatures (in Celsius) and pressures (in bar)
    # temperatures = [80]
    # pressures = [30]
    # powers = [100] # Power [W]
    # # FIXME: Review why bigger initial_thickness implies lower voltage
    # initial_thicknesses = [1.00e-2]  # Initial membrane thickness [cm]

    temperatures = [40,60,80]
    pressures = [1,10,30]
    powers = [100,200,500] # Power [W]
    # FIXME: Review why bigger initial_thickness implies lower voltage
    initial_thicknesses = [1.00e-2,1.75e-2]  # Initial membrane thickness [cm]
    k = None # Example constant parameter for data generation
    noise=0.0
    n = 10
    data_percentage = 3
    final_time = 8e5

    # Results file name
    results_csv = "../../Results/results.csv"
    all_results = []

    

    # Generate all combinations of parameters
    all_combinations = list(itertools.product(temperatures, pressures, powers, initial_thicknesses))

    # Select only m combinations if m is specified and less than the total number of combinations
    m = 10  # Example: Select 10 combinations


    if m and m < len(all_combinations):
        selected_combinations = np.random.choice(len(all_combinations), m, replace=False)
        selected_combinations = [all_combinations[i] for i in selected_combinations]
    else:
        selected_combinations = all_combinations

    for temp, press, power, initial_thickness in selected_combinations:
        losses = simulate_and_evaluate(
            temp_c=temp,
            press=press,
            power=power,
            initial_thickness=initial_thickness,
            k=k,
            Area_cell=Area_cell,
            decrease_type=decrease_type,
            final_time=final_time,
            n_steps=n_steps,
            n_train=n,
            data_percentage=data_percentage,
            noise=noise,
            lambda_phys=lambda_phys,
            lambda_mse=lambda_mse,
            epochs=epochs,
            lr=lr,
            patience=patience,
            lambda_boundary=lambda_boundary,
            factor=factor
        )
        all_results.append(losses)
        # Append the row to the CSV file
        df_temp = pd.DataFrame([losses])
        # If file exists, append without header; otherwise, write with header.
        if os.path.exists(results_csv):
            df_temp.to_csv(results_csv, mode='a', index=False, header=False)
        else:
            df_temp.to_csv(results_csv, mode='w', index=False)
        logging.info(f"Results appended for Temp: {temp}°C, Pressure: {press} bar")

    logging.info("All simulations complete. Final results:")
    logging.info(pd.DataFrame(all_results))


if __name__ == '__main__':
    main()

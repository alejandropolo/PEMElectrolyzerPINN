#!/usr/bin/env python3
"""
Dual-Output PINN Model for Chemical Decrease Simulation
=========================================================
This script implements a dual-output Physics-Informed Neural Network (PINN)
to simulate the chemical degradation of a membrane based on experimental data.
It first generates the required CSV file with synthetic data using the 
generateData function, then loads, preprocesses the data, defines the
ODE residuals, builds and trains the PINN model, and finally plots the results.
"""

### IMPORTS
import sys
import warnings
import logging
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
sys.path.append('../Scripts')
from DualOutputPINN import DualOutputPINN
from TrainingPINN import train, plot_results
from generateDataSolverMembraneThinning import generateData


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


### MAIN EXECUTION FLOW

def main():
    """
    Main function to run the data generation, training, and evaluation of the PINN model.
    """
    # ------------------------- Define Simulation Parameters -------------------------
    logging.info("Starting simulation setup...")
    decrease_type = 'chemical'
    # Define simulation parameters
    k = 1.0                           # Example constant parameter for data generation
    Tk = torch.tensor(353.0, dtype=torch.float64)  # Temperature [K]
    pres = 30                         # Pressure [bars]
    power = 1000                      # Power [W]
    initial_thickness = 1.78e-2       # Initial membrane thickness [cm]
    
    # Read final_time from the constants file
    constants_df = pd.read_csv('../Data/constants.csv')
    final_time = torch.tensor(constants_df['final_time'], dtype=torch.float64)[0]
    
    n_steps = 1000                   # Number of simulation steps for data generation

    # ------------------------- Generate Data -------------------------
    logging.info("Generating data file using generateDataS...")
    generateData(decreaseType=decrease_type,
                 k=k,
                 Tk=Tk.item(),
                 pres=pres,
                 power=power,
                 initial_thickness=initial_thickness,
                 final_time=final_time.item(),
                 n_steps=n_steps, save_path='../Data')
    logging.info("Data generation complete.")

    # ------------------------- Data Loading -------------------------
    file_path = f'../Data/membrane_thinning_voltage_data.csv'
    df = load_data(file_path)
    Area_cell = 680  # [cm^2] Cell area

    # ------------------------- Set Up Data for Training -------------------------
    torch.manual_seed(0)
    logging.info("Preparing training data for the PINN model...")
    
    # Scale factor for thickness training data
    factor = 1e2

    t_phys = torch.tensor(df['Time'].values, dtype=torch.float64).reshape(-1, 1)
    f_values = torch.tensor(df['V'].values, dtype=torch.float64).reshape(-1, 1)
    x_phys = f_values.clone()
    P_area = torch.tensor((df['P'].values / Area_cell), dtype=torch.float64).reshape(-1, 1)
    initial_thickness_csv = torch.tensor(df['memThickness'].values[0], dtype=torch.float64)

    # ------------------------- Define ODE Residuals for PINN -------------------------
    k1_mean = torch.tensor(constants_df['k1'].mean(), dtype=torch.float32)
    k2_mean = torch.tensor(constants_df['k2'].mean(), dtype=torch.float32)
    k3_mean = torch.tensor(constants_df['k3'].mean(), dtype=torch.float32)
    
    f_func = lambda x, y: k1_mean * torch.ones_like(x) + \
                            k2_mean * torch.log(P_area / x) + \
                            k3_mean * (factor / y) * P_area / x

    ode_residual_f_func = lambda f_pred, g_pred, df_dx, dg_dx, t, x: \
        df_dx - (-k2_mean * df_dx / f_pred + k3_mean * factor * P_area *
                 (-(dg_dx / (f_pred * g_pred**2)) - (df_dx / (g_pred * f_pred**2))))

    # Chemical model parameters for the membrane thickness ODE residual
    k10 = 1e3
    EW = 1.1
    rhonaf = 1980
    Cmemb = rhonaf / EW
    MMF = 18.998403
    A_const = 3.6 * k10 * Cmemb * MMF * 3600 / 1e4  # Derived constant from kinetics
    lam = A_const / 164  # Effective rate constant

    ode_residual_g_func = lambda f_pred, g_pred, dg_dx, t: \
        dg_dx + lam * compute_CH2O2_CHO(Tk, P_area / f_pred, pres) * g_pred * final_time

    # ------------------------- Prepare Training Data -------------------------
    t_train_mse = torch.tensor(df['Time'].values, dtype=torch.float64).reshape(-1, 1)
    x_train_mse = f_values.clone()
    y1_train = torch.tensor(df['V'].values, dtype=torch.float64).reshape(-1, 1)
    y2_train = factor * torch.tensor(df['memThickness'].values, dtype=torch.float64).reshape(-1, 1)

    n = 1
    half_index = len(t_train_mse) // 8
    indices = torch.linspace(0, half_index - 1, n).long()
    t_train_mse, x_train_mse = t_train_mse[indices], x_train_mse[indices]
    y1_train, y2_train = y1_train[indices], y2_train[indices]
    logging.info("Training data prepared.")

    # ------------------------- Build and Train the Model -------------------------
    logging.info("Initializing and training the PINN model...")
    model = DualOutputPINN(t0=t_train_mse[0], y01=y1_train[0], y02=y2_train[0])
    train(model=model,
          t_mse=t_train_mse,
          t_phys=t_phys,
          x_phys=x_phys,
          y1_train=y1_train,
          y2_train=y2_train,
          lambda_phys=1.0,
          lambda_mse=1.0,
          epochs=5000,
          lr=0.1,
          lambda_phys_f=1.0,
          lambda_phys_g=1.0,
          lambda_mse_f=1.0,
          lambda_mse_g=1,
          ode_residual_f_func=ode_residual_f_func,
          ode_residual_g_func=ode_residual_g_func)
    logging.info("Model training complete.")

    # ------------------------- Model Evaluation & Plotting -------------------------
    logging.info("Evaluating the model and plotting results...")
    g_test = factor * torch.tensor(df['memThickness'].values, dtype=torch.float64).reshape(-1, 1).detach().numpy()
    f_test = torch.tensor(df['V'].values, dtype=torch.float64).reshape(-1, 1).detach().numpy()
    plot_results(model, t_phys, t_train_mse, y1_train, y2_train,
                 f_test=f_test, g_test=g_test,
                 f_func=None, g_func=None, figsize=(18, 6))
    logging.info("Evaluation and plotting complete.")

if __name__ == '__main__':
    main()

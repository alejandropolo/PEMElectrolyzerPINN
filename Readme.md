# Files Description

This repository contains scripts, data, and notebooks for modeling, analyzing, and simulating the degradation processes in Proton Exchange Membrane Water Electrolysis (PEMWE) systems. The project leverages Physics-Informed Neural Networks (PINNs), data-driven approaches, and visualization tools to study membrane degradation and other related phenomena.

---

## SCRIPTS

### PEM Electrolyzer Model

1. **PEMModel.py**: Contains the PEM Electrolyzer Class, which computes all overvoltages and models membrane degradation.
2. **PlotIVCurve.py**: Plots the I/V (Intensity/Voltage) curve to characterize the efficiency of the electrolyzer under specific initial conditions.
3. **PlotTempCHO.py**: Visualizes the Hydroxyl Concentration vs. Temperature to analyze how the HO radical concentration varies with temperature.
4. **PlotMembraneThinning.py**: Plots membrane degradation over operating time, showing the percentage of thickness reduction under predefined conditions.
5. **plotVoltageComponents.py**: Calculates and visualizes the voltage-current (IV) characteristics of a PEMWE cell stack, including total voltage and its individual components (e.g., activation overpotential, ohmic voltage).

---

### Data Generation

1. **generateData.py**: Generates synthetic data for chemical, exponential, quadratic, and linear decay cases.

---

### Modelling

1. **PEMElectrolyzerPINN.py**: Implements a Physics-Informed Neural Network (PINN) for solving systems of ODEs with two dependent variables:  
   - **Voltage (V)**: Represents the electrical potential across the PEM electrolyzer.  
   - **Membrane Thickness (cm)**: Represents the thickness of the membrane, which decreases over time due to degradation.  
   The script includes:  
   - A custom neural network architecture for dual-output predictions.  
   - Methods for computing physics-based losses derived from ODE residuals.  
   - Support for boundary conditions and data-based loss computation.
- Utilities for forward propagation, training, and evaluation of the PINN model.  
   This script is the core of the modeling process, enabling the integration of physical laws and experimental data to predict system behavior.
2. **ChemicalPINNTraining.py**: Trains and evaluates a PINN for simulating chemical degradation processes in PEMWE systems. Simulates various combinations of temperature, pressure, power, and initial membrane thickness, and logs performance metrics for analysis.
3. **TrainingPINN.py**: Trains a PINN for modeling and solving ODEs related to PEMWE. Includes utilities for early stopping, saving the best model, and visualizing results such as voltage and membrane thickness predictions.
4. **MultiPINN.py**: Demonstrates a dual-output neural network trained using a combination of data-driven and physics-informed losses derived from ODE residuals. Includes a complete pipeline for generating synthetic data, defining the architecture, training, and visualizing results.
5. **DualOutputNN.py**: Provides an example of defining and training a dual-output neural network using PyTorch. Includes functions for generating synthetic data, defining the architecture, training, and visualizing results.


## DATA

The `Data` folder contains various datasets used for training and analysis:

1. **membrane_thinning_voltage_data_<temp>_<pressure>.csv**: Synthetic data generated for specific temperature and pressure combinations, including time, membrane thickness, voltage, current, and other parameters.
2. **constants.csv**: Contains physical constants and parameters used during data generation and training.
---

## RESULTS

The `Results` folder stores outputs from simulations, including trained models, performance metrics, and visualizations:

1. **Results_temp_pressure_power_thickness_noise_n.png**  
   Plots of predictions vs. true values for membrane thickness and voltage under specific conditions:  
   - **Voltage (V)**: Represents the predicted and true voltage values over time, which are critical for understanding the efficiency of the PEM electrolyzer.  
   - **Membrane Thickness (cm)**: Represents the predicted and true membrane thickness over time, which is essential for analyzing the degradation process.

2. **results.csv**: Contains simulation results for various combinations of temperature, pressure, power, and initial membrane thickness. Includes columns for:
   - `Temperature_C`: Temperature in Celsius.
   - `Pressure_bar`: Pressure in bar.
   - `Power_W`: Power in watts.
   - `Initial_Thickness_cm`: Initial membrane thickness in centimeters.
   - `Noise`: Noise level added to the training data.
   - `N`: Number of training points.
   - `Train_MSE`: Mean Squared Error (MSE) loss on the training subset.
   - `Test_MSE`: Mean Squared Error (MSE) loss on the full test set.

---

## USAGE

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create and activate the environment using the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate <environment-name>
   ```

3. Run the scripts or notebooks as needed. For example, to train a PINN with the parameters defined in config.yaml:
   ```bash
   cd Scripts/models
   python ChemicalPINNTraining.py 
   ```

4. Explore the results in the `Results` folder.
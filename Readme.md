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

1. **PEMElectrolyzerPINN.py**: Implements a Physics-Informed Neural Network (PINN) for solving systems of ODEs with two dependent variables. Includes methods for forward propagation, physics-based loss computation, and data-based loss computation.
2. **MultiPINN.py**: Demonstrates a dual-output neural network trained using a combination of data-driven and physics-informed losses derived from ODE residuals. Includes a complete pipeline for generating synthetic data, defining the architecture, training, and visualizing results.
3. **DualOutputNN.py**: Provides an example of defining and training a dual-output neural network using PyTorch. Includes functions for generating synthetic data, defining the architecture, training, and visualizing results.
4. **chemicalPINNTraining.py**: Trains and evaluates a PINN for simulating chemical degradation processes in PEMWE systems. Simulates various combinations of temperature, pressure, power, and initial membrane thickness, and logs performance metrics for analysis.
5. **TrainingPINN.py**: Trains a PINN for modeling and solving ODEs related to PEMWE. Includes utilities for early stopping, saving the best model, and visualizing results such as voltage and membrane thickness predictions.

---

## NOTEBOOKS

1. **PINNTraining.ipynb**: Notebook for training a PINN to model and solve ODEs related to PEMWE systems.
2. **analysisRealDataPEM.ipynb**: Preprocesses real data from PEM operating plants, infers membrane thickness using gradient descent, and performs parameter inference using a PINN to estimate the rate of membrane thickness degradation.

---

## DATA

The `Data` folder contains various datasets used for training and analysis:

1. 

---

## RESULTS

The `Results` folder stores outputs from simulations, including trained models, performance metrics, and visualizations.

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

3. Run the scripts or notebooks as needed. For example, to train a PINN:
   ```bash
   python TrainingPINN.py
   ```

4. Explore the results in the `Results` folder.
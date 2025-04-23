# Files Description


## SCRIPTS

### PEM Electrolyzar Model

1. **PEMModel.py**: File containing the PEM Electrolyzer Class with the compute of all the overvoltages and the membrane degradation
   
2. **PlotIVCurve.py**: Enable the plot of the I/V (Intensity/Voltage) curve that characterizes the efficency of the electrolyzer for a given set of initial conditions
   
3. **PlotTempCHO.py**: Enable the plot of the Hydroxyl Concentration vs. Temperature to understand how the concentration of the HO radical varies with temperature
   
4. **PlotMembraneThinning.py**: Enable the plot of the Membrane degradation over operating time to visualize the percentage of thickness reduction as the PEM operates under some predifined conditions
   
5. **plotVoltageComponents.py**: This script calculates and visualizes the voltage-current (IV) characteristics of a Proton Exchange Membrane Water Electrolyzer (PEMWE) cell stack. It computes the total voltage and its individual components (activation overpotential, ohmic voltage, etc.) over a range of current densities. The results are plotted for analysis
   

### Data Generation

1. **generateData**: File that generates the data for the case of chemical, exponential, quadratic and linear decay


### Modelling 

1. **DualOutputPINN.py**: The DualOutputPINN.py file implements a Physics-Informed Neural Network (PINN) designed to solve systems of ordinary differential equations (ODEs) with two dependent variables. This PyTorch-based class includes methods for forward propagation, physics-based loss computation, and data-based loss computation. The network architecture consists of hidden layers with sigmoid activations and two separate output layers, ensuring positive outputs through exponential activation. The class also supports the computation of physics-based losses using ODE residuals and mean squared error (MSE) losses for data-driven training. This file is central to modeling and solving dual-output ODE systems in the project.
   
2. **MultiPINN.py**: The MultiPINN.py file demonstrates the implementation of a neural network with dual outputs trained using a combination of data-driven loss (MSE) and physics-informed loss derived from the residuals of two ordinary differential equations (ODEs). The script includes a complete pipeline for generating synthetic data, defining the neural network architecture, training the model, and visualizing the results. The example ODEs used in this script are df/dx = dg/dx + 2x and dg/dx = 3x^2. This file serves as a practical example of combining physics-based constraints with machine learning to solve dual-output problems.
   
3. **DualOutputNN.py**: The DualOutputNN.py file provides an example of defining and training a dual-output neural network using PyTorch. It demonstrates how to handle multiple outputs in a single model and includes functions for generating synthetic training data, defining the network architecture, training the model, and visualizing the results. The script uses a simple architecture with shared hidden layers and two separate output layers to predict two functions simultaneously. By running this script, users can train the model on generated data and visualize its performance, making it a practical example for dual-output neural network applications.

4. **chemicalPINNTraining.py**: The ChemicalPINNTraining.py file is designed to train and evaluate a Physics-Informed Neural Network (PINN) for simulating chemical degradation processes in Proton Exchange Membrane Water Electrolysis (PEMWE) systems. This script performs simulations across various combinations of temperature, pressure, power, and initial membrane thickness, generating synthetic data, training the PINN model, and computing Mean Squared Error (MSE) losses for both training and test datasets. The results, including performance metrics, are logged into a CSV file for further analysis. This script provides a comprehensive framework for modeling and analyzing chemical degradation under different operating conditions.
     
5. **TrainingPINN.py**: This script is designed to train a Physics-Informed Neural Network (PINN) for modeling and solving systems of ordinary differential equations (ODEs) related to Proton Exchange Membrane Water Electrolysis (PEMWE). It includes functions for training the PINN using a combination of data-driven and physics-informed losses, early stopping, and saving the best model. Additionally, it provides utilities for visualizing the results, such as comparing predicted and true values for voltage and membrane thickness. The script is essential for analyzing the degradation processes in PEMWE systems and generating accurate predictions under various operating conditions.


## NOTEBOOKS

1. **PINNTraining.ipynb**: 
   
2. **analysisRealDataPEM.ipynb**: Notebook containing the preprocessing of the real Data obtained from PEM operating plants, the inference of the membrane thickness through gradient descent and the parameter inference using a PINN to state the rate of the membrane thickness.

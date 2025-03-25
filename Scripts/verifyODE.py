import torch
import pandas as pd

def numerical_derivative(x, y):
    """
    Compute the numerical derivative of y with respect to x using second-order accurate methods.
    """
    dydx = torch.zeros_like(y)
    
    # Central difference for interior points
    dydx[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    
    # Second-order forward difference for the first point
    dydx[0] = (-3*y[0] + 4*y[1] - y[2]) / (x[2] - x[0])
    
    # Second-order backward difference for the last point
    dydx[-1] = (3*y[-1] - 4*y[-2] + y[-3]) / (x[-1] - x[-3])
    
    return dydx

def check_ode(x, f_values, g_values, ode_equation, constants):
    """
    Check if the provided function values satisfy the given ODE.
    
    Parameters:
    x (array-like): The x values.
    f_values (array-like): The function values f(x).
    g_values (array-like): The function values g(x).
    ode_equation (function): A function that takes x, f_values, g_values, df_dx, dg_dx 
                             and returns the right-hand side of the ODE.
    
    Returns:
    bool: True if the ODE is satisfied within the specified tolerance, False otherwise.
    """
    df_dx = numerical_derivative(x, f_values)
    dg_dx = numerical_derivative(x, g_values)
    
    lhs = torch.zeros_like(df_dx)  # Left-hand side of the ODE
    rhs = ode_equation(x, f_values, g_values, df_dx, dg_dx, constants)  # Right-hand side of the ODE
    
    errors = torch.abs(lhs - rhs)
    
    return torch.allclose(lhs, rhs, atol=1e-5)

def main():
    df = pd.read_csv('membrane_thinning_voltage_data.csv')
    # Rename columns to [time, membrane_thickness, voltage, power]
    df.columns = ['Time', 'memThickness', 'V','VCheck','I','ICheck','P','PCheck']
    x_values = torch.tensor(df['Time'].values, dtype=torch.float32)
    f_values = torch.tensor(df['VCheck'].values, dtype=torch.float32)
    g_values = torch.tensor(df['memThickness'].values, dtype=torch.float32)

    # Read constants from file
    constants_df = pd.read_csv('constants.csv')
    k1 = torch.tensor(constants_df['k1'].mean(), dtype=torch.float32)
    k2 = torch.tensor(constants_df['k2'].mean(), dtype=torch.float32)
    k3 = torch.tensor(constants_df['k3'].mean(), dtype=torch.float32)
    Area_cell = 680
    P = torch.tensor((df['P']/Area_cell).mean(), dtype=torch.float32)

    # Generate constants dictionary
    constants = {'k1': k1, 'k2': k2, 'k3': k3, 'P': P}

    # Define the ODE equation as a function
    ############################# LINEAR DECREASE ########################################
    def ode_equation(x, f_values, g_values, df_dx, dg_dx, constants):
        k1 = constants.get('k1', 1)  
        print('k1:', k1.item())
        k2 = constants.get('k2', 1)  
        print('k2:', k2.item())
        k3 = constants.get('k3', 1)
        print('k3:', k3.item())  
        P = constants.get('P', 1)
        print('P:', P.item())
        return 2 * f_values * df_dx - k1 * df_dx - k2 * df_dx * torch.log(P/f_values) + k2 * df_dx + k3 * P * dg_dx/(g_values**2)

    # Check if the points satisfy the ODE
    is_valid = check_ode(x_values, f_values, g_values, ode_equation, constants)
    print("ODE satisfied?", is_valid)

if __name__ == "__main__":
    main()
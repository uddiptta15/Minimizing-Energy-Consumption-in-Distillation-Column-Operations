# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:28:53 2024

@author: Lenovo
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Given values
F = 100  # Feed flow rate (arbitrary units)
zF = 0.5  # Feed composition (50% of the more volatile component)
xD_min = 0.9  # Minimum required distillate purity
xB_max = 0.1  # Maximum allowed bottoms impurity
alpha = 1.5  # Relative volatility of the components
lambda_vaporization = 200  # Latent heat of vaporization (kJ/mol)


# Objective function for total energy consumption
def total_energy(x):
    R, xD, xB = x
    
    # Check if the values are within physical constraints
    if not (0 <= xB <= zF and xB < xD <= 1 and xD >= xD_min and xB <= xB_max):
        return 1e6  # Large penalty if constraints are violated

    # Calculate distillate flow rate D and bottoms flow rate B
    D = F * (zF - xB) / (xD - xB)
    B = F - D

    # Calculate condenser and reboiler duties using energy balances
    Q_C = D * R * lambda_vaporization  # Condenser duty
    Q_R = D * (R + 1) * lambda_vaporization  # Reboiler duty

    # Total energy consumption
    total_energy = Q_C + Q_R
    return total_energy

# Constraints for purity
constraints = [
    {'type': 'ineq', 'fun': lambda x: x[1] - xD_min},  # xD >= xD_min
    {'type': 'ineq', 'fun': lambda x: xB_max - x[2]},  # xB <= xB_max
    {'type': 'ineq', 'fun': lambda x: x[2] - 0},       # xB >= 0
    {'type': 'ineq', 'fun': lambda x: 1 - x[1]},       # xD <= 1
    {'type': 'ineq', 'fun': lambda x: x[1] - x[2]},    # xD > xB
]

# Bounds for the optimization variables
bounds = [(2, 10),  # Reflux ratio R
          (xD_min, 1), # Distillate composition xD
          (0, zF)]     # Bottoms composition xB

# Initial guess
x0 = [2, 0.95, 0.05]

# Run the optimization using SLSQP
result = minimize(total_energy, x0, method='SLSQP', bounds=bounds, constraints=constraints)

# Extracting optimized results
R_opt = result.x[0]
xD_opt = result.x[1]
xB_opt = result.x[2]
E_min = result.fun
success = result.success


# Display the optimized results
print(f"Optimized Reflux Ratio (R): {R_opt}")
print(f"Optimized Distillate Composition (xD): {xD_opt}")
print(f"Optimized Bottoms Composition (xB): {xB_opt}")
print(f"Minimum Energy Consumption: {E_min} kJ")
print(f"Optimization Successful: {success}")

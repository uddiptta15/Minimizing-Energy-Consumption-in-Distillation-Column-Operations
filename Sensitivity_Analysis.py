# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:59:29 2024

@author: Lenovo
"""

import matplotlib.pyplot as plt
import numpy as np

F = 100  # Feed flow rate (arbitrary units)
zF = 0.5  # Feed composition (50% of the more volatile component)
lambda_vaporization = 200  # Latent heat of vaporization (kJ/mol)
xD_min = 0.9  # Minimum required distillate purity
xB_max = 0.1  # Maximum allowed bottoms impurity


# The output values for optimized parameters for each combination of inputs
optimized_reflux_ratio = np.array([1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95, 2.00, 2.05, 2.10, 2.15, 2.20, 2.25, 2.30, 2.35, 2.40, 2.45])  # Example data
optimized_xD = np.array([0.960, 0.962, 0.964, 0.966, 0.968, 0.970, 0.972, 0.974, 0.976, 0.978, 0.980])  # Example data
optimized_xB = np.array([0.060, 0.062, 0.064, 0.066, 0.068, 0.070, 0.072, 0.074, 0.076, 0.078, 0.080])  # Example data


def energy_consumption(x):
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

x0 = [2.01, 0.972,0.071]
Q_r = []
Q_xD = []
Q_xB = []
# Plotting energy consumption vs. feed composition for different latent heats

for i in optimized_reflux_ratio:
    x0[0] = i
    Q_r.append(energy_consumption(x0))
x0 = [2.01, 0.972,0.071]
for i in optimized_xD:
    x0[1] = i
    Q_xD.append(energy_consumption(x0))
x0 = [2.01, 0.972,0.071]
for i in optimized_xB:
    x0[2] = i
    Q_xB.append(energy_consumption(x0))
    
print(Q_r)
# Create a figure and 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Plot the first subplot: Sine and Cosine
axs[0].plot(optimized_reflux_ratio, Q_r, color='b')
axs[0].set_title('Sensitivity Analysisr Energy Consumptlon vs reflux ratio')
axs[0].set_xlabel('reflux_ratio')
axs[0].set_ylabel('Q_r')
axs[0].grid(True)

axs[1].plot(optimized_xD, Q_xD, color='r')
axs[1].set_title('Sensitivity Analysisr Energy Consumptlon vs xD')
axs[1].set_xlabel('xD')
axs[1].set_ylabel('Q_xD')
axs[1].grid(True)

# Plot the second subplot: Tangent and Logarithm
axs[2].plot(optimized_xB, Q_xB, color='g')
axs[2].set_title('Sensitivity Analysisr Energy Consumptlon vs xB')
axs[2].set_xlabel('xB')
axs[2].set_ylabel('Q_xB')
axs[2].grid(True)



# Adjust layout and show plot
plt.tight_layout()
plt.show()
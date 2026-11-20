# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:28:34 2026

@author: 2370021
"""

import numpy as np
import matplotlib.pyplot as plt

f = 250.0              # lens focal length, mm
wavelength_nm = 633.0  # laser wavelength, nm
width_err = 0.25       # slit width uncertainty, mm
#group frequencies (lp/mm)
line_space = np.array([5.0, 6.3, 7.1, 8.0, 9.0, 10, 11, 12.5, 14, 16])
#slit width (mm)
width = np.array([0.66, 0.87, 0.955, 1.185, 1.47, 2.075, 2.11, 2.705, 3.18, 3.525])
#Linear fitting
m, c = np.polyfit(width, line_space, 1)
plt.figure(figsize=(6, 4))
plt.plot(width, m*width + c, "red",
         label=f"Linear fit: y = {m:.2}*x + {c:.2}")
plt.plot(width, line_space, marker='o', color='blue',linestyle='none', label='Measured')
plt.errorbar(width, line_space, xerr=width_err, fmt='o', color='blue',
             capsize=3, label='Uncertainty in x')
plt.title('Line Spacing versus Slit Width')
plt.xlabel('Slit Width (mm)')
plt.ylabel('Line Density (lp/mm)')
plt.tight_layout()
plt.legend(); plt.grid(True); plt.show()
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 20:32:59 2026

@author: 2370021
"""
import numpy as np
from matplotlib import pyplot as plt

LM_size = []
cycle_length = []
image = plt.imread('grid2.png')#import image
image = np.average(image, axis=2)#grayscale
N = 1024#px
n_groups = 5#5 gratings
bounds = np.array([0, 200, 400, 600, 800, 1024])#defining edges of the gratings

##-------------2D Fourier Transform------------------------
FT_image = np.fft.fftshift(np.fft.fft2(image))
##---------------------Cycle length & size---------------------------------
plt.ion()#interactive plot
plt.show(block=False)#open the window first
plt.pause(0.5)#gives time to draw the figures
for g in reversed(range(n_groups)):
    window_col = 0#resets each time
    while True:
        window_entry = int(input(f"Group {g+1}: window_col to try (or -1 to accept last): "))
        if window_entry == -1:#move to next group input
            break
    #copies the unfiltered FT each time
        window_col = window_entry
        FT = FT_image.copy()
        FT[:, :window_col] = 0
        FT[:, N - window_col:] = 0
        out = np.abs(np.fft.ifft2(np.fft.ifftshift(FT)))**2
        plt.figure(); plt.imshow(out[bounds[g]:bounds[g+1]], cmap='gray', aspect='auto')
        plt.title(f"Group {g+1}, window_col={window_col}") 
    LM_size.append(N - 2 * window_col)
    band = image[bounds[g]:bounds[g + 1], :].mean(axis=0)
    spec = np.abs(np.fft.rfft(band - band.mean()))
    k = np.argmax(spec[1:]) + 1
    cycle_length.append(N / k)
    print(f"Group {g+1}: peak at kx={k} cycles/image -> cycle length {N/k:.2f} px")

##-------------------Plotting--------------------------------
print("cycle length:", cycle_length)
print("unfiltered region size: ", LM_size)
c = np.array(cycle_length); s = np.array(LM_size)
theory = np.linspace(c.min(), c.max(), 200)
plt.figure()
plt.ioff()
plt.plot(c, s, 'o', label='measured')#measured curve
plt.plot(theory, 2048 / theory, '--', label='2×1024 / cycle length')#theoretical curve
plt.xlabel('Cycle length (px)'); plt.ylabel('Unfiltered region size (columns)')
plt.legend(); plt.grid(True); plt.show()

cycle_length = np.array(cycle_length)
nu = 1 / cycle_length#spatial freq

m, c = np.polyfit(nu, LM_size, 1)
print(f"slope = {m:.0f} (theory 2048), intercept = {c:.1f} columns")
plt.figure()
x = np.linspace(0, nu.max() * 1.05, 100)
plt.plot(nu, LM_size, 'o', label='Measured')
plt.plot(x, m * x + c, 'r-', label=f'Fit: {m:.0f}x + {c:.0f}')
plt.plot(x, 2048 * x, '--', label='Theory: 2048x')
plt.xlabel('Spatial frequency (cycles/px)')
plt.ylabel('Unfiltered region size (columns)')
plt.legend(); plt.grid(True); plt.show()
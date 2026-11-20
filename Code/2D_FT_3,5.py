# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:47:11 2026

@author: 2370021
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib; print(matplotlib.get_backend())

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
for g in range(n_groups):
    window_col = 0
    while True:
        window_entry = int(input(f"Group {g+1}: window_col to try (or -1 to accept last): "))
        if window_entry == -1:
            break
    #copies the unfiltered FT each time
    window_col = window_entry
    FT = FT_image.copy()
    FT[:, :window_col] = 0
    FT[:, N - window_col:] = 0
    out = np.abs(np.fft.ifft2(np.fft.ifftshift(FT)))**2
    plt.figure(); plt.imshow(out[bounds[g]:bounds[g+1]], cmap='gray', aspect='auto')
    plt.title(f"Group {g+1}, window_col={window_col}")
    plt.show()   
    LM_size.append(N - 2 * window_col)
    band = image[bounds[g]:bounds[g + 1], :].mean(axis=0)
    spec = np.abs(np.fft.rfft(band - band.mean()))
    k = np.argmax(spec[1:]) + 1
    cycle_length.append(N / k)
    print(f"Group {g+1}: peak at kx={k} cycles/image -> cycle length {N/k:.2f} px")

##-------------------Plotting--------------------------------
print("cycle_length:", cycle_length)
print("region_size: ", LM_size)
plt.plot(cycle_length, LM_size, 'o-')
plt.xlabel('Cycle length (px)'); plt.ylabel('Unfiltered region size (columns)')
plt.show()
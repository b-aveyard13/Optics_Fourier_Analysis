# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:37:54 2026

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

##---Removing spatial freqs in HORIZONTAL---------------
# window_row = 400#default 50-600
# FT_image[:window_row-1,:] = 0#slicing (removing) from start of FT_image
# FT_image[-window_row-1:,:] = 0#slicing from end of FT_image
#-------------VERTICAL----------------------------------
window_col = 490#range approx. 50-600
FT_image[:,:window_col-1] = 0#slicing (removing) from start of FT_image
FT_image[:,-window_col-1:] = 0#slicing from end of FT_image

#--------------------------------------------------------
plt.figure()
plt.imshow(np.log10(1 + np.abs(FT_image)**2))#The "1 +..." bit gets rid of the dividing by zero error
#plt.show()

##----------------------Inverse FT--------------------------
ift_image = np.fft.ifft2(np.fft.ifftshift(FT_image))#Inverse FT
ift_int = np.abs(ift_image)**2#intensity
vmin,vmax = ift_int.min(),ift_int.max()#colour scale
#----------------------Grating subplots-----------------------
fig,axs = plt.subplots(n_groups,1,figsize =(20,10),sharex=True)
for i in range(n_groups):
    band = ift_int[bounds[i]:bounds[i+1],:]
    axs[i].imshow(band,cmap='gray',vmin=vmin,vmax=vmax, aspect='auto',
                   extent=[0,N,bounds[i],bounds[i+1]])
    axs[i].set_ylabel(f"Group {i+1}\nrows {bounds[i]-bounds[i+1]}")
axs[-1].set_xlabel("Column(px)")
fig.suptitle(f"Inverse FT Groupings, window_col={window_col}")
plt.tight_layout()
plt.show()        
##---------------------3x1 Subplots-----------------------
fig, axes = plt.subplots(1,3, figsize=(10,10))
axes[0].imshow(image)
axes[0].set_title("Original")

axes[1].imshow(np.log(1 + np.abs(FT_image)))
axes[1].set_title("Log Magnitude of FT")

axes[2].imshow(np.abs(ift_image)**2)
axes[2].set_title("Inverse Fourier Transform (intensity)")
plt.show()

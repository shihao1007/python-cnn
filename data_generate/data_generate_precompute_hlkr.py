# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:27:17 2019

this program generates the training data set for inversion Mie problem using CNN
the hlkr and plcos-theta is precomputed
the evaluated plane is at the origin by back propagation of the plane outside of the sphere
all images are padded so they can be applied with different band pass filters later on

Editor:
    Shihao Ran
    STIM Laboratory
"""

# numpy for most of the data saving and cumputation
import numpy as np
# matplotlib for ploting the images
from matplotlib import pyplot as plt
# pyquaternion for ratating the vectors
from pyquaternion import Quaternion
# scipy for input/output files and special computing
import scipy as sp
from scipy import special
# math for calculations
import math
# import animation for plot animations
from matplotlib import animation as animation
# import sys for printing progress
import sys
# import random for MC sampling
import random

def coeff_b(l, k, n, a):
    jka = sp.special.spherical_jn(l, k * a)
    jka_p = sp.special.spherical_jn(l, k * a, derivative=True)
    jkna = sp.special.spherical_jn(l, k * n * a)
    jkna_p = sp.special.spherical_jn(l, k * n * a, derivative=True)

    yka = sp.special.spherical_yn(l, k * a)
    yka_p = sp.special.spherical_yn(l, k * a, derivative=True)

    hka = jka + yka * 1j
    hka_p = jka_p + yka_p * 1j

    bi = jka * jkna_p * n
    ci = jkna * jka_p
    di = jkna * hka_p
    ei = hka * jkna_p * n

    return (bi - ci) / (di - ei)

#compute the coordinates grid in Fourier domain for the calculation of
#corresponding phase shift value at each pixel
#return the frequency components at z axis in Fourier domain
def cal_kz(fov, simRes, n):
    #the coordinates in Fourier domain is constructed from the coordinates in
    #spatial domain, specifically,
    #1. Get the pixel size in spatial domain, P_size = FOV / Image_Size
    #2. Fourier domain size, F_size = 1 / P_size
    #3. Make a grid with [-F_size / 2, F_size / 2, same resolution]
    #4. Pixel size in Fourier domain will be 1 / Image_size
    
    #make grid in Fourier domain
    x = np.linspace(-simRes/(fov * 2), simRes/(fov * 2), simRes)
    xx, yy = np.meshgrid(x, x)
    
    #allocate the frequency components in x and y axis
    k_xy = np.zeros((simRes, simRes, 2))
    k_xy[..., 0], k_xy[..., 1] = xx, yy
    
    #compute the distance of x, y components in Fourier domain
    k_para_square = k_xy[...,0]**2 + k_xy[...,1]**2
    
    #initialize a z-axis frequency components
    k_z = np.zeros(xx.shape)
    
    #compute kz at each pixel
    for i in range(len(k_para_square)):
        for j in range(len(k_para_square)):
            if k_para_square[i, j] <= np.abs(n) ** 2:
                k_z[i, j] = np.sqrt(np.abs(n) ** 2 - k_para_square[i, j])
    
    #return it
    return k_z


#propogate the field with the specified frequency components and distance
    # Et: the field in spatial domain to be propagated
    # k_z: frequency component in z axis
    # l: distance to propagate
def propagate(Et, k_z, l):
    
    #compute the phase mask for shifting each pixel of the field
    phaseMask = np.exp(1j * k_z * 2 * np.pi * l)
    
    #Fourier transform of the field and do fft-shift to the Fourier image
    #so that the center of the Fourier transform is at the origin
    E_orig = Et
    fE_orig = np.fft.fft2(E_orig)
    fE_shift = np.fft.fftshift(fE_orig)
    
    #apply phase shift to the field in Fourier domain
    fE_propagated = fE_shift * phaseMask
    
    #inverse shift the image in Fourier domain
    #then apply inverse Fourier transform the get the spatial image
    fE_inversae_shift = np.fft.ifftshift(fE_propagated)
    E_prop = np.fft.ifft2(fE_inversae_shift)
    
    #return the propagated field
    return E_prop

# specify parameters for the forward model
# propagation direction vector k
k = [0, 0, -1]
# position of the sphere
ps = [0, 0, 0]
# resolution of the cropped image
res = 128
# in and out numerical aperture of the bandpass optics
NA_in = 0
NA_out = 0
# wave length
lambDa = 2 * np.pi
# padding size
padding = 2
# field of view
fov = 30

# set ranges of the features of the data set
# refractive index
n_r_min = 1.1
n_r_max = 2.0
# attenuation coefficient
n_i_min = 0.01
n_i_max = 0.05
# sphere radius
a_min = 5
a_max = 10

# the z position of the visiulization plane
z_max = a_max
# the maximal order
l_max = math.ceil(2*np.pi * a_max / lambDa + 4 * (2 * np.pi * a_max / lambDa) ** (1/3) + 2)
l = np.arange(0, l_max+1, 1)

# set the size of the features
nb_nr = 20
nb_ni = 20
nb_a = 20

# initialize features
nr = np.linspace(n_r_min, n_r_max, nb_nr)
ni = np.linspace(n_i_min, n_i_max, nb_ni)
a = np.linspace(a_min, a_max, nb_a)

# allocate space for data set
sphere_data = np.zeros((3, nb_nr, nb_ni, nb_a))
B_data_real = np.zeros((22, nb_nr, nb_ni, nb_a))
B_data_imag = np.zeros((22, nb_nr, nb_ni, nb_a))

# construct the evaluate plane
# simulation resolution
# in order to do fft and ifft, expand the image use padding
simRes = res*(2*padding + 1)
# initialize a plane to evaluate the field
# halfgrid is the size of a half grid
halfgrid = np.ceil(fov/2)*(2*padding +1)
# range of x, y
gx = np.linspace(-halfgrid, +halfgrid-1, simRes)
gy = gx
[x, y] = np.meshgrid(gx, gy)     
# make it a plane at z = 0 on the Z axis
z = np.zeros((simRes, simRes,)) + z_max

# initialize r vectors in the space
rVecs = np.zeros((simRes, simRes, 3))
# make x, y, z components
rVecs[:,:,0] = x
rVecs[:,:,1] = y
rVecs[:,:,2] = z
# compute the rvector relative to the sphere
rVecs_ps = rVecs - ps

# calculate the distance matrix
rMag = np.sqrt(np.sum(rVecs_ps ** 2, 2))
kMag = 2 * np.pi / lambDa

rNorm = rVecs_ps / rMag[...,None]

# preconpute scatter matrix = hlkr * plcos_theta

kr = kMag * rMag
cos_theta = np.dot(rNorm, k)

# calculate hlkr and plcos
scatter_matrix = np.zeros((simRes, simRes, l_max+1), dtype = np.complex128)
for i in l:
    
    alpha = (2 * i + 1) * (1j ** i)
    
    jkr = sp.special.spherical_jn(i, kr)
    ykr = sp.special.spherical_yn(i, kr)
    hlkr = jkr + ykr * 1j
    
    plcos = sp.special.eval_legendre(i, cos_theta)
    
    hlkr_plcos = hlkr * plcos * alpha 
    
    scatter_matrix[:, :, i] = hlkr_plcos
    
# pre compute Ef, incident field at z-max
Ef = np.ones((simRes, simRes), dtype = np.complex128)
Ef *= np.exp(z_max * 1j)

# initialize progress indicator
cnt = 0

# the data set is too big to fit it as a whole in the memory
# therefore split them into sub sets for different sphere sizes
for h in range(nb_a):
    # for each sphere size
    im_data = np.zeros((simRes, simRes, 2, nb_nr, nb_ni))
    # generate file name
    im_dir = r'D:\irimages\irholography\CNN\data_v8_padded\im_data' + '%3.3d'% (h)
    
    for i in range(nb_nr):
        for j in range(nb_ni):
            # for each specific sphere
            n0 = nr[i] + ni[j] * 1j
            a0 = a[h]

            # calculate B vector
            B = coeff_b(l, kMag, n0, a0)
            
            Bhlkr = scatter_matrix * B
            
            Es = np.sum(scatter_matrix * B, axis = 2)
            Et = Es + Ef
            
            k_z = cal_kz(fov, simRes, n0)
            Eprop = propagate(Et, k_z, -z_max)
            
            # save real and imaginary part of the field
            im_data[:, :, 0, i, j] = np.real(Eprop)
            im_data[:, :, 1, i, j] = np.imag(Eprop)
            
            # save label
            sphere_data[:, i, j, h] = [nr[i], ni[j], a[h]]
            
            #save B vector
            B_data_real[:np.size(B), i, j, h] = np.real(B)
            B_data_imag[:np.size(B), i, j, h] = np.imag(B)
            
            # print progress
            cnt += 1
            sys.stdout.write('\r' + str(cnt / (nb_a*nb_ni*nb_nr) * 100)  + ' %')
            sys.stdout.flush() # important

    # save the data
    np.save(im_dir, im_data)
    
np.save(r'D:\irimages\irholography\CNN\data_v8_padded\lable_data', sphere_data)
np.save(r'D:\irimages\irholography\CNN\data_v8_padded\B_data_real', B_data_real)
np.save(r'D:\irimages\irholography\CNN\data_v8_padded\B_data_imag', B_data_imag)
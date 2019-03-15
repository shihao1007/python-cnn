# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:05:15 2019

this program generates the training data set for inversion Mie problem using CNN
the scattering matrix hlkr_asymptotic * plcos * alpha is precomputed
the evaluated plane is at the origin by doing inverse Fourier Transform of the
Far Field simulation of the sphere
all images are padded so they can be applied with different band pass filters later on

Editor:
    Shihao Ran
    STIM Laboratory
"""

import numpy as np
import scipy as sp
import scipy.special
import math
import matplotlib.pyplot as plt
import sys

#%%
# Calculate the sphere scattering coefficients
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

    # return ai * (bi - ci) / (di - ei)
    return (bi - ci) / (di - ei)

#%%
class planewave():
    #implement all features of a plane wave
    #   k, E, frequency (or wavelength in a vacuum)--l
    #   try to enforce that E and k have to be orthogonal
    
    #initialization function that sets these parameters when a plane wave is created
    def __init__ (self, k, E):
        
        #self.phi = phi
        self.k = k/np.linalg.norm(k)                      
        self.E = E
        
        #force E and k to be orthogonal
        if ( np.linalg.norm(k) > 1e-15 and np.linalg.norm(E) >1e-15):
            s = np.cross(k, E)              #compute an orthogonal side vector
            s = s / np.linalg.norm(s)       #normalize it
            Edir = np.cross(s, k)              #compute new E vector which is orthogonal
            self.k = k
            self.E = Edir / np.linalg.norm(Edir) * np.linalg.norm(E)
    
    def __str__(self):
        return str(self.k) + "\n" + str(self.E)     #for verify field vectors use print command

    #function that renders the plane wave given a set of coordinates
    def evaluate(self, X, Y, Z):
        k_dot_r = self.k[0] * X + self.k[1] * Y + self.k[2] * Z     #phase term k*r
        ex = np.exp(1j * k_dot_r)       #E field equation  = E0 * exp (i * (k * r)) here we simply set amplitude as 1
        Ef = self.E.reshape((3, 1, 1)) * ex
        return Ef


#%%
def get_order(a, lambDa):
    #calculate the order of the integration based on size of the sphere and the
    #wavelength
    # a: radius of the sphere
    # lambDa: wavelength of the incident field
    
    l_max = math.ceil(2*np.pi * a / lambDa + 4 * (2 * np.pi * a / lambDa) ** (1/3) + 2)
    l = np.arange(0, l_max+1, 1)
    
    return l

#%%
# precalculate the scattering matrix
def cal_scatter_matrix_Ei(lambDa, k_dir, E, res, fov, l, padding, working_dis):
    
    # calculate the prefix alpha term
    alpha = (2*l + 1) * 1j ** l
    
    # construct the evaluate plane
    # simulation resolution
    # in order to do fft and ifft, expand the image use padding
    simRes = res*(2*padding + 1)
    simFov = fov*(2*padding + 1)
    # halfgrid is the size of a half grid
    halfgrid = np.ceil(simFov/2)
    # range of x, y
    gx = np.linspace(-halfgrid, +halfgrid, simRes)
    gy = gx
    [x, y] = np.meshgrid(gx, gy)     
    # make it a plane at z = 0 (plus the working distance) on the Z axis
    z = np.zeros((simRes, simRes)) + working_dis
    
    # initialize r vectors in the space
    rVecs = np.zeros((simRes, simRes, 3))
    # make x, y, z components
    rVecs[:,:,0] = x
    rVecs[:,:,1] = y
    rVecs[:,:,2] = z
    # compute the rvector relative to the sphere
    # put the sphere at the origin by defualt
    rVecs_ps = rVecs - [0, 0, 0]
    
    # calculate the distance matrix
    rMag = np.sqrt(np.sum(rVecs_ps ** 2, 2))
    kMag = 2 * np.pi / lambDa
    # calculate k dot r
    kr = kMag * rMag
    
    # calculate the asymptotic form of hankel funtions
    hlkr_asym = np.zeros((kr.shape[0], kr.shape[1], l.shape[0]), dtype = np.complex128)
    for i in l:
        hlkr_asym[..., i] = np.exp(1j*(kr-i*math.pi/2))/(1j * kr)
    
    # calculate the legendre polynomial
    # get the frequency components
    fx = np.fft.fftfreq(simRes, simFov/simRes)
    fy = fx
    
    # create a meshgrid in the Fourier Domain
    [kx, ky] = np.meshgrid(fx, fy)
    # calculate the sum of kx ky components so we can calculate 
    # cos_theta in the Fourier Domain later
    kxky = kx ** 2 + ky ** 2
    # create a mask where the sum of kx^2 + ky^2 is 
    # bigger than 1 (where kz is not defined)
    mask = kxky > 1
    # mask out the sum
    kxky[mask] = 0
    # calculate cos theta in Fourier domain
    cos_theta = np.sqrt(1 - kxky)
    cos_theta[mask] = 0
    # calculate the Legendre Polynomial term
    pl_cos_theta = sp.special.eval_legendre(l, cos_theta[..., None])
    # mask out the light that is propagating outside of the objective
    pl_cos_theta[mask] = 0
    
    # calculate the matrix
    scatter_matrix = hlkr_asym * pl_cos_theta * alpha
    
    # pre compute Ef, incident field at z-max
    E_obj = planewave(k_dir, E)
    Ep = E_obj.evaluate(x, y, np.zeros((simRes, simRes)))
    Ei = Ep[0,...]
    
    return scatter_matrix, Ei


#%%
def cal_feature_space(a_min, a_max,
                      nr_min, nr_max,
                      ni_min, ni_max, num):
# set the range of the features:
    # a_max: the maximal of the radius of the spheres
    # a_min: the minimal of the radius of the spheres
    # nr_max: the maximal of the refractive indices
    # nr_min: the minimal of the refractive indices
    # ni_max: the maximal of the attenuation coefficients
    # ni_min: the minimal of the attenuation coefficients
    # num: dimention of each feature
    a = np.linspace(a_min, a_max, num)
    nr = np.linspace(nr_min, nr_max, num)
    ni = np.linspace(ni_min, ni_max, num)
    
    return a, nr, ni

#%%
# set the size and resolution of both planes
fov = 16                    # field of view
res = 128                   # resolution

lambDa = 1                  # wavelength

k = 2 * math.pi / lambDa    # wavenumber
padding = 2                 # padding
working_dis = 1000          # working distance

# scale factor of the intensity
scale_factor = working_dis * 2 * math.pi * res/fov           

# simulation resolution
# in order to do fft and ifft, expand the image use padding
simRes = res*(2*padding + 1)
simFov = fov*(2*padding + 1)
ps = [0, 0, 0]              # position of the sphere
k_dir = [0, 0, -1]          # propagation direction of the plane wave
E = [1, 0, 0]               # electric field vector

# define feature space

num = 20
num_samples = num ** 3

a_min = 1.0
a_max = 2.0

nr_min = 1.1
nr_max = 2.0

ni_min = 0.01
ni_max = 0.05

a, nr, ni = cal_feature_space(a_min, a_max,
                              nr_min, nr_max,
                              ni_min, ni_max, num)

#get the maximal order of the integration
l = get_order(a_max, lambDa)

# pre-calculate the scatter matrix and the incident field
scatter_matrix, E_incident = cal_scatter_matrix_Ei(lambDa, k_dir, E, res, fov, l, padding,
                                    working_dis)

# allocate space for data set
sphere_data = np.zeros((3, num, num, num))
B_data_real = np.zeros((l.shape[0], num, num, num))
B_data_imag = np.zeros((l.shape[0], num, num, num))

#%%
# generate data sets
# initialize a progress counter
cnt = 0

# the data set is too big to fit it as a whole in the memory
# therefore split them into sub sets for different sphere sizes
for h in range(num):
    # for each sphere size
    im_data = np.zeros((simRes, simRes, 2, num, num))
    # generate file name
    im_dir = r'D:\irimages\irholography\CNN\data_v9_far_field\im_data' + '%3.3d'% (h)
    
    for i in range(num):
        for j in range(num):
            # for each specific sphere
            n0 = nr[i] + ni[j] * 1j
            a0 = a[h]

            # calculate B vector
            B = coeff_b(l, k, n0, a0)
            
            E_scatter_fft = np.sum(scatter_matrix * B, axis = 2) * scale_factor
            
            # shift the Forier transform of the scatttering field for visualization
            E_scatter_fftshift = np.fft.fftshift(E_scatter_fft)
            
            # convert back to spatial domain
            E_scatter_b4_shift = np.fft.ifft2(E_scatter_fft)
            
            # shift the scattering field in the spacial domain for visualization
            E_scatter = np.fft.fftshift(E_scatter_b4_shift)
            
            Et = E_scatter + E_incident
            
            # save real and imaginary part of the field
            im_data[:, :, 0, i, j] = np.real(Et)
            im_data[:, :, 1, i, j] = np.imag(Et)
            
            # save label
            sphere_data[:, i, j, h] = [nr[i], ni[j], a[h]]
            
            #save B vector
            B_data_real[:np.size(B), i, j, h] = np.real(B)
            B_data_imag[:np.size(B), i, j, h] = np.imag(B)
            
            # print progress
            cnt += 1
            sys.stdout.write('\r' + str(cnt / (num_samples) * 100)  + ' %')
            sys.stdout.flush() # important

    # save the data
    np.save(im_dir, im_data)
    
np.save(r'D:\irimages\irholography\CNN\data_v9_far_field\lable_data', sphere_data)
np.save(r'D:\irimages\irholography\CNN\data_v9_far_field\B_data_real', B_data_real)
np.save(r'D:\irimages\irholography\CNN\data_v9_far_field\B_data_imag', B_data_imag)
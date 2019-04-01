# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:22:52 2019

data visualizatioin

plot out the training and testing images

Editor:
    Shihao Ran
    STIM Laboratory
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as animation
import sys

#%%
def apply_bpf(Etot, bpf):
    # apply the bandpass filter to the input field
    
    #2D fft to the input field
    Et_d = np.fft.fft2(Etot)
    
    #apply bandpass filter in the fourier domain
    Et_d *= bpf
    
    #invert FFT back to spatial domain
    Et_bpf = np.fft.ifft2(Et_d)
    
    #initialize cropping
    cropsize = res
    startIdx = int(np.fix(simRes /2) - np.floor(cropsize/2))
    endIdx = int(startIdx + cropsize - 1)
    
    D_Et = np.zeros((cropsize, cropsize), dtype = np.complex128)
    D_Et = Et_bpf[startIdx:endIdx+1, startIdx:endIdx+1]
    
    return D_Et

def BPF(halfgrid, simRes, NA_in, NA_out):
    #create a bandpass filter
    #change coordinates into frequency domain
        
    df = 1/(halfgrid*2)
    
    iv, iu = np.meshgrid(np.arange(0, simRes, 1), np.arange(0, simRes, 1))
    
    u = np.zeros(iu.shape)
    v = np.zeros(iv.shape)
    
    #initialize the filter as All Pass
    BPF = np.ones(iv.shape)
    
    idex1, idex2 = np.where(iu <= simRes/2)
    u[idex1, idex2] = iu[idex1, idex2]
    
    idex1, idex2 = np.where(iu > simRes/2)
    u[idex1, idex2] = iu[idex1, idex2] - simRes +1
    
    u *= df
    
    idex1, idex2 = np.where(iv <= simRes/2)
    v[idex1, idex2] = iv[idex1, idex2]
    
    idex1, idex2 = np.where(iv > simRes/2)
    v[idex1, idex2] = iv[idex1, idex2] - simRes +1
    
    v *= df
    
    magf = np.sqrt(u ** 2 + v ** 2)
    
    #block lower frequency
    idex1, idex2 = np.where(magf < NA_in / lambDa)
    BPF[idex1, idex2] = 0
    #block higher frequency
    idex1, idex2 = np.where(magf > NA_out / lambDa)
    BPF[idex1, idex2] = 0
    
    return BPF
#%%
def bandpass_filtering(sphere_data, bpf):
    # apply bandpass filter to all the data sets
    
    # allocate space for the image data set
    bp_data_complex = np.zeros((num_test, res, res, 2))
    bp_data_intensity = np.zeros((num_test, res, res, 1))
    
    cnt = 0
    
    complex_im = sphere_data[..., 0] + sphere_data[..., 1] * 1j
    
    for i in range(num_test):
        filtered_im_complex = apply_bpf(complex_im[i, ...], bpf)
        bp_data_complex[i, :, :, 0] = np.real(filtered_im_complex)
        bp_data_complex[i, :, :, 1] = np.imag(filtered_im_complex)
        
        bp_data_intensity[i, :, :, 0] = np.abs(filtered_im_complex) ** 2     
        
        # print progress
        cnt += 1
        sys.stdout.write('\r' + str(cnt / num_test * 100)  + ' %')
        sys.stdout.flush() # important
    
    return bp_data_complex, bp_data_intensity


#%%
def anime(image, FPS, fname, option = 'Real', autoscale = False):
    img = []
    fig = plt.figure()
    
    if image.shape[-1] == 1:
        _min, _max = np.amin(image[...,0]), np.amax(image[...,0])  
        for i in range(image.shape[0]):
            if autoscale:
                img.append([plt.imshow(image[i,:,:,0], vmin = _min, vmax = _max)])
            else:
                img.append([plt.imshow(image[i,:,:,0])])
    else:
        if option == 'Real':
            _min, _max = np.amin(image[...,0]), np.amax(image[...,0])
            for i in range(image.shape[0]):
                if autoscale:
                    img.append([plt.imshow(image[i,:,:,0], vmin = _min, vmax = _max)])
                else:
                    img.append([plt.imshow(image[i,:,:,0])])
            
        elif option == 'Imaginary':
            _min, _max = np.amin(image[...,1]), np.amax(image[...,1])
            for i in range(image.shape[0]):
                if autoscale:
                    img.append([plt.imshow(image[i,:,:,1], vmin = _min, vmax = _max)])
                else:
                    img.append([plt.imshow(image[i,:,:,1])])
            
        else:
            print('Invalid Image Type')
            return
    
    ani = animation.ArtistAnimation(fig,img,interval=int(1000/FPS))
    writer = animation.writers['ffmpeg'](fps=FPS)
    
    ani.save(data_dir + '\\' + fname + '.mp4',writer=writer)
    
#%%
# load data set

raw_im = np.load(r'D:\irimages\irholography\CNN\data_v9_far_field\raw_data\im_data010.npy')
raw_im = np.reshape(raw_im, (640, 640, 2, 400))

raw_im = np.swapaxes(raw_im, 0, -1)
raw_im = np.swapaxes(raw_im, -2, -1)

data_dir = r'D:\irimages\irholography\CNN\data_v9_far_field\raw_data'

#%%
num_test = 400
lambDa = 1
NA_in = 0
NA_out = 0.25
res = 128
fov = 16
padding = 2
simRes = 640
halfgrid = np.ceil(fov / 2) * (padding * 2 + 1)


bpf = BPF(halfgrid, simRes, NA_in, NA_out)
im_bp_comp, im_bp_inten = bandpass_filtering(raw_im, bpf)

#%%
anime(im_bp_inten, 20, 'intensityBP0.25', 'Real')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 18:29:31 2021

@author: yanni
"""

import torch
import torch.fft as torch_fft
import torch.nn as nn

import numpy as np
import sigpy as sp


def rotate_batch_torch(img,thetas):
    #assumes img has shape: [batch , H,W]
    # theta has shape: [angles]
    x = np.linspace(-int(img.shape[-2]/2)+1, int(img.shape[-2]/2)+1, img.shape[-2])
    y = np.linspace(-int(img.shape[-1]/2)+1, int(img.shape[-1]/2)+1, img.shape[-2])
    xv, yv = np.meshgrid(x, y)
    r = torch.tensor(np.vstack((xv[None,...],yv[None,...]))).cuda()

    kx, ky = np.meshgrid(x, y)
    k = torch.tensor(np.vstack((kx[None,...],ky[None,...]))).cuda()

    scale_1 = torch.tensor(img.shape[-2]/(2*np.pi)).cuda()
    scale_2 = torch.tensor(img.shape[-1]/(2*np.pi)).cuda()

    thetas_r = torch.tensor(np.pi)*thetas/180
    sin = torch.sin(thetas_r).cuda()
    tan = torch.tan(thetas_r/2).cuda()


    v_sin_stack = torch.zeros((thetas.shape[0], img.shape[-2], img.shape[-1]), dtype = torch.complex64).cuda()
    v_tan_stack = torch.zeros((thetas.shape[0], img.shape[-2], img.shape[-1]), dtype = torch.complex64).cuda()

    for i in range(thetas.shape[0]):
        v_sin_stack[i,...] = torch.exp(-1j*sin[i]*r[0,...]*k[1,...]/scale_1)
        v_tan_stack[i,...] = torch.exp(1j*tan[i]*r[1,...]*k[0,...]/scale_2)

    norm_fft  = 'ortho'
    norm_ifft = 'ortho'

    #shear 1
    k_r = torch_fft.fftshift(torch_fft.fft(img, dim=-2,norm=norm_fft), dim = -2)
    #k_r here is [1,H,W]
    k_r = torch.mul(v_tan_stack, k_r)
    #k_r is now [angles, H,W]
    i_s1 = torch_fft.ifft(torch_fft.ifftshift(k_r,dim = -2), dim=-2,norm=norm_ifft)

    #shear 2
    k_r = torch_fft.fftshift(torch_fft.fft(i_s1, dim=-1,norm=norm_fft), dim = -1)
    k_r = torch.mul(v_sin_stack ,k_r)
    i_s2 = torch_fft.ifft(torch_fft.ifftshift(k_r,dim = -1), dim=-1,norm=norm_ifft)

    #shear 3
    k_r  = torch_fft.fftshift(torch_fft.fft(i_s2, dim=-2,norm=norm_fft), dim = -2)
    k_r  = torch.mul(v_tan_stack, k_r)
    i_s3 = torch_fft.ifft(torch_fft.ifftshift(k_r,dim = -2), dim=-2,norm=norm_ifft)

    return i_s3#, i_s2, i_s1


def translate_batch_torch(img, dx, dy):
    #img shape = [TRs, H, W]
    #dx, dy = [TRs]
    x = np.linspace(-int(img.shape[-2]/2)+1, int(img.shape[-2]/2), img.shape[-2])
    y = np.linspace(-int(img.shape[-1]/2)+1, int(img.shape[-1]/2), img.shape[-1])

    kx, ky = np.meshgrid(x, y)
    k = torch.tensor(np.vstack((kx[None,...],ky[None,...]))).cuda()
    scale_1 = torch.tensor(img.shape[-2]/(2*np.pi))
    scale_2 = torch.tensor(img.shape[-1]/(2*np.pi))

    ksp = fft(img)

    phase_stack = torch.zeros((dx.shape[0],img.shape[-2],img.shape[-1]), dtype = torch.complex64).cuda()
    for i in range(dx.shape[0]):
        phase_stack[i] = torch.exp(-1j*(k[0,...]*dx[i]/scale_1 + k[1,...]*dy[i]/scale_2))

    ksp_shift = ksp*phase_stack
    i_T = ifft(ksp_shift)




    return i_T#, i_s2, i_s1




# Centered, orthogonal ifft in torch >= 1.7
def ifft(x):
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    x = torch_fft.ifft2(x, dim=(-2, -1), norm='ortho')
    x = torch_fft.fftshift(x, dim=(-2, -1))
    return x

# Centered, orthogonal fft in torch >= 1.7
def fft(x):
    x = torch_fft.fftshift(x, dim=(-2, -1))
    x = torch_fft.fft2(x, dim=(-2, -1), norm='ortho')
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    return x

def get_mvue(kspace, s_maps):
    ''' Get mvue estimate from coil measurements '''
    return np.sum(sp.ifft(kspace, axes=(-1, -2)) * \
                  np.conj(s_maps), axis=1)
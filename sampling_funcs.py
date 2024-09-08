# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from utils import nrmse
from motion_ops import motion_forward



#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])



def ODE_motion_sampler(net, y, maps, traj, latents, img_l_ss=1.0, motion_l_ss=1.0, second_order=False,
                     class_labels=None, randn_like=torch.randn_like, num_steps=100,
                      sigma_min=0.002, sigma_max=80, rho=7, S_churn=0,S_min=0,  
                      S_max=float('inf'), S_noise=1, verbose=True, motion_est=1,
                      gt_img=None, gt_theta=None, gt_dx=None, gt_dy=None,group_ETL=True):
    img_stack = []
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0


    # initialize motion estimates
    if group_ETL:
        est_theta = torch.zeros_like(gt_theta)
        est_dx    = torch.zeros_like(gt_dx)
        est_dy    = torch.zeros_like(gt_dy)
    else:
        est_theta = torch.zeros(traj.shape[0]).cuda()
        est_dx    = torch.zeros(traj.shape[0]).cuda()
        est_dy    = torch.zeros(traj.shape[0]).cuda()
    

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        x_hat = x_cur
        x_hat = x_hat.requires_grad_() #starting grad tracking with the noised img

        # Euler step.
        denoised = net(x_hat, t_cur, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised)/t_cur
        # take step over prior score and add noise
        x_next = x_hat + (t_next - t_cur) * d_cur #+ ((2*t_cur)*(t_cur-t_next))**0.5 * randn_like(x_cur)
        # print(((2*t_cur)*(t_cur-t_next))**0.5)
        # print(t_cur-t_next)
        # Likelihood step
        denoised_cplx = torch.view_as_complex(denoised.permute(0,-2,-1,1).contiguous())[None]

        Ax = motion_forward(image=denoised_cplx, s_maps=maps, coords=traj,
                             angles=est_theta, dx=est_dx, dy=est_dy, 
                             device=denoised_cplx.device)
        residual = y - Ax
        sse = torch.norm(residual)**2
        likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_hat)[0]

        # take step on motion parameters
        if motion_est:
            est_theta = est_theta.requires_grad_()
            est_dy    = est_dy.requires_grad_()
            est_dx    = est_dx.requires_grad_()
            # x_next_cplx = torch.view_as_complex(x_next.permute(0,-2,-1,1).contiguous())[None]
            x_next_cplx = denoised_cplx
            Ax = motion_forward(image=x_next_cplx, s_maps=maps, coords=traj,
                             angles=est_theta, dx=est_dx, dy=est_dy, 
                             device=x_next_cplx.device)
            residual = y - Ax
            sse_m = torch.norm(residual)**2
            meas_grad_motion = torch.autograd.grad(outputs = sse_m, inputs = (est_theta, est_dx, est_dy), create_graph = not True)
            norm = 100#torch.sqrt(sse_m)
            est_theta = est_theta - (motion_l_ss/norm)*meas_grad_motion[0]
            est_dx = est_dx - (motion_l_ss/norm)*meas_grad_motion[1]
            est_dy = est_dy - (motion_l_ss/norm)*meas_grad_motion[2]

        #take step on image
        x_next = x_next - (img_l_ss / torch.sqrt(sse)) * likelihood_score

        # Cleanup 
        x_next = x_next.detach()
        x_hat = x_hat.requires_grad_(False)
        # est_theta = est_theta.requires_grad_(False)
        # est_dx = est_dx.requires_grad_(False)
        # est_dy = est_dy.requires_grad_(False)


        with torch.no_grad():
            if verbose:
                if group_ETL:
                    cplx_recon = torch.view_as_complex(x_next.permute(0,-2,-1,1).contiguous())[None]
                    img_nrmse =  nrmse(abs(gt_img), abs(cplx_recon)).item()
                    theta_nrmse =  nrmse(gt_theta, est_theta).item()
                    dx_nrmse =  nrmse(gt_dx, est_dx).item()
                    dy_nrmse =  nrmse(gt_dy, est_dy).item()
                    print('Step:%d , img NRMSE: %.3f, theta NRMSE: %.3f, dx NRMSE: %.3f, dy NRMSE: %.3f'%(i, img_nrmse, theta_nrmse, dx_nrmse, dy_nrmse))
                else:
                    cplx_recon = torch.view_as_complex(x_next.permute(0,-2,-1,1).contiguous())[None]
                    img_nrmse =  nrmse(abs(gt_img), abs(cplx_recon)).item()
                    print('Step:%d , img NRMSE: %.3f'%(i, img_nrmse))
    return x_next, est_theta, est_dx, est_dy
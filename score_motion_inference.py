#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from lzma import MODE_FAST
import sys
sys.path.insert(0, './bart-0.6.00/python')
sys.path.append('./bart-0.6.00/python')
import time
import torch, os, argparse
import numpy as np
import sigpy as sp
import copy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOOLBOX_PATH"]    = './bart-0.6.00'
# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
from dotmap import DotMap
from ncsnv2.models.ncsnv2 import NCSNv2Deepest

from utils import MulticoilForwardMRI, get_mvue
from utils import ifft, normalize, normalize_np, unnormalize

from tqdm import tqdm
from matplotlib import pyplot as plt

from skimage.metrics import structural_similarity as ssim_loss
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from torch.nn import MSELoss
import glob
from trajectory_creator import Cart2D_traj_gen, prop2D_traj_gen
from motion_ops import motion_adjoint, motion_forward, motion_normal

def nrmse(X,Y):
    error_norm = torch.norm(X - Y, p=2)
    self_norm  = torch.norm(X,p=2)

    return error_norm / self_norm


# Seeds
torch.manual_seed(2021)
np.random.seed(2021)

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--noise_boost', type=float, default=1.0)
parser.add_argument('--normalize_grad', type=int, default=1)
parser.add_argument('--dc_boost', type=float, default=1.)
parser.add_argument('--motion_noise_boost', type=float, default=1.)
parser.add_argument('--step_lr', nargs='+', type=float, default=[9e-6])
parser.add_argument('--lambda_2', type=float, default=1.)
parser.add_argument('--motion_lr_init', type=float, default=1.)
parser.add_argument('--motion_norm', type=int, default=1)
parser.add_argument('--gamma', type=float, default=1.)
parser.add_argument('--beta', type=float, default=1.)
parser.add_argument('--m_noise_scale', type=float, default=1.)
parser.add_argument('--motion_est', type=int, default=1)
parser.add_argument('--R', type=int, default=1)
parser.add_argument('--est_start', type=int, default=0)
parser.add_argument('--skip_levels', type=int, default=0)
parser.add_argument('--level_steps', type=int, default=4)
parser.add_argument('--sample_num', type=int, default=0)
parser.add_argument('--traj_type', type=str, default='Cart')

args   = parser.parse_args()

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu);

# Load a diffusion model
target_file = '/home/blevac/Score_Motion_Correction/ncsnv2-mri-mvue/logs/mri-mvue/checkpoint_100000.pth'
# Model config
config        = DotMap()
config.device = 'cuda:0'
# Inner model
config.model.normalization = 'InstanceNorm++'
config.model.nonlinearity  = 'elu'
config.model.sigma_dist    = 'geometric'
config.model.sigma_begin   = 232
config.model.sigma_end     = 0.00066
config.model.num_classes   = 2311
config.model.ngf           = 128
# Data
meta_step_lr = args.step_lr
config.sampling.log_steps      = 5
config.sampling.snapshot_steps = 100
# Inference
config.inference.sigma_offset   = args.skip_levels#800 # !!! Skip some early (huge) sigmas
config.inference.num_steps_each = args.level_steps   # !!! More budget here
config.inference.noise_boost    = args.noise_boost
config.inference.num_steps      = \
    config.model.num_classes - config.inference.sigma_offset # Leftover

# Data
config.data.channels = 2
# Model
diffuser = NCSNv2Deepest(config)
diffuser = torch.nn.DataParallel(diffuser)
# Load weights
model_state = torch.load(target_file)
diffuser.load_state_dict(model_state[0], strict=True)
# Switch to eval mode and extract module
diffuser = diffuser.module
diffuser.eval()

#set inference parameters
lambda_2          = torch.tensor(args.lambda_2).cuda()
motion_lr_init    = args.motion_lr_init
beta              = args.beta
gamma             = args.gamma
motion_est        = args.motion_est
est_start         = args.est_start



#load validation sample
R           = args.R
folder          = sorted(glob.glob('/home/blevac/Score_Motion_Correction/data_ISBI_final/ETL_8_TR_48_Uniform2.00/*.pt'))
contents        = torch.load(folder[args.sample_num])
local_maps      = torch.tensor(contents['new_maps']).cuda() #shape: [1,coils, H, W]
local_gt_thetas = torch.tensor(contents['gt_theta'][0::R]).cuda() #shape: [TRs]
local_gt_dx     = torch.tensor(contents['gt_dx'][0::R]).cuda() #shape: [TRs]
local_gt_dy     = torch.tensor(contents['gt_dy'][0::R]).cuda() #shape: [TRs]
local_gt_img    = torch.tensor(contents['gt_img']).cuda()
local_coil_imgs = torch.tensor(contents['gt_coil_imgs']).cuda()
noise_lvl       = contents['noise_lvl']

img_prog = []
theta_prog = []
dx_prog = []
dy_prog = []
# create sampling patterns and simulate corrupted measruements
if args.traj_type =='Cart':
    coords = Cart2D_traj_gen(TRs=48, ETL=8, N_RO=384, ro_dir='x', ordering='interleave')
    ktraj = torch.tensor(coords[0::R]).cuda()
elif args.traj_type =='Prop':
    coords = prop2D_traj_gen(TRs=48, ETL=8, N_RO=384)
    accel_TRs = 10
    ktraj = torch.tensor(coords[0:accel_TRs]).cuda()

ACS_traj = torch.tensor(Cart2D_traj_gen(TRs=1, ETL=24, N_RO=24, ro_dir = 'y', ordering='linear')).cuda()

meas_ksp = motion_forward(image = local_coil_imgs, s_maps=torch.ones_like(local_maps), coords=ktraj, angles=local_gt_thetas, dx=local_gt_dx, dy=local_gt_dy, device=config.device)

ACS_img= motion_normal(image = local_coil_imgs, s_maps=torch.ones_like(local_maps), coords=ACS_traj, angles=torch.zeros(1)[None].cuda(), dx=torch.zeros(1)[None].cuda(), dy=torch.zeros(1)[None].cuda(), device=config.device)
#local_ksp.shape = [1, Coils, TRs, N_RO*ETL]
meas_ksp = meas_ksp.detach().cpu().numpy()
local_ksp = torch.tensor(meas_ksp).cuda()
#add IID noise to have a more accurate simulation
local_ksp_clean = copy.deepcopy(local_ksp)
local_ksp = local_ksp + noise_lvl*(torch.randn(local_ksp.shape).cuda()+1j*torch.randn(local_ksp.shape).cuda())

#initialize motion estimates
if motion_est:
    est_thetas = 0.01*torch.randn_like(local_gt_thetas, requires_grad=True).cuda()
    est_dx     = 0.01*torch.randn_like(local_gt_dx, requires_grad=True).cuda()
    est_dy     = 0.01*torch.randn_like(local_gt_dy, requires_grad=True).cuda()
    
# if you arent esimtating motion load the gt motion values for recon
else:
    est_thetas = copy.deepcopy(local_gt_thetas).cuda()
    est_dx = copy.deepcopy(local_gt_dx).cuda()
    est_dy = copy.deepcopy(local_gt_dy).cuda()
    motion_lr_init = 0.0

# For each hyperparameter
for idx, local_step_lr in enumerate(meta_step_lr):
    # Set configuration
    config.sampling.step_lr = local_step_lr

    # Global metrics
    num_metric_steps     = int(np.ceil((config.inference.num_steps) /\
                                        config.sampling.log_steps))

    # Global outputs
    num_log_steps        = int(np.ceil((config.inference.num_steps) /\
                                        config.sampling.snapshot_steps))



    # Results
    result_dir = 'results/sample%d_added_noise%.1e/accel_%d/ktraj_%s/skip_noise_lvl_%d_level_steps%d_est_start%d_normMeas%d_dcBoost%.1f_motionNoise%.1e_motionNorm%d' % (
        args.sample_num,noise_lvl,R,args.traj_type,args.skip_levels,args.level_steps,est_start, args.normalize_grad, args.dc_boost,args.m_noise_scale, args.motion_norm)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_nrmse_list = []
    theta_nrmse_list = []
    dx_nrmse_list = []
    dy_nrmse_list = []

    # # Get ZF MVUE
    with torch.no_grad():
        if args.traj_type == 'Cart':
            if not bool(motion_est):
                estimated_mvue = motion_adjoint(ksp=local_ksp, s_maps=local_maps, coords=ktraj,
                angles=local_gt_thetas, dx=local_gt_dx, 
                dy=local_gt_dy, img_shape=local_maps.shape, device=config.device)
            elif bool(motion_est):
                estimated_mvue = motion_adjoint(ksp=local_ksp, s_maps=local_maps, coords=ktraj,
                angles=torch.zeros_like(local_gt_thetas), dx=torch.zeros_like(local_gt_dx), 
                dy=torch.zeros_like(local_gt_dy), img_shape=local_maps.shape, device=config.device)
        elif args.traj_type == 'Prop':
            #only use one ACS to calculate the est_mvue scaling
            estimated_mvue = ACS_img

        norm = torch.quantile(estimated_mvue.abs(), 0.99)

    # Initialize starting point for drawing samples using langevin dynamics [B, 2(real,imag), H,W]
    samples = torch.rand(1, config.data.channels,
                         local_gt_img.shape[-2], local_gt_img.shape[-1],
                         dtype=torch.float32).cuda()

    # FINALLY the Inference, for each noise level
    for noise_idx in tqdm(range(config.inference.num_steps)):
        # Get current noise power
        sigma  = diffuser.sigmas[noise_idx + config.inference.sigma_offset]
        labels = torch.ones(samples.shape[0],
                            device=samples.device) * \
            (noise_idx + config.inference.sigma_offset)
        labels = labels.long()

        image_step_size = config.sampling.step_lr * (
            sigma / diffuser.sigmas[-1]) ** 2
        # will likely use a schedule later   
        motion_step_size = torch.tensor(motion_lr_init) #motion_lr_schedule[noise_idx]
        # For each step spent there
        for step_idx in range(config.inference.num_steps_each):
            with torch.no_grad():
                # Generate noise
                image_noise = torch.randn_like(samples) * \
                    torch.sqrt(args.noise_boost * image_step_size * 2)

                theta_noise = torch.randn_like(local_gt_thetas) * \
                    torch.sqrt(args.motion_noise_boost * motion_step_size)
                dx_noise = torch.randn_like(local_gt_dx) * \
                    torch.sqrt(args.motion_noise_boost * motion_step_size)
                dy_noise = torch.randn_like(local_gt_dy) * \
                    torch.sqrt(args.motion_noise_boost * motion_step_size)

                ##############stage one: generate gradients w.r.t image#######################
                # get score from model
                p_grad_start = time.time()
                p_grad = diffuser(samples.float(), labels)
                p_grad_end = time.time()
                ######may need to make a complex version of "samples" as input to operators###############
                copy_samples = copy.deepcopy(samples).permute(0,-2,-1,1)
                cplx_samples = torch.view_as_complex(copy_samples.contiguous())[None,...] # want the shape as [1,1,H,W]
                # get measurements and DC loss for current estimate
                meas_start = time.time()
                meas    = motion_forward(image = cplx_samples*norm, s_maps=local_maps, coords=ktraj, angles=est_thetas, dx=est_dx, dy=est_dy, device=config.device)
                meas_end = time.time()
                dc      = meas - local_ksp
                dc_loss = torch.norm(dc, p = 2)**2
# both are normalized kind-of
                if bool(args.normalize_grad):
                    # normalize 
                    # compute gradient, i.e., gradient = A_adjoint * ( y - Ax_hat )
                    meas_grad_start = time.time()
                    meas_grad = 2*torch.view_as_real(motion_adjoint(ksp=dc, s_maps=local_maps, coords=ktraj,
                                                    angles=est_thetas, dx=est_dx, dy=est_dy, img_shape=local_maps.shape,
                                                    device=config.device))[:,0,...].permute(0, 3, 1, 2)
                    # meas_grad shape : [B,2,H,W]
                    meas_grad_end = time.time()
                    # Normalize
                    # to make the gradient importance relatively the same
                    meas_grad = meas_grad / torch.norm(meas_grad)
                    meas_grad = meas_grad * torch.norm(p_grad)
                else:
                    # compute gradient, i.e., gradient = A_adjoint * ( y - Ax_hat )
                    # meas_grad = 2 * torch.view_as_real(torch.sum(ifft(dc_loss) * torch.conj(local_maps), axis=1) / (sigma ** 2)).permute(0, 3, 1, 2)

                    meas_grad = 2*torch.view_as_real(motion_adjoint(ksp = local_ksp, s_maps=local_maps, coords=ktraj,
                                                    angles=est_thetas, dx=est_dx, dy=est_dy, img_shape=local_maps.shape,
                                                    device=config.device)\
                     - motion_normal(image = cplx_samples, s_maps=local_maps, coords=ktraj, angles=est_thetas, dx=est_dx, dy=est_dy, device=config.device)/ (sigma ** 2)).permute(0, 3, 1, 2)
                    # re-normalize, since measuremenets are from a normalized estimate
                    meas_grad = unnormalize(meas_grad, estimated_mvue)

                # combine measurement gradient, prior gradient and noise
                samples_prev      = copy.deepcopy(samples)
                samples_prev_cplx = copy.deepcopy(cplx_samples)
                # compute gradient step for image
                samples = samples + image_step_size * ( p_grad - args.dc_boost * meas_grad) + image_noise
            if motion_est and (noise_idx>=args.est_start):
                #want to enable gradient tracking here unlike above
                ##############stage two: generate gradients w.r.t motion parameters#######################
                residual   = motion_forward(image = samples_prev_cplx*norm, s_maps=local_maps, coords=ktraj, angles=est_thetas, dx=est_dx, dy=est_dy, device=config.device) - local_ksp
                motion_likelihood_mse = torch.norm(input = residual, p = 2)**2
                motion_prior_mse      = lambda_2*(torch.norm(input = est_thetas, p = 2)**2 + torch.norm(input = est_dx, p = 2)**2 + torch.norm(input = est_dy, p = 2)**2)

                meas_grad_motion = torch.autograd.grad(outputs = motion_likelihood_mse, inputs = (est_thetas, est_dx, est_dy), create_graph = not True)
                prior_grad_motion = torch.autograd.grad(outputs = motion_prior_mse, inputs = (est_thetas, est_dx, est_dy), create_graph = not True)
                #noramlize gradients like above
                if bool(args.motion_norm):
                    theta_meas_grad = meas_grad_motion[0] / torch.norm(meas_grad_motion[0])
                    theta_meas_grad = theta_meas_grad * torch.norm(prior_grad_motion[0])
                    dx_meas_grad = meas_grad_motion[1] / torch.norm(meas_grad_motion[1])
                    dx_meas_grad = dx_meas_grad * torch.norm(prior_grad_motion[1])
                    dy_meas_grad = meas_grad_motion[2] / torch.norm(meas_grad_motion[2])
                    dy_meas_grad = dy_meas_grad * torch.norm(prior_grad_motion[2])
                else:
                    theta_meas_grad = meas_grad_motion[0]
                    dx_meas_grad = meas_grad_motion[1]
                    dy_meas_grad = meas_grad_motion[2]


                est_thetas = est_thetas - motion_step_size*(theta_meas_grad - prior_grad_motion[0]) + args.m_noise_scale*theta_noise
                est_dx     = est_dx - motion_step_size*(dx_meas_grad - prior_grad_motion[1]) + args.m_noise_scale*dx_noise
                est_dy     = est_dy - motion_step_size*(dy_meas_grad - prior_grad_motion[2]) + args.m_noise_scale*dy_noise
                est_thetas = torch.clamp(input=est_thetas, min=-15, max=15) #for stability
                est_dx     = torch.clamp(input=est_dx, min=-15, max=15) #for stability
                est_dy     = torch.clamp(input=est_dy, min=-15, max=15) #for stability




            #performance metrics
            with torch.no_grad():
                # est_thetas.requires_grad = True
                normalized_cplx_samples = cplx_samples*norm
                img_nrmse = nrmse(local_gt_img, normalized_cplx_samples)
                theta_nrmse = nrmse(local_gt_thetas, est_thetas)
                dx_nrmse = nrmse(local_gt_dx, est_dx)
                dy_nrmse = nrmse(local_gt_dy, est_dy)

                img_nrmse_list.append(img_nrmse.detach().cpu())
                theta_nrmse_list.append(theta_nrmse.detach().cpu())
                dx_nrmse_list.append(dx_nrmse.detach().cpu())
                dy_nrmse_list.append(dy_nrmse.detach().cpu())

                print('dc loss:', dc_loss.item(), ',  img_nrmse:',img_nrmse.item(),
                ',  theta_nrmse:',theta_nrmse.item(), ',  dx_nrmse:',dx_nrmse.item(),
                ',  dy_nrmse:',dy_nrmse.item())

            # print(norm)
            if noise_idx%100 == 0 and step_idx ==0:
                img_prog.append(normalized_cplx_samples.cpu())
                theta_prog.append(est_thetas.detach().cpu())
                dx_prog.append(est_dx.detach().cpu())
                dy_prog.append(est_dy.detach().cpu())
                # Save to file
                filename = result_dir + '/motion_est%d_imgnoise%.2e_motionnoise%.2e_step%.2e_motionLRinit_%.3e_beta%.3f_gamm%.3f_lamda%.2f.pt' % ( motion_est, args.noise_boost,args.motion_noise_boost ,local_step_lr, motion_lr_init, beta, gamma ,lambda_2)
                torch.save({'Recon_img': normalized_cplx_samples.cpu(),
                            'img_prog': img_prog,
                            'theta_prog': theta_prog,
                            'dx_prog': dx_prog,
                            'dy_prog': dy_prog,
                            'est_mvue': estimated_mvue.cpu(),
                            'GT_img': local_gt_img.cpu(),
                            'gt_thetas':local_gt_thetas.cpu(),
                            'gt_dx': local_gt_dx.cpu(),
                            'gt_dy': local_gt_dy.cpu(),
                            'est_thetas': est_thetas.cpu(),
                            'est_dx': est_dx.cpu(),
                            'est_dy': est_dy.cpu(),
                            'theta_nrmse': theta_nrmse_list,
                            'dx_nrmse': dx_nrmse_list,
                            'dy_nrmse': dy_nrmse_list,
                            'img_nrmse':img_nrmse_list,
                            'ktraj': ktraj.cpu(),
                            'noise_lvl': noise_lvl,
                            'kspace': local_ksp,
                            'kspace_clean': local_ksp_clean,
                            'maps':local_maps.cpu(),
                            'gt_coil_imgs':local_coil_imgs,
                            'args': args}, filename)

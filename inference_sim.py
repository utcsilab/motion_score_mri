# boiler plate imports
import numpy as np
import torch
from tqdm import tqdm
# import sigpy as sp
import matplotlib.pyplot as plt
import os
import argparse
from utils import nrmse
from sampling_funcs import StackedRandomGenerator,ODE_motion_sampler
import pickle
import dnnlib
from torch_utils import distributed as dist
# dist.init()


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--img_l_ss', type=float, default=2)
parser.add_argument('--mot_l_ss', type=float, default=1)
parser.add_argument('--S_noise', type=float, default=0)
parser.add_argument('--sigma_max', type=float, default=5)
parser.add_argument('--num_steps', type=int, default=300)
parser.add_argument('--group_ETL', type=int, default=1)
parser.add_argument('--R', type=int, default=3)
parser.add_argument('--ETL', type=int, default=16)
parser.add_argument('--TR', type=int, default=24)
parser.add_argument('--pat', type=str, default='cart')
parser.add_argument('--sample', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--motion_est', type=int, default=1)
parser.add_argument('--net_arch', type=str, default='ddpmpp') 
parser.add_argument('--discretization', type=str, default='edm') # ['vp', 've', 'iddpm', 'edm']
parser.add_argument('--solver', type=str, default='euler') # ['euler', 'heun']
parser.add_argument('--schedule', type=str, default='vp') # ['vp', 've', 'linear']
parser.add_argument('--scaling', type=str, default='vp') # ['vp', 'none']

args   = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
#seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device=torch.device('cuda')


# load sample 
# data_file = '/csiNAS2/slow/brett/diff_mo_co_9_2_23_noisy/sim_inf_data/%s_R%d_ETL%d_TR%d/sample%d.pt'%(args.pat,args.R,args.ETL,args.TR,args.sample)
data_file = '/csiNAS2/slow/brett/newest_diff_mo_co_12_7_23_val/sim_val_data/%s_R%d_ETL%d_TR%d/sample1.pt'%(args.pat,args.R,args.ETL,args.TR)
contents = torch.load(data_file)


s_maps        = contents['maps'].cuda() # shape: [1,C,H,W]
ksp           = contents['ksp'].cuda()# shape: [1,C,H,W]
traj          = contents['traj'].cuda()# shape: [1,C,H,W]


if not args.group_ETL:
    N_RO = traj.shape[1]//args.ETL
    print('Not assuming ETL groupings')
    ktraj_reshaped = traj.reshape(traj.shape[0],N_RO,args.ETL,2)
    ktraj_reshaped = ktraj_reshaped.permute(0,2,1,-1)
    traj = ktraj_reshaped.reshape(-1,N_RO,2)
    ksp = ksp.reshape(ksp.shape[0],ksp.shape[1], 1, ksp.shape[2]*args.ETL, N_RO)




gt_img        = contents['gt_img'].cuda() # shape [1,1,H,W]
gt_theta      = contents['gt_theta'].cuda() # shape [TR]
gt_dx         = contents['gt_dx'].cuda() # shape [TR]
gt_dy         = contents['gt_dy'].cuda() # shape [TR]
norm_c        = contents['norm'].cuda() # scalar

# normalize
ksp = ksp/norm_c
gt_img = gt_img/norm_c

batch_size = 1

results_dir = './results_sim_12_7_23/%s_motionEst%d_R%d_ETL%d_TR%d_groupETL%d/net-%s_step-%d_imglss-%.1e_mot_lss-%.1e_sigmaMax-%.1e/sample%d/seed%d/'%(args.pat, args.motion_est, args.R, args.ETL, args.TR, args.group_ETL, args.net_arch, args.num_steps,  args.img_l_ss, args.mot_l_ss, args.sigma_max, args.sample, args.seed)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# load network
net_save = '/csiNAS2/slow/brett/edm_outputs/00027-T2-uncond-ddpmpp-edm-gpus4-batch16-fp32-T2/network-snapshot-010000.pkl'
if dist.get_rank() != 0:
        torch.distributed.barrier()

# Load network.
dist.print0(f'Loading network from "{net_save}"...')
with dnnlib.util.open_url(net_save, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)



# Pick latents and labels.
rnd = StackedRandomGenerator(device, [args.seed])
latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
class_labels = None
class_idx = None
if net.label_dim:
    class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
if class_idx is not None:
    class_labels[:, :] = 0
    class_labels[:, class_idx] = 1




image_recon, est_theta, est_dx, est_dy = ODE_motion_sampler(net=net, y=ksp, maps=s_maps,traj=traj, 
                                        latents=latents, img_l_ss=args.img_l_ss,motion_l_ss=args.mot_l_ss, class_labels=class_labels, 
                                        randn_like=torch.randn_like, num_steps=args.num_steps, sigma_min=0.002, 
                                        sigma_max=args.sigma_max, rho=7, S_churn=420, S_min=0, S_max=float('inf'),
                                        S_noise=args.S_noise, motion_est = args.motion_est,
                                        gt_img=gt_img, gt_theta=gt_theta, gt_dx=gt_dx, gt_dy=gt_dy, group_ETL=args.group_ETL)


cplx_recon = torch.view_as_complex(image_recon.permute(0,-2,-1,1).contiguous())[None] #shape: [1,1,H,W]


print('Sample %d, Seed %d, NRMSE: %.3f'%(args.sample,args.seed, nrmse(abs(gt_img), abs(cplx_recon)).item()))

dict = { 'gt_img': gt_img.cpu().numpy(),
        'recon':cplx_recon.cpu().numpy(),
        'est_theta': est_theta.detach().cpu().numpy(),
        'est_dx': est_dx.detach().cpu().numpy(),
        'est_dy': est_dy.detach().cpu().numpy(),
        'gt_theta': gt_theta.cpu().numpy(),
        'gt_dx': gt_dx.cpu().numpy(),
        'gt_dy': gt_dy.cpu().numpy(),
        'ktraj':traj.cpu().numpy()

}

torch.save(dict, results_dir + '/checkpoint.pt')



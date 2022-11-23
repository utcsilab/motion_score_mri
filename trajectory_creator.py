import numpy as np
import torch
import matplotlib.pyplot as plt
import os, sys
os.environ["TOOLBOX_PATH"] = "/home/blevac/misc/HW/bart-0.6.00"  ## change to YOUR path
sys.path.append("/home/blevac/misc/HW/bart-0.6.00/python")        ## change to YOUR path
from bart import bart

# functions for PROPELLER
def rot_mat(theta):
    theta = np.pi * theta/180
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, 0), (s, c, 0), (0,0,0)))
    return R

def prop2D_traj_gen(TRs, ETL, N_RO):
    base_traj = bart(1, 'traj -x '+str(N_RO)+' -y '+str(ETL)).real[...]
    base_flat = base_traj.reshape(3,-1)
    angle = []
    full_traj = []
    for tr in range(TRs):
        angle.append(tr*111.25)
        full_traj.append(np.matmul(rot_mat(angle[tr]),base_flat))

    return np.transpose(np.asarray(full_traj), (0,-1,-2))[:,:,0:2]



#trajectory generator for Cartesian FSE

def Cart2D_traj_gen(TRs, ETL, N_RO, ro_dir = 'y', ordering='interleave'):
    # ordering = 'linear', 'center_out', 'interleave' 
    if ro_dir == 'x':
        base_traj = bart(1, 'traj -x '+str(N_RO)+' -y '+str(ETL*TRs)).real[...]
        out_traj = np.zeros((3, TRs, N_RO, ETL))
        
        if ordering == 'linear':
            for i in range(TRs):
                out_traj[:, i, :, :] = base_traj[:,:,i*ETL:(i+1)*ETL]
        if ordering == 'interleave':
            for i in range(TRs):
                out_traj[:, i, :, :] = base_traj[:,:,i::TRs]
    
    if ro_dir == 'y':
        base_traj = bart(1, 'traj -x '+str(ETL*TRs)+' -y '+str(N_RO)).real[...]
        out_traj = np.zeros((3, TRs, ETL, N_RO))
        
        if ordering == 'linear':
            for i in range(TRs):
                out_traj[:, i, :, :] = base_traj[:,i*ETL:(i+1)*ETL,:]

        if ordering == 'interleave':
            for i in range(TRs):
                out_traj[:, i, :, :] = base_traj[:,i::TRs,:]

        if ordering == 'center_out':
            for i in range(TRs):
                if i<=int(TRs/2):
                    out_traj[:,i,:,:] = base_traj[:,i:int(ETL*TRs/2):floor(TRs/2),:]
                else:
                    out_traj[:,i,:,:] = base_traj[:,int(ETL*TRs/2) + int(i-TRs/2):ETL*TRs:floor(TRs/2),:]


    return np.transpose(out_traj.reshape(3,TRs, ETL*N_RO), (-2,-1,0))[...,0:2]
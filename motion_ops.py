import numpy as np
import torch
from Nufft_Torch.nufft import nufft, nufft_adjoint
import time

#forward and adjoint function defs
def rot_batch(coords, angles, device):
    rot_coords = torch.zeros_like(coords)
    for i in range(len(angles)):
        rot_mat = torch.zeros(2,2, dtype=torch.double).to(device)
        theta = torch.tensor(np.pi) * angles[i]/180

        rot_mat[0,0] = torch.cos(theta)
        rot_mat[0,1] = -1*torch.sin(theta)
        rot_mat[1,0] = torch.sin(theta)
        rot_mat[1,1] =torch.cos(theta)
        # print(coords)
        # print(rot_mat)
        rot_coords[i,:,:] = coords[i,...]@rot_mat
    return rot_coords

def translate_op(coord, dx, dy, sx, sy, device):
    trans_vec = torch.zeros((1,1,len(dx),coord.shape[-2]), dtype=torch.complex64, device=device)
    for i in range(len(dx)):
        trans_vec[0,0,i,:] = torch.exp(-1j*2*torch.tensor(np.pi)*(coord[i,:,1]*dx[i]/sx+coord[i,:,0]*dy[i]/sy))
    return trans_vec

def motion_forward(image, s_maps, coords, angles, dx, dy, device):
    #image: [Batch, Coil, H, W]
    #s_maps: [Batch, Coil, H, W]
    #coord: [TR, N_RO, 2]
    #angle: [TR]
    #dx: [TR]
    #dy: [TR]

    #apply sensitivity maps
    c_images = s_maps*image
    #apply roation and convert to kspace
    coords_rot = rot_batch(coords, angles, device)
    rot_ksp = nufft(input=c_images, coord=coords_rot, oversamp=2.00, width=4.0, device=device)

    #apply translation
    tra_vec = translate_op(coords_rot, dx, dy, image.shape[-1], image.shape[-2], device)

    motion_ksp = rot_ksp*tra_vec

    return motion_ksp

def motion_adjoint(ksp, s_maps, coords, angles, dx, dy, img_shape,device):
    #ksp: [Batch, Coil, H, W]
    #s_maps: [Batch, Coil, H, W]
    #coord: [TR, N_RO, 2]
    #angle: [TR]
    #dx: [TR]
    #dy: [TR]
    #img shape: give this s_maps.shape
    
    coords_rot = rot_batch(coords, angles, device)

    #apply translation
    tra_vec = translate_op(coords_rot, -1*dx, -1*dy, img_shape[-1], img_shape[-2], device)
    tra_ksp = ksp*tra_vec
    #apply roation and convert to image space
    rot_imgs = nufft_adjoint(input=tra_ksp, coord=coords_rot, out_shape = img_shape, oversamp=2.00, width=4.0, device=device)
    #apply conjugate sensitivity maps and sum over coils
    c_images = torch.conj(s_maps)*rot_imgs
    output = torch.sum(c_images, dim=1)[:,None,...]
    
    return output

def motion_normal(image, s_maps, coords, angles, dx, dy, device):
    ksp = motion_forward(image=image, s_maps=s_maps, coords=coords, angles=angles, dx=dx, dy=dy, device=device)
    img_out = motion_adjoint(ksp=ksp, s_maps=s_maps, coords=coords, angles=angles, dx=dx, dy=dy, img_shape=s_maps.shape,device=device)

    return img_out
    
import torch
import numpy as np
from motion_ops import motion_adjoint, motion_forward, motion_normal

"""Vector operations for use in calculating conjugate gradient descent."""
def itemize(x):
    """Converts a Tensor into a list of Python numbers.
    Args:
        x (Tensor): The tensor being itemized.
    Returns:
        Python list containing the itemized contents of the tensor.
    """

    if len(x.shape) < 1:
        x = x[None]
    if x.shape[0] > 1:
        return [xx.item() for xx in x]
    else:
        return x.item()

def dot(x1, x2):
    """Finds the dot product of two vectors.
    Args:
        x1 (Tensor): The first input vector.
        x2 (Tensor): The second input vector.
    Returns:
        The dot product of x1 and x2.
    """

    return torch.sum(x1*x2)

def dot_single(x):
    """Finds the dot product of a vector with itself
    Args:
        x (Tensor): The input vector.
    Returns:
        The dot product of x and x.
    """

    return dot(x, x)

def dot_batch(x1, x2):
    """Finds the dot product of two multidimensional Tensors, preserving the batch dimension.
    Args:
        x1 (Tensor): The first multidimensional Tensor.
        x2 (Tensor): The second multidimensional Tensor.
    Returns:
        The dot products along each dimension of x1 and x2.
    """

    batch = x1.shape[0]
    return torch.reshape(x1*x2, (batch, -1)).sum(1)


def dot_single_batch(x):
    """Finds the dot product of a multidimensional Tensors with itself, preserving the batch dimension.
    Args:
        x (Tensor): The multidimensional Tensor.
    Returns:
        The dot products along each non-batch dimension of x and x.
    """

    return dot_batch(x, x)

def zdot(x1, x2):
    """Finds the complex-valued dot product of two complex-valued vectors.
    Args:
        x1 (Tensor): The first input vector.
        x2 (Tensor): The second input vector.
    Returns:
        The dot product of x1 and x2, defined as sum(conj(x1) * x2)
    """

    return torch.sum(torch.conj(x1)*x2)

def zdot_single(x):
    """Finds the complex-valued dot product of a complex-valued vector with itself
    Args:
        x (Tensor): The input vector.
    Returns:
        The dot product of x and x., defined as sum(conj(x) * x)
    """

    return zdot(x, x)

def zdot_batch(x1, x2):
    """Finds the complex-valued dot product of two complex-valued multidimensional Tensors, preserving the batch dimension.
    Args:
        x1 (Tensor): The first multidimensional Tensor.
        x2 (Tensor): The second multidimensional Tensor.
    Returns:
        The dot products along each dimension of x1 and x2.
    """

    batch = x1.shape[0]
    return torch.reshape(torch.conj(x1)*x2, (batch, -1)).sum(1)


def zdot_single_batch(x):
    """Finds the complex-valued dot product of a multidimensional Tensors with itself, preserving the batch dimension.
    Args:
        x (Tensor): The multidimensional Tensor.
    Returns:
        The dot products along each non-batch dimension of x and x.
    """

    return zdot_batch(x, x)

def l2ball_proj_batch(x, eps):
    """ Performs a batch projection onto the L2 ball.
    Args:
        x (Tensor): The tensor to be projected.
        eps (Tensor): A tensor containing epsilon values for each dimension of the L2 ball.
    Returns:
        The projection of x onto the L2 ball.
    """

    #print('l2ball_proj_batch')
    reshape = (-1,) + (1,) * (len(x.shape) - 1)
    x = x.contiguous()
    q1 = torch.real(zdot_single_batch(x)).sqrt()
    #print(eps,q1)
    q1_clamp = torch.min(q1, eps)

    z = x * q1_clamp.reshape(reshape) / (1e-8 + q1.reshape(reshape))
    #q2 = torch.real(zdot_single_batch(z)).sqrt()
    #print(eps,q1,q2)
    return z

def conjgrad_motion_nufft(x, b, ktraj, maps, thetas, dx, dy, max_iter=10, l2lam=0., eps=1e-4, verbose=True, complex=True, device='cpu'):
    """Conjugate Gradient Algorithm applied to batches; assumes the first index is batch size.
    Args:
    x (Tensor): The initial input to the algorithm.
    b (Tensor): The residual vector
    Aop_fun (func): A function performing the normal equations, A.adjoint * A
    max_iter (int): Maximum number of times to run conjugate gradient descent.
    l2lam (float): The L2 lambda, or regularization parameter (must be positive).
    eps (float): Determines how small the residuals must be before termination…
    verbose (bool): If true, prints extra information to the console.
    complex (bool): If true, uses complex vector space
    Returns:
    	A tuple containing the output Tensor x and the number of iterations performed.
    """

    if complex:
        _dot_single_batch = lambda r: zdot_single_batch(r).real
        _dot_batch = lambda r, p: zdot_batch(r, p).real
    else:
        _dot_single_batch = dot_single_batch
        _dot_batch = dot_batch

    # explicitly remove r from the computational graph
    #r = b.new_zeros(b.shape, requires_grad=False, dtype=torch.cfloat)

    # the first calc of the residual may not be necessary in some cases...
#     # note that l2lam can be less than zero when training due to finite # of CG iterations
#     print('b shape:', b.shape)
#     print('x shape:', x.shape)
#     print('normal shape:',A_aug_normal(img=x, mask=masks, maps=maps, thetas=thetas).shape )
    r = b - (motion_normal(image=x, s_maps=maps, coords = ktraj, angles=thetas, dx = dx, dy = dy, device = device) + l2lam * x) #expects the shape [batch , H, W]
#     print('residual shape:',r.shape)
    p = r

    rsnot = _dot_single_batch(r)
    rsold = rsnot
    rsnew = rsnot
    # print('rsnot shape:',rsnot.shape)
    eps_squared = eps ** 2

    reshape = (-1,) + (1,) * (len(x.shape) - 1)

    num_iter = 0

    for i in range(max_iter):

        if verbose:
            print('{i}: {rsnew}'.format(i=i, rsnew=itemize(torch.sqrt(rsnew))))

        if rsnew.max() < eps_squared:
            if i == 0:
                # no iterations were run, so manually put x on the computation graph
                x.requires_grad_()
            break
#         print('p shape:', p.shape)
        Ap = motion_normal(image=p, s_maps=maps, coords = ktraj, angles=thetas, dx = dx, dy = dy, device = device) + l2lam * p
        pAp = _dot_batch(p, Ap)

        #print(utils.itemize(pAp))

        alpha = (rsold / pAp).reshape(reshape)
#         print('alpha shape:',alpha.shape)
        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = _dot_single_batch(r)

        beta = (rsnew / rsold).reshape(reshape)

        rsold = rsnew

        p = beta * p + r
        num_iter += 1


    if verbose:
        print('FINAL: {rsnew}'.format(rsnew=torch.sqrt(rsnew)))

    return x, num_iter

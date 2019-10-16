import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from scipy import linalg

def orthogonal_procrustes_torch(A, B):
    # Be clever with transposes, with the intention to save memory.
    A_device = A.device
    B_copy = B.clone().to(A_device)
    
    input = torch.transpose(torch.matmul(torch.transpose(B_copy,0,1),A),0,1)
    u, w, vt = torch.svd(input)
    #u, w, vt = torch.svd(torch.transpose(torch.matmul(torch.transpose(B,0,1),A),0,1))
    R = torch.matmul(u,torch.transpose(vt,0,1))
    scale = torch.sum(w)
    return R, scale

def criterion_procrustes(data1, data2):
    device = data1.device
    mtx1 = data1
    mtx2 = data2.clone().to(device)

    # translate all the data to the origin
    mtx3 =mtx1 - torch.mean(mtx1, 0)
    mtx4 =mtx2 - torch.mean(mtx2, 0)
    
    norm1 = torch.norm(mtx3)
    norm2 = torch.norm(mtx4)

    if norm1 == 0:
        norm1 = 1e-16
    if norm2 == 0:
        norm2 = 1e-16
    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx3 = mtx3/ norm1
    mtx4 = mtx4/ norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes_torch(mtx3, mtx4)
    mtx4 = torch.matmul(mtx4, torch.transpose(R,0,1)) * s
    
    # measure the dissimilarity between the two datasets
    disparity = torch.sum((mtx3 - mtx4)**2)

    return disparity

def manual_procrustes(data1, data2):
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)
    #print("manual norm")
    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)
    #print(norm1,norm2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s
    
    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity

def orthogonal_procrustes(A, B, check_finite=True):
    """
    Compute the matrix solution of the orthogonal Procrustes problem.
    Given matrices A and B of equal shape, find an orthogonal matrix R
    that most closely maps A to B [1]_.
    Note that unlike higher level Procrustes analyses of spatial data,
    this function only uses orthogonal transformations like rotations
    and reflections, and it does not use scaling or translation.
    Parameters
    """
    if check_finite:
        A = np.asarray_chkfinite(A)
        B = np.asarray_chkfinite(B)
    else:
        A = np.asanyarray(A)
        B = np.asanyarray(B)
    if A.ndim != 2:
        raise ValueError('expected ndim to be 2, but observed %s' % A.ndim)
    if A.shape != B.shape:
        raise ValueError('the shapes of A and B differ (%s vs %s)' % (
            A.shape, B.shape))
    # Be clever with transposes, with the intention to save memory.
    input = B.T.dot(A).T
    u, w, vt = linalg.svd(input)
    R = u.dot(vt)
    scale = w.sum()
    return R, scale
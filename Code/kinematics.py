# -*- coding: utf-8 -*-
import math
import torch

def config_to_matrix(p1, t1, xi):
    p1[p1 == 0] = 1e-10

    L = 15/100
    m0_0 = torch.cos(t1 + math.pi / 3) ** 2 + torch.sin(t1 + math.pi / 3) ** 2 - torch.cos(t1 + math.pi / 3) ** 2 * xi ** 2 * p1 ** 2 / 2 + torch.cos(t1 + math.pi / 3) ** 2 * xi ** 4 * p1 ** 4 / 24 - torch.cos(t1 + math.pi / 3) ** 2 * xi ** 6 * p1 ** 6 / 720 + torch.cos(t1 + math.pi / 3) ** 2 * xi ** 8 * p1 ** 8 / 40320 - torch.cos(t1 + math.pi / 3) ** 2 * xi ** 10 * p1 ** 10 / 3628800 + torch.cos(t1 + math.pi / 3) ** 2 * xi ** 12 * p1 ** 12 / 479001600 - torch.cos(t1 + math.pi / 3) ** 2 * xi ** 14 * p1 ** 14 / 87178291200
    m0_1 = torch.cos(t1 + math.pi / 3) * xi ** 2 * p1 ** 2 * torch.sin(t1 + math.pi / 3) / 2 - torch.cos(t1 + math.pi / 3) * xi ** 4 * p1 ** 4 * torch.sin(t1 + math.pi / 3) / 24 + torch.cos(t1 + math.pi / 3) * xi ** 6 * p1 ** 6 * torch.sin(t1 + math.pi / 3) / 720 - torch.cos(t1 + math.pi / 3) * xi ** 8 * p1 ** 8 * torch.sin(t1 + math.pi / 3) / 40320 + torch.cos(t1 + math.pi / 3) * xi ** 10 * p1 ** 10 * torch.sin(t1 + math.pi / 3) / 3628800 - torch.cos(t1 + math.pi / 3) * xi ** 12 * p1 ** 12 * torch.sin(t1 + math.pi / 3) / 479001600 + torch.cos(t1 + math.pi / 3) * xi ** 14 * p1 ** 14 * torch.sin(t1 + math.pi / 3) / 87178291200
    m0_2 = -torch.cos(t1 + math.pi / 3) * xi * p1 + torch.cos(t1 + math.pi / 3) * xi ** 3 * p1 ** 3 / 6 - torch.cos(t1 + math.pi / 3) * xi ** 5 * p1 ** 5 / 120 + torch.cos(t1 + math.pi / 3) * xi ** 7 * p1 ** 7 / 5040 - torch.cos(t1 + math.pi / 3) * xi ** 9 * p1 ** 9 / 362880 + torch.cos(t1 + math.pi / 3) * xi ** 11 * p1 ** 11 / 39916800 - torch.cos(t1 + math.pi / 3) * xi ** 13 * p1 ** 13 / 6227020800
    m0_3 = -torch.cos(t1 + math.pi / 3) * xi ** 2 * p1 * L / 2 + torch.cos(t1 + math.pi / 3) * xi ** 4 * p1 ** 3 * L / 24 - torch.cos(t1 + math.pi / 3) * xi ** 6 * p1 ** 5 * L / 720 + torch.cos(t1 + math.pi / 3) * xi ** 8 * p1 ** 7 * L / 40320 - torch.cos(t1 + math.pi / 3) * xi ** 10 * p1 ** 9 * L / 3628800 + torch.cos(t1 + math.pi / 3) * xi ** 12 * p1 ** 11 * L / 479001600 - torch.cos(t1 + math.pi / 3) * xi ** 14 * p1 ** 13 * L / 87178291200
    m1_0 = torch.cos(t1 + math.pi / 3) * xi ** 2 * p1 ** 2 * torch.sin(t1 + math.pi / 3) / 2 - torch.cos(t1 + math.pi / 3) * xi ** 4 * p1 ** 4 * torch.sin(t1 + math.pi / 3) / 24 + torch.cos(t1 + math.pi / 3) * xi ** 6 * p1 ** 6 * torch.sin(t1 + math.pi / 3) / 720 - torch.cos(t1 + math.pi / 3) * xi ** 8 * p1 ** 8 * torch.sin(t1 + math.pi / 3) / 40320 + torch.cos(t1 + math.pi / 3) * xi ** 10 * p1 ** 10 * torch.sin(t1 + math.pi / 3) / 3628800 - torch.cos(t1 + math.pi / 3) * xi ** 12 * p1 ** 12 * torch.sin(t1 + math.pi / 3) / 479001600 + torch.cos(t1 + math.pi / 3) * xi ** 14 * p1 ** 14 * torch.sin(t1 + math.pi / 3) / 87178291200
    m1_1 = torch.cos(t1 + math.pi / 3) ** 2 + torch.sin(t1 + math.pi / 3) ** 2 - torch.sin(t1 + math.pi / 3) ** 2 * xi ** 2 * p1 ** 2 / 2 + torch.sin(t1 + math.pi / 3) ** 2 * xi ** 4 * p1 ** 4 / 24 - torch.sin(t1 + math.pi / 3) ** 2 * xi ** 6 * p1 ** 6 / 720 + torch.sin(t1 + math.pi / 3) ** 2 * xi ** 8 * p1 ** 8 / 40320 - torch.sin(t1 + math.pi / 3) ** 2 * xi ** 10 * p1 ** 10 / 3628800 + torch.sin(t1 + math.pi / 3) ** 2 * xi ** 12 * p1 ** 12 / 479001600 - torch.sin(t1 + math.pi / 3) ** 2 * xi ** 14 * p1 ** 14 / 87178291200
    m1_2 = torch.sin(t1 + math.pi / 3) * xi * p1 - torch.sin(t1 + math.pi / 3) * xi ** 3 * p1 ** 3 / 6 + torch.sin(t1 + math.pi / 3) * xi ** 5 * p1 ** 5 / 120 - torch.sin(t1 + math.pi / 3) * xi ** 7 * p1 ** 7 / 5040 + torch.sin(t1 + math.pi / 3) * xi ** 9 * p1 ** 9 / 362880 - torch.sin(t1 + math.pi / 3) * xi ** 11 * p1 ** 11 / 39916800 + torch.sin(t1 + math.pi / 3) * xi ** 13 * p1 ** 13 / 6227020800
    m1_3 = torch.sin(t1 + math.pi / 3) * xi ** 2 * p1 * L / 2 - torch.sin(t1 + math.pi / 3) * xi ** 4 * p1 ** 3 * L / 24 + torch.sin(t1 + math.pi / 3) * xi ** 6 * p1 ** 5 * L / 720 - torch.sin(t1 + math.pi / 3) * xi ** 8 * p1 ** 7 * L / 40320 + torch.sin(t1 + math.pi / 3) * xi ** 10 * p1 ** 9 * L / 3628800 - torch.sin(t1 + math.pi / 3) * xi ** 12 * p1 ** 11 * L / 479001600 + torch.sin(t1 + math.pi / 3) * xi ** 14 * p1 ** 13 * L / 87178291200
    m2_0 = torch.cos(t1 + math.pi / 3) * xi * p1 - torch.cos(t1 + math.pi / 3) * xi ** 3 * p1 ** 3 / 6 + torch.cos(t1 + math.pi / 3) * xi ** 5 * p1 ** 5 / 120 - torch.cos(t1 + math.pi / 3) * xi ** 7 * p1 ** 7 / 5040 + torch.cos(t1 + math.pi / 3) * xi ** 9 * p1 ** 9 / 362880 - torch.cos(t1 + math.pi / 3) * xi ** 11 * p1 ** 11 / 39916800 + torch.cos(t1 + math.pi / 3) * xi ** 13 * p1 ** 13 / 6227020800
    m2_1 = torch.sin(t1 + math.pi / 3) * xi * p1 + torch.sin(t1 + math.pi / 3) * xi ** 3 * p1 ** 3 / 6 - torch.sin(t1 + math.pi / 3) * xi ** 5 * p1 ** 5 / 120 + torch.sin(t1 + math.pi / 3) * xi ** 7 * p1 ** 7 / 5040 - torch.sin(t1 + math.pi / 3) * xi ** 9 * p1 ** 9 / 362880 + torch.sin(t1 + math.pi / 3) * xi ** 11 * p1 ** 11 / 39916800 - torch.sin(t1 + math.pi / 3) * xi ** 13 * p1 ** 13 / 6227020800
    m2_2 = 1 - xi ** 2 * p1 ** 2 / 2 + xi ** 4 * p1 ** 4 / 24 - xi ** 6 * p1 ** 6 / 720 + xi ** 8 * p1 ** 8 / 40320 - xi ** 10 * p1 ** 10 / 3628800 + xi ** 12 * p1 ** 12 / 479001600 - xi ** 14 * p1 ** 14 / 87178291200
    m2_3 = xi * L - xi ** 3 * p1 ** 2 * L / 6 + xi ** 5 * p1 ** 4 * L / 120 - xi ** 7 * p1 ** 6 * L / 5040 + xi ** 9 * p1 ** 8 * L / 362880 - xi ** 11 * p1 ** 10 * L / 39916800 + xi ** 13 * p1 ** 12 * L / 6227020800
    m3_0 = torch.zeros([p1.shape[0], 1], device=p1.device, dtype=torch.float32)
    m3_1 = torch.zeros([p1.shape[0], 1], device=p1.device, dtype=torch.float32)
    m3_2 = torch.zeros([p1.shape[0], 1], device=p1.device, dtype=torch.float32)
    m3_3 = torch.ones([p1.shape[0], 1], device=p1.device, dtype=torch.float32)
    
    res = torch.cat((m0_0, m0_1, m0_2, m0_3, 
                          m1_0, m1_1, m1_2, m1_3,
                          m2_0, m2_1, m2_2, m2_3,
                          m3_0, m3_1, m3_2, m3_3),
                         dim=1).reshape((p1.shape[0], 4, 4))

    return res

def intermediate_config_to_coord(p, xi_list=None):
    '''
    Takes a list of configurations p in the shape [n, 2*k], where
    n is the number of configurations and k is the number of arms of
    the soft robot, and xi_list in the shape [n, 3], where xi_list[:,k]
    is the proportion up arm k to calculate, and returns the coordinates
    for all n points in an array of shape [n,3], where the result at [n,:]
    is the solution for configuration p[n,:] and xi_list[n,:]
    '''
    
    # Assume we mean tip position if xi_list is None
    if(xi_list is None):
        xi_list = torch.ones([p.shape[0], 
                            int(p.shape[1]/2)], 
                            device=p.device,
                            dtype=torch.float32)
    
    # Calcualte the matrices for each configuration
    m=config_to_matrix(p[:,0:1], 
                       p[:,1:2], 
                       xi_list[:,0:1])
    
    # Multiply matrices up the robot arm, one segment at a time
    i = 1
    while i < xi_list.shape[1] and xi_list[:,i].any():
        m = m @ config_to_matrix(
            p[:,2*i:2*i+1],
            p[:,2*i+1:2*i+2],
            xi_list[:,i:i+1])
        i += 1
        
    # Return only the position at the end (rightmost column of matrix)
    return m.permute(0, 2, 1)[:,3,0:3]

def random_configuration(num=1, segments=3, device="cpu"):
    '''
    Creates num random configurations of size [num, 2*segments]
    '''
    configs = torch.rand(size=[num, 2*segments], 
                         device=device,
                         dtype=torch.float32)
    configs[::2] *= math.pi
    configs[1::2] *= math.pi * 2
    return configs

def grid_configurations(phi_res = 128, theta_res = 128, segments=3, device="cpu"):
    '''
    Creates configurations in a uniform grid over phi and theta
    with resolution for each phi_res and theta_res
    '''
    
    p = torch.linspace(0, math.pi, phi_res, 
                       device=device, dtype=torch.float32)
    t = torch.linspace(0, math.pi*2, theta_res, device=device, 
                       dtype=torch.float32)
    
    c = []
    for _ in range(segments):
        c.append(p.clone())
        c.append(t.clone())
    c = torch.meshgrid(c, indexing="ij")
    c = torch.stack(c)
    return c
    
def flatten_grid_config(g):
    '''
    Take a grid of [k*2, m, n, m, n, ..., m, n]
    where k is the number of segments and m and n are the
    theta and phi resolutions, respectively, and 
    flattens all configurations into [(m*n)^k, k*2] for
    batch style forward kinematics
    '''
    
    return g.flatten(1).permute(1,0)

def flattened_grid_result_to_grid(r, m, n, k):
    '''
    Takes a result in the shape [(m*n)^k, R],
    where k is the number of segments and m and n are the
    theta and phi resolutions, respectively, and 
    R is the dimensionality of the result for each
    configuration, and returns a grid of size
    [R, m, n, m, n, ..., m, n]
    '''
    
    s = [r.shape[1]]
    for _ in range(k):
        s.append(m)
        s.append(n)
    
    return r.permute(1,0).reshape(s)
    
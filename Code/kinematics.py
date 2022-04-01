# -*- coding: utf-8 -*-
import numpy as np
import math
import os
import torch
import numba as nb

def config_to_matrix(p1,t1,xi):
    p1[p1 == 0] = 1e-10

    L=15/100
    m0_0 = np.cos(t1 + math.pi / 3) ** 2 + np.sin(t1 + math.pi / 3) ** 2 - np.cos(t1 + math.pi / 3) ** 2 * xi ** 2 * p1 ** 2 / 2 + np.cos(t1 + math.pi / 3) ** 2 * xi ** 4 * p1 ** 4 / 24 - np.cos(t1 + math.pi / 3) ** 2 * xi ** 6 * p1 ** 6 / 720 + np.cos(t1 + math.pi / 3) ** 2 * xi ** 8 * p1 ** 8 / 40320 - np.cos(t1 + math.pi / 3) ** 2 * xi ** 10 * p1 ** 10 / 3628800 + np.cos(t1 + math.pi / 3) ** 2 * xi ** 12 * p1 ** 12 / 479001600 - np.cos(t1 + math.pi / 3) ** 2 * xi ** 14 * p1 ** 14 / 87178291200
    m0_1 = np.cos(t1 + math.pi / 3) * xi ** 2 * p1 ** 2 * np.sin(t1 + math.pi / 3) / 2 - np.cos(t1 + math.pi / 3) * xi ** 4 * p1 ** 4 * np.sin(t1 + math.pi / 3) / 24 + np.cos(t1 + math.pi / 3) * xi ** 6 * p1 ** 6 * np.sin(t1 + math.pi / 3) / 720 - np.cos(t1 + math.pi / 3) * xi ** 8 * p1 ** 8 * np.sin(t1 + math.pi / 3) / 40320 + np.cos(t1 + math.pi / 3) * xi ** 10 * p1 ** 10 * np.sin(t1 + math.pi / 3) / 3628800 - np.cos(t1 + math.pi / 3) * xi ** 12 * p1 ** 12 * np.sin(t1 + math.pi / 3) / 479001600 + np.cos(t1 + math.pi / 3) * xi ** 14 * p1 ** 14 * np.sin(t1 + math.pi / 3) / 87178291200
    m0_2 = -np.cos(t1 + math.pi / 3) * xi * p1 + np.cos(t1 + math.pi / 3) * xi ** 3 * p1 ** 3 / 6 - np.cos(t1 + math.pi / 3) * xi ** 5 * p1 ** 5 / 120 + np.cos(t1 + math.pi / 3) * xi ** 7 * p1 ** 7 / 5040 - np.cos(t1 + math.pi / 3) * xi ** 9 * p1 ** 9 / 362880 + np.cos(t1 + math.pi / 3) * xi ** 11 * p1 ** 11 / 39916800 - np.cos(t1 + math.pi / 3) * xi ** 13 * p1 ** 13 / 6227020800
    m0_3 = -np.cos(t1 + math.pi / 3) * xi ** 2 * p1 * L / 2 + np.cos(t1 + math.pi / 3) * xi ** 4 * p1 ** 3 * L / 24 - np.cos(t1 + math.pi / 3) * xi ** 6 * p1 ** 5 * L / 720 + np.cos(t1 + math.pi / 3) * xi ** 8 * p1 ** 7 * L / 40320 - np.cos(t1 + math.pi / 3) * xi ** 10 * p1 ** 9 * L / 3628800 + np.cos(t1 + math.pi / 3) * xi ** 12 * p1 ** 11 * L / 479001600 - np.cos(t1 + math.pi / 3) * xi ** 14 * p1 ** 13 * L / 87178291200
    m1_0 = np.cos(t1 + math.pi / 3) * xi ** 2 * p1 ** 2 * np.sin(t1 + math.pi / 3) / 2 - np.cos(t1 + math.pi / 3) * xi ** 4 * p1 ** 4 * np.sin(t1 + math.pi / 3) / 24 + np.cos(t1 + math.pi / 3) * xi ** 6 * p1 ** 6 * np.sin(t1 + math.pi / 3) / 720 - np.cos(t1 + math.pi / 3) * xi ** 8 * p1 ** 8 * np.sin(t1 + math.pi / 3) / 40320 + np.cos(t1 + math.pi / 3) * xi ** 10 * p1 ** 10 * np.sin(t1 + math.pi / 3) / 3628800 - np.cos(t1 + math.pi / 3) * xi ** 12 * p1 ** 12 * np.sin(t1 + math.pi / 3) / 479001600 + np.cos(t1 + math.pi / 3) * xi ** 14 * p1 ** 14 * np.sin(t1 + math.pi / 3) / 87178291200
    m1_1 = np.cos(t1 + math.pi / 3) ** 2 + np.sin(t1 + math.pi / 3) ** 2 - np.sin(t1 + math.pi / 3) ** 2 * xi ** 2 * p1 ** 2 / 2 + np.sin(t1 + math.pi / 3) ** 2 * xi ** 4 * p1 ** 4 / 24 - np.sin(t1 + math.pi / 3) ** 2 * xi ** 6 * p1 ** 6 / 720 + np.sin(t1 + math.pi / 3) ** 2 * xi ** 8 * p1 ** 8 / 40320 - np.sin(t1 + math.pi / 3) ** 2 * xi ** 10 * p1 ** 10 / 3628800 + np.sin(t1 + math.pi / 3) ** 2 * xi ** 12 * p1 ** 12 / 479001600 - np.sin(t1 + math.pi / 3) ** 2 * xi ** 14 * p1 ** 14 / 87178291200
    m1_2 = np.sin(t1 + math.pi / 3) * xi * p1 - np.sin(t1 + math.pi / 3) * xi ** 3 * p1 ** 3 / 6 + np.sin(t1 + math.pi / 3) * xi ** 5 * p1 ** 5 / 120 - np.sin(t1 + math.pi / 3) * xi ** 7 * p1 ** 7 / 5040 + np.sin(t1 + math.pi / 3) * xi ** 9 * p1 ** 9 / 362880 - np.sin(t1 + math.pi / 3) * xi ** 11 * p1 ** 11 / 39916800 + np.sin(t1 + math.pi / 3) * xi ** 13 * p1 ** 13 / 6227020800
    m1_3 = np.sin(t1 + math.pi / 3) * xi ** 2 * p1 * L / 2 - np.sin(t1 + math.pi / 3) * xi ** 4 * p1 ** 3 * L / 24 + np.sin(t1 + math.pi / 3) * xi ** 6 * p1 ** 5 * L / 720 - np.sin(t1 + math.pi / 3) * xi ** 8 * p1 ** 7 * L / 40320 + np.sin(t1 + math.pi / 3) * xi ** 10 * p1 ** 9 * L / 3628800 - np.sin(t1 + math.pi / 3) * xi ** 12 * p1 ** 11 * L / 479001600 + np.sin(t1 + math.pi / 3) * xi ** 14 * p1 ** 13 * L / 87178291200
    m2_0 = np.cos(t1 + math.pi / 3) * xi * p1 - np.cos(t1 + math.pi / 3) * xi ** 3 * p1 ** 3 / 6 + np.cos(t1 + math.pi / 3) * xi ** 5 * p1 ** 5 / 120 - np.cos(t1 + math.pi / 3) * xi ** 7 * p1 ** 7 / 5040 + np.cos(t1 + math.pi / 3) * xi ** 9 * p1 ** 9 / 362880 - np.cos(t1 + math.pi / 3) * xi ** 11 * p1 ** 11 / 39916800 + np.cos(t1 + math.pi / 3) * xi ** 13 * p1 ** 13 / 6227020800
    m2_1 = np.sin(t1 + math.pi / 3) * xi * p1 + np.sin(t1 + math.pi / 3) * xi ** 3 * p1 ** 3 / 6 - np.sin(t1 + math.pi / 3) * xi ** 5 * p1 ** 5 / 120 + np.sin(t1 + math.pi / 3) * xi ** 7 * p1 ** 7 / 5040 - np.sin(t1 + math.pi / 3) * xi ** 9 * p1 ** 9 / 362880 + np.sin(t1 + math.pi / 3) * xi ** 11 * p1 ** 11 / 39916800 - np.sin(t1 + math.pi / 3) * xi ** 13 * p1 ** 13 / 6227020800
    m2_2 = 1 - xi ** 2 * p1 ** 2 / 2 + xi ** 4 * p1 ** 4 / 24 - xi ** 6 * p1 ** 6 / 720 + xi ** 8 * p1 ** 8 / 40320 - xi ** 10 * p1 ** 10 / 3628800 + xi ** 12 * p1 ** 12 / 479001600 - xi ** 14 * p1 ** 14 / 87178291200
    m2_3 = xi * L - xi ** 3 * p1 ** 2 * L / 6 + xi ** 5 * p1 ** 4 * L / 120 - xi ** 7 * p1 ** 6 * L / 5040 + xi ** 9 * p1 ** 8 * L / 362880 - xi ** 11 * p1 ** 10 * L / 39916800 + xi ** 13 * p1 ** 12 * L / 6227020800
    m3_0 = np.zeros([p1.shape[0], 1])
    m3_1 = np.zeros([p1.shape[0], 1])
    m3_2 = np.zeros([p1.shape[0], 1])
    m3_3 = np.ones([p1.shape[0], 1])
    
    res = np.concatenate((m0_0, m0_1, m0_2, m0_3, 
                          m1_0, m1_1, m1_2, m1_3,
                          m2_0, m2_1, m2_2, m2_3,
                          m3_0, m3_1, m3_2, m3_3),
                         axis=1).reshape((p1.shape[0], 4, 4))

    return res

def intermediate_config_to_coord(p,xi_list=None):
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
        xi_list = np.ones([p.shape[0], 3])
    
    # Calcualte the matrices for each configuration
    m=config_to_matrix(p[:,0:1], p[:,1:2], xi_list[:,0:1])
    
    # Multiply matrices up the robot arm, one segment at a time
    for i in range(xi_list.shape[1] - 2):
        m = m @ config_to_matrix(p[:,2*(i+1):2*(i+1)+1],
                                 p[:,2*(i+1)+1:2*(i+1)+2],
                                 xi_list[:,i+1:i+2])
    # Return only the position at the end (rightmost column of matrix)
    return m.transpose(0, 2, 1)[:,3,0:3]

def random_configuration(num=1, segments=3):
    '''
    Creates num random configurations of size [num, 2*segments]
    '''
    configs = np.random.random(size=[num, 2*segments])
    configs[::2] *= math.pi
    configs[1::2] *= math.pi * 2
    return configs
















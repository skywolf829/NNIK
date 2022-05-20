import torch
import torch.nn as nn
import math
from kinematics import grid_configurations, flatten_grid_config

class FK(nn.Module):  
    def __init__(self, n_segments=3):
        super().__init__()        
        self.n_segments = n_segments
        
    def config_to_matrix(self, p1, t1, xi):
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
        m3_0 = torch.zeros([p1.shape[0], 1], device=p1.device)
        m3_1 = torch.zeros([p1.shape[0], 1], device=p1.device)
        m3_2 = torch.zeros([p1.shape[0], 1], device=p1.device)
        m3_3 = torch.ones([p1.shape[0], 1], device=p1.device)
        
        res = torch.cat((m0_0, m0_1, m0_2, m0_3, 
                            m1_0, m1_1, m1_2, m1_3,
                            m2_0, m2_1, m2_2, m2_3,
                            m3_0, m3_1, m3_2, m3_3),
                            dim=1).reshape((p1.shape[0], 4, 4))

        return res

    def find_configs_for_from_grid_sampling(self, p, err = 0.01, max_GB = 8.0):
        '''
        Finds configurations on a grid near the target point p
        '''
        # need to work on the max GB limitation, rn you have to lowball
        # i.e. put max 6GB if you want to actually use 8
        res = ((max_GB*1024*1024*1024)/(4*(6+16+16+3)))**(1/(2*self.n_segments))
        res = int(res)
        g = grid_configurations(res, res, 
                                segments=self.n_segments, 
                                device=p.device)
        c = flatten_grid_config(g)
        print(f"Sampling {c.shape[0]} configuration forward kinematics")
        fk = self(c)
        d = (fk-p).norm(dim=1)
        return c[d<err,:], fk[d<err,:]
    
    def forward(self, x):
        '''
        Does forward kinematics for x
        '''
        # Assume we mean tip position if xi_list is None
        xi_list = torch.ones([x.shape[0], 
                                int(x.shape[1]/2)], 
                                device=x.device,
                                dtype=torch.float32)
        
        # Calcualte the matrices for each configuration
        m = self.config_to_matrix(x[:,0:1], x[:,1:2], xi_list[:,0:1])
        
        i = 1
        while i < xi_list.shape[1] and xi_list[:,i].any():
            m = m @ self.config_to_matrix(
                x[:,2*i:2*i+1],
                x[:,2*i+1:2*i+2],
                xi_list[:,i:i+1])
            i += 1
        # Return only the position at the end (rightmost column of matrix)
        return m.permute(0, 2, 1)[:,3,0:3]
    
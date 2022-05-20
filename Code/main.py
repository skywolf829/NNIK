from kinematics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import time
from kinematics_modules import *
from mpl_toolkits import mplot3d

def sample_grid_example(m, n, k, device="cpu"):
    '''
    An example of sampling a grid with phi resolution 
    of m, theta resolution of n, and k segments.
    Really only works for 1 segment, since it visualizes
    the result at the end in 2 dimensions (only one phi, theta)
    '''
    
    
    t0 = time.time()
    #c = random_configuration(num=1000000, segments=3, device=device)
    c = grid_configurations(phi_res = m, theta_res = n,
                            segments=k, device=device)
    
    t_elapsed = time.time() - t0
    print(f"Computation time for {(m*n)**k} points: {t_elapsed : 0.05f}s")
    c = flatten_grid_config(c)
    p = intermediate_config_to_coord(c)
    
    p = flattened_grid_result_to_grid(p, m, n, k)
    p = p.norm(dim=0)
    plt.pcolormesh(np.linspace(0, 2*math.pi, p.shape[1]),
                   np.linspace(0, math.pi, p.shape[0]),
                    p.clone().detach().cpu().numpy(),
                    cmap=cm.gray)
    plt.colorbar()
    plt.contour(np.linspace(0, 2*math.pi, p.shape[1]),
                   np.linspace(0, math.pi, p.shape[0]),
                    p.clone().detach().cpu().numpy(),
                    levels=10)
    #plt.colorbar()
    plt.xlabel("Theta")
    plt.ylabel("Phi")
    plt.title("Distance from arm base")
    plt.show()

def visualize_configurations(c, target_point=None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    n_lines_per_config = 5*3
    all_points = None
    ps = []
    for i in range(n_lines_per_config):
        p = (i / (n_lines_per_config-1))
        
        xi1 = min(1.0, p*3)
        xi2 = min(1.0, max(0.0, (p-(1/3))*3))
        xi3 = min(1.0, max(0.0, (p-(2/3))*3))
        
        xi = torch.tensor([xi1, xi2, xi3], 
                        dtype=torch.float32,
                        device=c.device).view(1, 3).repeat(c.shape[0], 1)
        
        points = intermediate_config_to_coord(c, xi)
        
        if(all_points is None):
            all_points = points
        else:
            all_points = torch.cat([all_points, points], dim=0)
        
        if(xi2 < 0.0):
            ps.append("red")
        elif(xi3 < 0.0):
            ps.append("green")
        else:
            ps.append("blue")
    print(all_points.shape)
    ax.scatter3D(all_points.cpu().numpy()[:,0], 
                all_points.cpu().numpy()[:,1],
                all_points.cpu().numpy()[:,2],
                alpha=1.0,
                c=ps)
    
    
    if(target_point is not None):
        ax.scatter3D(target_point[0].cpu().numpy(),
                     target_point[1].cpu().numpy(),
                     target_point[2].cpu().numpy(),
                     label="Target", color="red")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)       
    torch.random.manual_seed(0)    
    device = "cuda"
    
    # Example 1: 1 segment arm, sample on grid and
    # Visualize result
    # sample_grid_example(100, 100, 1, device)
    
    # Example 2: torch module for FK, find
    # solutions near target_point
    # using brute force grid search
    # Note: device="cuda" is much faster here
    # by nearly 3 orders of magnitude
    # cpu takes 1 minute, cuda takes 2.8 seconds
    target_point = [0.1, 0.03, 0.06]
    max_GB = 6
    
    t0 = time.time()
    err = 0.01
    model = FK(n_segments=3)
    c, fk = model.find_configs_for_from_grid_sampling(
        torch.tensor(target_point, device=device),
        err=err,
        max_GB=max_GB)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t0
    gb = torch.cuda.max_memory_allocated() / (1024*1024*1024)
    print(f"Found {c.shape[0]} solutions within {err} meters in {elapsed_time: 0.04f} seconds.")
    print(f"Memory use (if cuda): {gb}")
    
    visualize_configurations(c[0::100], 
                             target_point=torch.tensor(target_point, 
                                                       device=device))
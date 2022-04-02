from kinematics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import time
from kinematics_modules import *

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
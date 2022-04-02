from kinematics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import time

if __name__ == "__main__":
    
    t0 = time.time()
    np.random.seed(0)
    device="cpu"
    
    m = 128
    n = 128
    k = 1
    
    #c = random_configuration(num=1000000, segments=3, device=device)
    c = grid_configurations(phi_res = m, theta_res = n,
                            segments=k, device=device)
    c = flatten_grid_config(c)
    p = intermediate_config_to_coord(c)
    t_elapsed = time.time() - t0
    print(f"Time elapsed: {t_elapsed : 0.05f}s")
    
    p = flattened_grid_result_to_grid(p, m, n, k)
    p = p.norm(dim=0)
    plt.pcolormesh(np.linspace(0, 2*math.pi, p.shape[1]),
                   np.linspace(0, math.pi, p.shape[0]),
                    p.clone().detach().cpu().numpy(),
                    cmap=cm.gray)
    plt.colorbar()
    plt.xlabel("Theta")
    plt.ylabel("Phi")
    plt.show()
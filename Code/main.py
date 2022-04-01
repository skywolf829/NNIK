from kinematics import random_configuration, intermediate_config_to_coord
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    
    t0 = time.time()
    np.random.seed(0)
    
    for _ in range(1):
        r = random_configuration(10000)
        p = intermediate_config_to_coord(r)
    t_elapsed = time.time() - t0
    print(p.shape)
    print(t_elapsed)
    #print(p)
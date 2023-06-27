import numpy as np
import matplotlib.pyplot as plt
from milestone2 import collision_term, W, rho, equilibruim_distribution, mu
from milestone1 import direction, create_density_grid, animate, plot_density_grid, result_repo, streaming2D
from milestone4 import set_couette_boundary_fixed
import random

# def equilibrium_distribution_efficient(rho, U_vector) :

def set_poiseuille_boundary(probability_density_grid) :
    Rhin = np.full(probability_density_grid.shape[1:], 1.1)
    Rhout = np.full(probability_density_grid.shape[1:], .9)
    
    velocity_grid = mu(probability_density_grid)
    
    
    
    uN = np.repeat(velocity_grid[:, :, -2][:, :, np.newaxis], velocity_grid.shape[2], axis=2)
    u1 = np.repeat(velocity_grid[:, :, 1][:, :, np.newaxis], velocity_grid.shape[2], axis=2)
    
    
    
    feq_Rhin_uN = equilibruim_distribution(Rhin, uN)[:, :, -2] # XN
    feq_Rhout_u1 = equilibruim_distribution(Rhout, u1)[:, :, 1] #X1
    feq = equilibruim_distribution(rho(probability_density_grid), velocity_grid)



    fill1 = feq_Rhin_uN + (probability_density_grid[:, :, -2] - feq[:, :, -2]) # x N
    fill2 = feq_Rhout_u1 + (probability_density_grid[:, :, 1] - feq[:, :, 1]) # x 1



    probability_density_grid[[1, 5, 8], :, -1] = fill2[[1, 5, 8]] # x N + 1
    probability_density_grid[[1, 5, 8], :, 0] = fill1[[1, 5, 8]] # x 0
    
    return probability_density_grid

def create_poiseuille_grid() :
    probability_density_grid = create_density_grid(uniform=True, rand=False, x_shape=52, y_shape=302, uniform_value=.1)
    return probability_density_grid
    

def plot_velocity_profile() :
    framestop = 1000
    omega = 1.2
    tmax = 8000
    collision_function = lambda density_grid : collision_term(density_grid, omega)
    density_grid_plot = create_poiseuille_grid()

    plot_arr = np.zeros(300)
    
    plt.figure()
    for i in range(framestop) :
        print(i, '/', framestop, end='\r')
        density_grid_plot = streaming2D(density_grid_plot, direction, collision=collision_function, boundary=set_couette_boundary_fixed, pressure=set_poiseuille_boundary, test=True)
        plot_arr = mu(density_grid_plot)[0, :, 150]
        if i%100 == 0 and i != 0:
            plt.plot(plot_arr, label=str(i))
        
    plt.xlabel('Y axis')
    plt.ylabel('Velocity')
    plt.legend()
    plt.savefig(result_repo+'velocity_on_axis_poiseuille_flow.png')
    plt.close()
    
    
if __name__ == "__main__" :
    set_poiseuille_boundary(create_density_grid(uniform=True, rand=False, x_shape=302, y_shape=302))
    collision_function = lambda density_grid : collision_term(density_grid, .7)
    animate(file="poiseuille_boundary.mp4", velocity_active=True, collision=collision_function, create_grid=create_poiseuille_grid, frames=200, interval=100, boundary=set_couette_boundary_fixed, pressure=set_poiseuille_boundary)
    plot_velocity_profile()
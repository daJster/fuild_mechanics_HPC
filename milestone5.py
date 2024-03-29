import numpy as np
import matplotlib.pyplot as plt
from milestone2 import collision_term, W, rho, equilibruim_distribution, mu
from milestone1 import direction, create_density_grid, animate, result_repo, streaming2D
from milestone4 import set_couette_boundary_fixed
import random

def set_poiseuille_boundary(probability_density_grid) :

    cs = 1/3
    pout = .3
    pin = .03
    delta_p = pout - pin
    
    Rhin = np.full(probability_density_grid.shape[1:], (pout + delta_p)/cs)
    Rhout = np.full(probability_density_grid.shape[1:], pout/cs)
    
    velocity_grid = mu(probability_density_grid)
    
    
    
    uN = np.repeat(velocity_grid[:, :, -2][:, :, np.newaxis], velocity_grid.shape[2], axis=2)
    u1 = np.repeat(velocity_grid[:, :, 1][:, :, np.newaxis], velocity_grid.shape[2], axis=2)
    
    
    
    feq_Rhin_uN = equilibruim_distribution(Rhin, uN)[:, :, -2] # X N
    feq_Rhout_u1 = equilibruim_distribution(Rhout, u1)[:, :, 1] # X 1
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
    framestop = 1300
    omega = 1.2
    viscosity = 0.5
    pout = .3
    pin = .03
    delta_p = pout - pin
    collision_function = lambda density_grid : collision_term(density_grid, omega)
    density_grid_plot = create_poiseuille_grid()

    plot_arr = np.zeros(300)
    
    plt.figure()
    for i in range(framestop) :
        print(i, '/', framestop, end='\r')
        density_grid_plot = streaming2D(density_grid_plot, direction, collision=collision_function, boundary=set_couette_boundary_fixed, pressure=set_poiseuille_boundary, test=True)
        plot_arr = mu(density_grid_plot)[0, :, 150]
        if i%100 == 0 and i != 0 and i > 999:
            plt.plot(plot_arr, label=str(i))
    
    analytical_solution = [-delta_p*y*(y - density_grid_plot.shape[1])/(2*viscosity*rho(density_grid_plot).mean(axis=0).sum()) for y in range(density_grid_plot.shape[1])]
    plt.plot(analytical_solution, label='theory')
    plt.xlabel('Y axis')
    plt.ylabel('Velocity')
    plt.legend()
    plt.savefig(result_repo+'velocity_on_axis_poiseuille_flow.png')
    plt.close()
    

def plot_velocity_profile_heatmap(framestop = 1000) :
    omega = 1.2
    collision_function = lambda density_grid : collision_term(density_grid, omega)
    density_grid_plot = create_poiseuille_grid()

    plot_arr = np.zeros(300)
    
    plt.figure()
    
    for i in range(framestop + 1) :
        print(i, '/', framestop, end='\r')
        density_grid_plot = streaming2D(density_grid_plot, direction, collision=collision_function, boundary=set_couette_boundary_fixed, pressure=set_poiseuille_boundary, test=True)
        if i == framestop :
            plot_arr = mu(density_grid_plot)[0, :, :]
    
    plt.imshow(plot_arr, cmap='viridis')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig(result_repo+'velocity_profile_poiseuille_flow_' + str(framestop) + '.png')
    plt.close()
    
    
if __name__ == "__main__" :
    collision_function = lambda density_grid : collision_term(density_grid, 1.2)
    
    animate(file="poiseuille_boundary.mp4", velocity_active=True, collision=collision_function, create_grid=create_poiseuille_grid, frames=200, interval=100, boundary=set_couette_boundary_fixed, pressure=set_poiseuille_boundary)
    
    plot_velocity_profile()
    plot_velocity_profile_heatmap(framestop=1)
    plot_velocity_profile_heatmap(framestop=500)
    plot_velocity_profile_heatmap(framestop=1000)
    
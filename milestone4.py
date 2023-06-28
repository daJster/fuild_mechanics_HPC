import numpy as np
import matplotlib.pyplot as plt
from milestone2 import collision_term, W, rho, equilibruim_distribution, mu
from milestone1 import direction, create_density_grid, animate, plot_density_grid, result_repo, streaming2D
import random

top_channels = [5, 2, 6]
bottom_channels = [7, 4, 8]
right_channels = [1, 5, 8]
left_channels = [3, 6, 7]
    
def create_couette_grid_fixed() :
    density_grid_animate = create_density_grid(y_shape=302, x_shape=302, uniform=True, rand=False)
    offset_x = 60
    offset_y = 70
    # filling the center of the grid
    for x in range(100) :
        for y in range(20) :
            density_grid_animate[:, y + offset_y, x + offset_x] = [density_grid_animate[k, y + offset_y, x + offset_x] + .2 for k in range(9)]

    return density_grid_animate


def create_couette_grid() :
    density_grid_animate = create_density_grid(y_shape=302, x_shape=100, uniform=True, rand=False)
    return density_grid_animate



def set_couette_boundary_fixed(probability_density_grid) :
    for x in range(probability_density_grid.shape[2]) :
        cell_top = np.copy(probability_density_grid[:, -1, x])
        cell_bottom = np.copy(probability_density_grid[:, 0, x])
        # top boundary
        probability_density_grid[bottom_channels, -1, x] = cell_top[top_channels]
        # bottom boundary
        probability_density_grid[top_channels, 0, x] = cell_bottom[bottom_channels]

    return probability_density_grid

def set_couette_boundary(probability_density_grid) :
    wall_speed = [.1, 0]
    for x in range(probability_density_grid.shape[2]) :
        cell_top = np.copy(probability_density_grid[:, -1, x])
        cell_bottom = np.copy(probability_density_grid[:, 0, x])
        proba_cell_top = cell_top.sum()
        # new moving densities
        new_cell_top_8 = cell_top[6] - 2*W[6]*proba_cell_top*np.dot(direction[:, 6], wall_speed)/np.dot(direction[:, 6], direction[:, 6])
        new_cell_top_4 = cell_top[2] - 2*W[2]*proba_cell_top*np.dot(direction[:, 2], wall_speed)/np.dot(direction[:, 2], direction[:, 2])
        new_cell_top_7 = cell_top[5] - 2*W[5]*proba_cell_top*np.dot(direction[:, 5], wall_speed)/np.dot(direction[:, 5], direction[:, 5])
        # top boundary
        probability_density_grid[bottom_channels, -1, x] = [new_cell_top_7, new_cell_top_4, new_cell_top_8]
        # bottom boundary
        probability_density_grid[top_channels, 0, x] = cell_bottom[bottom_channels]

    return probability_density_grid


def plot_couette_moving_boundary() :
    omega = .6
    tmax = 8000
    collision_function = lambda density_grid : collision_term(density_grid, omega)
    density_grid_plot = create_couette_grid()

    plot_arr = np.zeros(300)
    
    plt.figure()
    for i in range(tmax) :
        print(i, '/', tmax, end='\r')
        density_grid_plot = streaming2D(density_grid_plot, direction, collision=collision_function, boundary=set_couette_boundary, test=True)
        plot_arr = mu(density_grid_plot)[0 , :, 150]
        if (i%1000 == 0 and i != 0) or i == 200 :
            plt.plot(plot_arr, label=str(i))
        
    plt.plot(np.linspace(plot_arr[0], plot_arr[-1], 100), label='inf')
    plt.xlabel('Y axis')
    plt.ylabel('Velocity')
    plt.legend()
    plt.savefig(result_repo+'velocity_on_axis_couette_flow_moving_east.png')
    plt.close()
    
    
def plot_im_velocity(framestop) :
    omega = .6
    collision_function = lambda density_grid : collision_term(density_grid, omega)
    density_grid_plot = create_couette_grid()

    plot_arr = np.zeros(300)
    
    plt.figure()
    for i in range(framestop) :
        print(i, '/', framestop, end='\r')
        density_grid_plot = streaming2D(density_grid_plot, direction, collision=collision_function, boundary=set_couette_boundary, test=True)
        if i == framestop-1 :
            plot_arr = mu(density_grid_plot)[0 , :, :]
            
            
    plt.imshow(plot_arr, cmap='viridis')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig(result_repo+'velocity_img_framestop_'+ str(framestop) +'.png')
    plt.close()
    
    
if __name__ == "__main__" :
    plot_density_grid(create_couette_grid_fixed(), file="couette_grid_fixed.png")
    collision_function = lambda density_grid : collision_term(density_grid, .7)
    animate(file="couette_boundary_fixed.mp4", collision=collision_function, create_grid=create_couette_grid_fixed, frames=200, interval=100, boundary=set_couette_boundary_fixed)
    animate(file="couette_boundary.mp4", collision=collision_function, create_grid=create_couette_grid, frames=200, interval=100, boundary=set_couette_boundary)
    plot_couette_moving_boundary()
    plot_im_velocity(4000)
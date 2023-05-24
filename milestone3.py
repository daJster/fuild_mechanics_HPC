import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from milestone2 import collision_term, equilibruim_distribution
from milestone1 import streaming2D, result_repo, direction
import random



#---------------------------------------------------------------------------------------------
def create_sinus_density(x_shape=300, y_shape=300, epsilon=.01, rho0=1) :
    velocity_grid = np.zeros((2, x_shape, y_shape))
    Lx = x_shape  # Length of the x-axis
    x = np.arange(Lx)
    
    density_variation = rho0 + epsilon * np.sin(2 * np.pi * x / Lx) # array 300
    reshaped_array = density_variation.reshape(300, 1)
    rho_grid = np.tile(reshaped_array, (1, 300))
    return rho_grid, velocity_grid
            
def create_sinus_velocity(x_shape=300, y_shape=300, epsilon=.1) :
    rho_grid = np.full((x_shape, y_shape), 1)
    Ly = y_shape  # Length of the y-axis
    y = np.arange(y_shape)
    velocity_variation = epsilon * np.sin(2 * np.pi * y / Ly)
    velocity_grid = np.expand_dims(velocity_variation, axis=(0, 2))
    velocity_grid = np.repeat(velocity_grid, repeats=300, axis=2)
    velocity_grid = np.repeat(velocity_grid, repeats=2, axis=0)
    return rho_grid, velocity_grid
#---------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------
def animate(file=None, frames=200, interval=100, collision=None, cmap="Blues", create_grid=None) :
    rho_grid_animate, velocity_grid_animate = create_grid()
    density_grid_animate = equilibruim_distribution(rho_grid_animate, velocity_grid_animate)
    
    total_density = density_grid_animate.sum(axis=0)
    fig = plt.figure()
    im = plt.imshow(total_density, animated=True, cmap=cmap)
    i = 0 
    def updatefig(frame) :
        nonlocal density_grid_animate, i
        i += 1
        density_grid_animate = streaming2D(density_grid_animate, direction, collision=collision, test=True)
        frame = density_grid_animate.sum(axis=0)
        im.set_array(frame)
        print('frame :', i, "/", frames, end='\r')
        return im,

    ani = FuncAnimation(fig, updatefig, blit=True, frames=frames, interval=interval)
    cbar = fig.colorbar(im)
    ani.save(result_repo+file, writer='ffmpeg')

def plot_rho(rho, file=None) :
    plt.imshow(rho, cmap='Blues')
    plt.colorbar()
    plt.savefig(result_repo+file)
#---------------------------------------------------------------------------------------------

def plot_density_on_time() :
    return

def plot_velocity_on_time() :
    return
   
if __name__ == "__main__" :
         
    omega = 1
    collision_function = lambda density_grid : collision_term(density_grid, omega)
    plot_rho(create_sinus_density(epsilon=.1)[0], file="sinus_density.png")
    #animate(file="sinus_density.mp4", collision=collision_function, create_grid=create_sinus_density, frames=200, interval=100)
    #animate(file="sinus_velocity.mp4", collision=collision_function, create_grid=create_sinus_velocity, frames=400, interval=100)
    plot_density_on_time()
    plot_velocity_on_time()
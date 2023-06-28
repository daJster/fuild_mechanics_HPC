import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

result_repo = 'result_xy/'

np.random.seed(1234)
random.seed(12)

direction = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                      [0, 0, 1, 0, -1, 1, 1, -1, -1]])

def probability_density(f) :
    return f.sum(axis=0)

def velocity(f) :
    return np.einsum("ij,jkl->ikl", direction, f)/probability_density(f)

def create_density_grid(y_shape=15, x_shape=10, v_shape=9, rand=.2, uniform=False, uniform_value=(1/9)) :
    if uniform : 
        density_grid = np.full((v_shape, x_shape, y_shape), uniform_value)
    else :
        density_grid = np.random.rand(v_shape, x_shape, y_shape)
        density_grid /= density_grid.sum(axis=0)
    # create holes in the matrix
    if rand :
        for y in range(y_shape) :
            for x in range(x_shape) :
                if np.random.random() > rand :
                    density_grid[:, x, y] = np.zeros(v_shape)
                    
    return density_grid

def plot_density_grid(density_grid, file=None) :
    plt.figure()
    total_density = density_grid.sum(axis=0)
    plt.imshow(total_density, cmap='Blues')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig(result_repo+file)
    plt.close()


#---------------------------------------------------------------------------------------------
def streaming2D(arr, direction, test=False, collision=False, boundary=False, pressure=False) :
    acc = np.copy(arr)
    
    if pressure :
        arr = pressure(arr)
    
    for i in range(direction.shape[1]) :
        arr[i, :, :] = np.roll(arr[i, :, :], shift=(direction[1, i], direction[0, i]), axis=(0, 1))
  
    if boundary :
        arr = boundary(arr)      
    if collision :
       arr = collision(arr)
    if test :
        try :
            assert np.isclose(arr.sum(), acc.sum(), rtol=10**(-2)) # testing if the mass is conserved after one stream
        except :
            print('AssertionError : the mass is not conserved during the stream.\n probability before : ', np.sum(arr), ' probablity after :',  np.sum(acc))
            raise AssertionError
    return arr
#---------------------------------------------------------------------------------------------

def animate(file=None, frames=200, interval=100, velocity_active=False, collision=None, pressure=False, cmap="Blues", create_grid=(lambda : create_density_grid(x_shape=30, y_shape=30)), boundary=False) :
    density_grid_animate = create_grid()
    total_density = density_grid_animate.sum(axis=0)
    fig = plt.figure()
    if velocity_active :
        if velocity_active == 'norm' :
            im  = plt.imshow(np.linalg.norm(velocity(density_grid_animate), axis=0)[:, :], animated=True, cmap=cmap)
        else :
            im = plt.imshow(velocity(density_grid_animate)[0, :, :], animated=True, cmap=cmap)
    else :
        im = plt.imshow(total_density, animated=True, cmap=cmap)
    i = 0
    
    def updatefig(frame) :
        nonlocal density_grid_animate, i
        i += 1
        density_grid_animate= streaming2D(density_grid_animate, direction, collision=collision, test=True, boundary=boundary, pressure=pressure)
        if velocity_active :
            if velocity_active == 'norm' :
                frame  = np.linalg.norm(velocity(density_grid_animate), axis=0)[:, :]
            else  :
                frame  = velocity(density_grid_animate)[0, :, :]
        else :
            frame = density_grid_animate.sum(axis=0)
        im.set_array(frame)
        print('frame :', i, "/", frames, end='\r')
        return im,

    ani = FuncAnimation(fig, updatefig, blit=True, frames=frames, interval=interval)
    cbar = fig.colorbar(im)
    plt.gca().invert_yaxis()
    ani.save(result_repo+file, writer='ffmpeg')



if __name__ == "__main__":
    density_grid = create_density_grid(x_shape=30, y_shape=30)
    plot_density_grid(density_grid, file='density_grid.png')
    density_grid = streaming2D(density_grid,direction, test=True)
    plot_density_grid(density_grid, file='density_grid_plusone.png')
    animate(file='density_animation.mp4', frames=200)
    
    
    test_directions_grid = create_density_grid(x_shape=5, y_shape=5, uniform=True, rand=False)
    for i in range(9) :
        test_directions_grid[i, 2, 2] = i

    test_directions_grid = streaming2D(test_directions_grid, direction, test=True)
    plot_density_grid(test_directions_grid, file='test_directions.png')
    
    
    density_grid_animate = create_density_grid(x_shape=300, y_shape=300, uniform=True, rand=False)
    offset_x = 10
    offset_y = 0 
    # filling the center of the grid
    for x in range(50) :
        for y in range(50) :
            density_grid_animate[:, y + offset_y, x + offset_x] = [1 for k in range(9)]
            

    plot_density_grid(density_grid_animate, file='testing_xy.png')
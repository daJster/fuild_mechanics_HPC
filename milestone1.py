import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

result_repo = 'result/'

np.random.seed(1234)
random.seed(12)

direction = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                      [0, 0, 1, 0, -1, 1, 1, -1, -1]])

def probability_density(f) :
    return f.sum(axis=0)

def velocity(f) :
    return np.einsum("ij,jkl->ikl", direction, f)/probability_density(f)

def create_density_grid(x_shape=15, y_shape=10, v_shape=9, rand=.2, uniform=False, uniform_value=(1/9)) :
    if uniform : 
        density_grid = np.full((v_shape, x_shape, y_shape), uniform_value)
    else :
        density_grid = np.random.rand(v_shape, x_shape, y_shape)
        density_grid /= density_grid.sum(axis=0)
    # create holes in the matrix
    if rand :
        for x in range(x_shape) :
            for y in range(y_shape) :
                if np.random.random() > rand :
                    density_grid[:, x, y] = np.zeros(v_shape)
    return density_grid

def plot_density_grid(density_grid, file=None) :
    total_density = density_grid.sum(axis=0)
    plt.imshow(total_density, cmap='Blues')
    plt.colorbar()
    plt.savefig(result_repo+file)


#---------------------------------------------------------------------------------------------
def streaming2D(arr, direction, test=False, collision=False) :
    acc = np.zeros_like(arr)
    for i in range(direction.shape[1]) :
        acc[i, :, :] = np.roll(arr[i, :, :], shift=(direction[0][i], direction[1][i]), axis=(0, 1))
    if collision :
        acc = collision(acc)
    if test :
        try :
            assert np.isclose(arr.sum(), acc.sum(), rtol=.1) # testing if the mass is conserved after one stream
        except :
            print('AssertionError : the mass is not conserved during the stream.\n probability before : ', np.sum(arr), ' probablity after :',  np.sum(acc))
            raise AssertionError
    # print(arr.sum(), "/" , acc.sum())
    return acc
#---------------------------------------------------------------------------------------------

def animate(file=None, frames=200, interval=100, collision=None, cmap="Blues", create_grid=(lambda : create_density_grid(x_shape=30, y_shape=30))) :
    density_grid_animate = create_grid()
    total_density = density_grid_animate.sum(axis=0)
    fig = plt.figure()
    im = plt.imshow(total_density, animated=True, cmap=cmap)
    i = 0 
    def updatefig(frame) :
        nonlocal density_grid_animate, i
        #print(velocity_grid_animate.sum())
        i += 1
        density_grid_animate= streaming2D(density_grid_animate, direction, collision=collision, test=True)
        frame = density_grid_animate.sum(axis=0)
        im.set_array(frame)
        print('frame :', i, "/", frames, end='\r')
        return im,

    ani = FuncAnimation(fig, updatefig, blit=True, frames=frames, interval=interval)
    cbar = fig.colorbar(im)
    ani.save(result_repo+file, writer='ffmpeg')



if __name__ == "__main__":
    density_grid = create_density_grid(x_shape=30, y_shape=30)
    plot_density_grid(density_grid, file='density_grid.png')
    streaming2D(density_grid,direction, test=True)
    animate(file='density_animation.mp4', frames=200)
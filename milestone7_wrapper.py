import subprocess
import os
import numpy as np
from milestone1 import result_repo
import time
import matplotlib.pyplot as plt

def calculate_execution_time(script_path, arguments_set):
    results = []
    for arguments in arguments_set :
        results_inner = []
        for arg in arguments :
            start_time = time.time()

            # Prepare the command to execute the script with the current argument
            command = 'mpirun -np '+str(arg['np'])+' --use-hwthread-cpus python3.8 ' + script_path
            for key, value in arg.items():
                if key == 'np' :
                    continue
                command += ' -' + key + ' '
                if isinstance(value, list) :
                    for v in value : 
                        command += str(v) + ' '           
                else :
                    command += str(value) + ' '
            try:
                os.system(command)

                # Calculate the execution time
                execution_time = time.time() - start_time

                result = {
                    'argument': arg,
                    'execution_time': execution_time
                }
                results_inner.append(result)
                print(f"Argument: {result['argument']}")
                print(f"Execution time: {result['execution_time']} seconds")
                print('-' * 30)
                print('\n')
                
            except subprocess.CalledProcessError as e:
                print(f"Error executing script with argument {arg}: {e}")
        results.append(results_inner)
    return results

def plot_results_execution_time(results_set) :
    plt.figure()
    for results in results_set :
        plot_arr_mlups = []
        plot_arr_nodes = []
        for result in results :
            plot_arr_mlups.append(result['argument']['f']*result['argument']['gs'][0]*result['argument']['gs'][1]/(result['execution_time']*10**6))
            plot_arr_nodes.append(result['argument']['np'])
        plt.plot(plot_arr_nodes, plot_arr_mlups, label=str(result['argument']['gs'][0]), marker='s')
        
    plt.legend('LB size')
    plt.xlabel('number of steps for 4 processors')
    plt.ylabel('MLUPS')
    plt.yscale('log')
    plt.savefig(result_repo+'plot_MLUPS.png')
    plt.close()

def plot_full_velocity() :
    velocity_x_path = result_repo+"velocity_x.npy"
    velocity_y_path = result_repo+"velocity_y.npy"
    
    velocity_x = np.load(velocity_x_path)
    velocity_y = np.load(velocity_y_path)
    
    x, y = np.arange(0, velocity_x.shape[1]), np.arange(0, velocity_x.shape[0])
    x, y = np.meshgrid(x, y)

    # Create a new figure and axis
    fig, ax = plt.subplots()
    ax.streamplot(x, y, velocity_x, velocity_y, density=1)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    plt.savefig(result_repo+'velocity_streamplot_sliding_lid_parallel.png')
    plt.close()


def generate_arguments(grid_sizes, omega, frames, nodes=120) :
    result = []
    for size in grid_sizes :
        i = 2
        acc = []
        while i**2 < nodes :
            acc.append(
                {'np' : i**2,
                 'w' : omega,
                 'gs' : [size, size],
                 'ns' : [i, i],
                 'f' : frames}
            )
            i += 1 
        result.append(acc)
    return result

if __name__ == "__main__" :
    script_path = 'milestone7.py'
    grid_sizes = [500, 1000, 5000]
    omega = 1.6
    frames = 400
    arguments = generate_arguments(grid_sizes, omega, frames, nodes=120)
    
    results = calculate_execution_time(script_path, arguments)

    plot_results_execution_time(results)
    
    plot_full_velocity()

    
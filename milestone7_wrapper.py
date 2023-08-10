import subprocess
import os
import numpy as np
from milestone1 import result_repo
import argparse
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re

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

def read_output_parallel(filename) :
    # Read the file
    with open(filename, 'r') as file:
        data = file.read()

    # Extract relevant information using regular expressions
    pattern = r"Argument: {'np': (\d+), 'w': ([\d.]+), 'gs': \[(\d+), (\d+)\], 'ns': \[(\d+), (\d+)\], 'f': (\d+)}\nExecution time: ([\d.]+) seconds\n------------------------------"
    matches = re.findall(pattern, data)

    # Create a list of dictionaries to store the extracted data
    extracted_data = []
    for match in matches:
        extracted_data.append({
            'np': int(match[0]),
            'w': float(match[1]),
            'gs_x': int(match[2]),
            'gs_y': int(match[3]),
            'ns_x': int(match[4]),
            'ns_y': int(match[5]),
            'f': int(match[6]),
            'execution_time': float(match[7])
        })

    # Create a pandas DataFrame from the extracted data
    df = pd.DataFrame(extracted_data)
    
    return df


def plot_results_execution_time(filename) :
    df = read_output_parallel(filename)
    df["grid_size"] = df['gs_x']*df['gs_y']
    df["mlups"] = round(df['grid_size']*df['f']/(df['execution_time']*10**6))
    print("results :\n", df)
    plt.figure()
    sns.set_theme()
    sns.lineplot(x="np", y="mlups", hue="grid_size", marker='o', markers=True, palette="Set2", markersize=5, data=df)
    plt.legend(title="grid size")
    plt.xlabel('number of processors')
    plt.ylabel('MLUPS (million lattice units per second)')
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
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-run", "--run", action='store_true', default=False)
    parser.add_argument('-plot', '--plot', action='store_true', default=False)
    args = parser.parse_args()
    
    if args.run :
        script_path = 'milestone7.py'
        grid_sizes = [500, 1000, 5000]
        omega = 1.6
        frames = 400
        arguments = generate_arguments(grid_sizes, omega, frames, nodes=120)
        
        calculate_execution_time(script_path, arguments)

    if args.plot :
        output_parallel_file = "./output_parallel.txt" 
        plot_results_execution_time(output_parallel_file)
        # plot_full_velocity()

    
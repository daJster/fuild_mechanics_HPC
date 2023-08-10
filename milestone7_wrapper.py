#!/bin/python3.8

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

def calculate_execution_time(script_command, arguments_set):
    """
    Execute a script with different sets of arguments and measure execution times.

    Parameters:
    - script_path (str): Path to the script to be executed.
    - arguments_set (list of list of dict): List of different sets of arguments for the script.

    Returns:
    list of list of dict: A nested list containing execution time results for each argument set.
    """
    results = []
    # Iterate through different sets of arguments
    for arguments in arguments_set:
        results_inner = []

        # Iterate through individual arguments in the set
        for arg in arguments:
            start_time = time.time()

            # Prepare the command to execute the script with the current argument
            command = script_command
            for key, value in arg.items():
                command += ' -' + key + ' '
                if isinstance(value, list):
                    for v in value:
                        command += str(v) + ' '
                else:
                    command += str(value) + ' '

            try:
                # Execute the command
                os.system(command)

                # Calculate the execution time
                execution_time = time.time() - start_time

                # Create a result dictionary
                result = {
                    'argument': arg,
                    'execution_time': execution_time
                }
                results_inner.append(result)

                # Print the result information
                print(f"Argument: {result['argument']}")
                print(f"Execution time: {result['execution_time']} seconds")
                print('-' * 30)
                print('\n')
            except subprocess.CalledProcessError as e:
                print(f"Error executing script with argument {arg}: {e}")

        results.append(results_inner)

    return results

def read_output_parallel(filename):
    """
    Read data from a text file, extract relevant information using regular expressions,
    and create a pandas DataFrame to store the extracted data.

    Parameters:
    - filename (str): Path to the output file containing execution time results.

    Returns:
    pandas.DataFrame: A DataFrame containing the extracted execution time results.
    """
    # Read the content of the file
    with open(filename, 'r') as file:
        data = file.read()

    # Define a regular expression pattern to match the data
    pattern = r"Argument: {'np': (\d+), 'w': ([\d.]+), 'gs': \[(\d+), (\d+)\], 'ns': \[(\d+), (\d+)\], 'f': (\d+)}\nExecution time: ([\d.]+) seconds\n------------------------------"

    # Find all matches in the data using the regular expression pattern
    matches = re.findall(pattern, data)

    # Initialize an empty list to store the extracted data as dictionaries
    extracted_data = []
    
    # Iterate through each match and extract data into a dictionary
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


def plot_results_execution_time(filename):
    """
    Read data from an output file, process it, and create a line plot of MLUPS vs. number of processors.

    Parameters:
    - filename (str): Path to the output file containing execution time results.

    Returns:
    None
    """
    # Read the output file and load it into a DataFrame
    df = read_output_parallel(filename)
    
    # Calculate grid size and MLUPS
    df["grid_size"] = df['gs_x'] * df['gs_y']
    df["mlups"] = round(df['grid_size'] * df['f'] / (df['execution_time'] * 10**6))
    
    # Print the processed results
    print("results :\n", df)
    
    # Create a line plot using seaborn
    plt.figure()
    sns.set_theme()
    sns.lineplot(x="np", y="mlups", hue="grid_size", marker='o', markers=True, palette="Set2", markersize=5, data=df)
    
    # Add legend and labels
    plt.legend(title="grid size")
    plt.xlabel('number of processors')
    plt.ylabel('MLUPS (million lattice units per second)')
    
    # Save the plot as an image
    plt.savefig(result_repo + 'plot_MLUPS.png')
    
    # Close the plot
    plt.close()

def plot_full_velocity():
    """
    Load velocity data from files and create a streamplot visualization.

    Parameters:
    None

    Returns:
    None
    """
    # Paths to velocity data files
    velocity_x_path = result_repo + "velocity_x.npy"
    velocity_y_path = result_repo + "velocity_y.npy"
    
    # Load velocity data
    velocity_x = np.load(velocity_x_path)
    velocity_y = np.load(velocity_y_path)
    
    # Create grid coordinates
    x, y = np.arange(0, velocity_x.shape[1]), np.arange(0, velocity_x.shape[0])
    x, y = np.meshgrid(x, y)

    # Create a new figure and axis
    fig, ax = plt.subplots()
    
    # Create a streamplot using velocity data
    ax.streamplot(x, y, velocity_x, velocity_y, density=1)
    
    # Set axis labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    
    # Save the plot as an image
    plt.savefig(result_repo + 'velocity_streamplot_sliding_lid_parallel.png')
    
    # Close the plot
    plt.close()


def generate_arguments(grid_sizes, omega, frames, nodes, debug):
    """
    Generate a list of dictionaries containing different combinations of simulation parameters.

    Parameters:
    - grid_sizes (list of int): List of grid sizes to consider.
    - omega (float): Relaxation parameter for collision term.
    - frames (int): Number of simulation frames.
    - nodes (int, optional): Total number of nodes (default is 120).

    Returns:
    list of list of dict: A nested list containing dictionaries with various simulation parameter combinations.
    """
    result = []

    # Iterate through each grid size
    for size in grid_sizes:
        i = 2
        acc = []

        # Generate arguments with increasing node counts until nodes limit
        while i**2 <= nodes:
            acc.append(
                {
                    'np': i**2,
                    'w': omega,
                    'gs': [size, size],
                    'ns': [i, i],
                    'f': frames
                }
            )
            if debug : 
                acc['debug'] = ''
            i += 1

        result.append(acc)

    return result


if __name__ == "__main__" :
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add command-line arguments for running and plotting
    parser.add_argument("-run", "--run", action='store_true', default=False, help="Run the simulation")
    parser.add_argument('-plot', '--plot', action='store_true', default=False, help="Plot the results")
    parser.add_argument('-np', '--nodes', type=int, default=4, help="number of nodes availables")
    parser.add_argument("-debug", "--debug", action='store_true', default=False, help="enables debugging")
    args = parser.parse_args()

    if args.run:
        # If the run flag is provided
        script_command = './milestone7'
        grid_sizes = [500, 1000, 5000]
        omega = 1.6
        frames = 400
        arguments = generate_arguments(grid_sizes, omega, frames, nodes=args.nodes, debug=args.debug)
        
        # Execute the simulation with generated arguments
        calculate_execution_time(script_command, arguments)

    if args.plot:
        # If the plot flag is provided
        output_parallel_file = "./output_parallel.txt"
        
        # Plot execution time results
        plot_results_execution_time(output_parallel_file)
        
        # plot velocity data
        plot_full_velocity()

    
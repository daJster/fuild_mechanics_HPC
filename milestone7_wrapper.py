import subprocess
import os
import time

def calculate_execution_time(script_path, arguments):
    results = []

    for arg in arguments:
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

        print(command[:-1])
        try:
            output = os.system(command[:-1])

            # Calculate the execution time
            execution_time = time.time() - start_time

            result = {
                'argument': arg,
                'output': output,
                'execution_time': execution_time
            }
            results.append(result)
        except subprocess.CalledProcessError as e:
            print(f"Error executing script with argument {arg}: {e}")
    
    return results


script_path = 'milestone7.py'
arguments = [
    {'np': 4, 'w': 1.6, 'gs': [300, 300], 'ns' : [2, 2], 'f': 400},
    {'np': 4, 'w': 1.6, 'gs': [300, 300], 'ns' : [2, 2], 'f': 400},
    {'np': 4, 'w': 1.6, 'gs': [300, 300], 'ns' : [2, 2], 'f': 400}
]

results = calculate_execution_time(script_path, arguments)

# Print the results
for result in results:
    print(f"Argument: {result['argument']}")
    print(f"Output: {result['output']}")
    print(f"Execution time: {result['execution_time']} seconds")
    print('-' * 30)
    print('\n')
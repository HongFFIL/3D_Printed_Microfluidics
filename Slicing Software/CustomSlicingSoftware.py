from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt
import os
import json
import math

Z_OFFSET = 4
Z_HIGH = Z_OFFSET + 8
Z_TRAVEL = 0.5
TOP_LAYER_START = 6
TOP_LAYER_STOP = 9

# Power increasing
INITIAL_POWER = 100
TIME_CONST = 10

# Mesh level equation
X_SLOPE = -0.0085
Y_SLOPE = -0.0009
Z_INT = 0.9568


def extract_lines(input_file_path, gcode_file_path):
    
    try:
        with open(input_file_path, 'r') as input_file:
            lines = input_file.readlines()

            x_values = []
            y_values = []
            z_count = 0
            layer = 0

            with open(gcode_file_path, 'w') as gcode_file:

                for i, line in enumerate(lines):
                    columns = line.strip().split()
                        
                    if len(columns) == 4 and columns[0].startswith("G") and columns[1].startswith("X") and columns[2].startswith("Y") and columns[3].startswith("F"):
                        x_pos = float(columns[1][1:])
                        y_pos = float(columns[2][1:])
                        gcode_file.write(f'G0 X{x_pos:.2f} Y{y_pos:.2f} {columns[3]}\n')
                        x_values.append(x_pos)
                        y_values.append(y_pos)

                    elif len(columns) == 4 and columns[0].startswith("G") and columns[1].startswith("X") and columns[2].startswith("Y") and columns[3].startswith("E"):
                        x_pos = float(columns[1][1:])
                        y_pos = float(columns[2][1:])
                        gcode_file.write(f'G1 X{x_pos:.2f} Y{y_pos:.2f} {columns[3]}\n')
                        x_values.append(x_pos)
                        y_values.append(y_pos)

                    elif len(columns) == 3 and columns[0].startswith("G") and columns[1].startswith("X") and columns[2].startswith("Y"):
                        x_pos = float(columns[1][1:])
                        y_pos = float(columns[2][1:])
                        gcode_file.write(f'G0 X{x_pos:.2f} Y{y_pos:.2f}\n')
                        x_values.append(x_pos)
                        y_values.append(y_pos)
                   
                    elif 'lift nozzle' in line and columns[1].startswith("Z") and columns[2].startswith("F"):
                        gcode_file.write(f'G0 Z{Z_HIGH} {columns[2]}\n')

                    elif ";LAYER_CHANGE" in line:
                        layer_bool = 1
                        gcode_file.write(f'\n;Layer {layer}\n')
                        layer += 1

                    elif len(columns) == 3 and columns[1].startswith("Z") and layer_bool == 0:
                        z_lift = z_height + Z_TRAVEL
                        gcode_file.write(f'G0 Z{z_lift:.2f} {columns[2]}\n')

                    elif len(columns) == 3 and columns[1].startswith("Z") and layer_bool == 1:
                        # z_height = z_count * LAYER_HEIGHT + Z_OFFSET
                        # z_height = float(columns[1][1:]) + Z_OFFSET - LAYER_HEIGHT
                        z_height = float(columns[1][1:]) + Z_OFFSET
                        gcode_file.write(f'G0 Z{z_height:.2f} {columns[2]}\n') # testing three decimal z-change (this line was commented out before??)
                        z_count += 1
                        layer_bool = 0

                    elif len(columns) == 5 and columns[0].startswith("G") and columns[1].startswith("X") and columns[2].startswith("Y") and columns[3].startswith("Z") and columns[4].startswith("F"):
                        x_pos = float(columns[1][1:])
                        y_pos = float(columns[2][1:])
                        z_lift = z_height + Z_TRAVEL
                        gcode_file.write(f'G0 X{x_pos:.2f} Y{y_pos:.2f} Z{z_lift:.2f} {columns[4]}\n')
                        x_values.append(x_pos)
                        y_values.append(y_pos)

                    elif len(columns) == 2 and columns[1].startswith("Z"):
                        z_height = float(columns[1][1:]) + Z_OFFSET
                        gcode_file.write(f'G0 Z{z_height:.2f}\n') # testing three decimal z-change

                    elif len(columns) == 2 and columns[1].startswith("F"):
                        gcode_file.write(f'G0 {columns[1]}\n')

                    # elif len(columns) == 2 and columns[1] == "E0":
                    #     gcode_file.write(f'G92 {columns[1]}\n')

                print(f"Lines extracted and saved to {gcode_file}")

    except FileNotFoundError:
        print("Error: File not found.")

    return x_values, y_values

def plot_part(x_values, y_values):
    plt.figure()
    plt.plot(x_values, y_values)

    # Set x and y axis limits
    plt.xlim(0, 25)
    plt.ylim(0, 75)

    # Set aspect ratio to be equal
    plt.gca().set_aspect('equal', adjustable='box')

    # Add labels and legend
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    # plt.legend()

    # Show the plot
    plt.show()

def calculate_power(initial_power, elapsed_time, time_constant):
    """
    Calculate adjusted power based on elapsed time with a slower progression.
    Default time_constant is 12 hours (doubling every 12 hours).
    """
    MAX_POWER = 255
    power = initial_power * (2 ** (elapsed_time / time_constant))
    return min(power, MAX_POWER)

def modify_file_with_conditions(input_file_path, commands_file_path):
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()

        with open(commands_file_path, 'w') as commands_file:

            layer = 0
            extrusion_bool = 1
            speed_bool = 1
            part_powers = [INITIAL_POWER]  # Initial powers for part 0 and part 1

            for i, line in enumerate (lines):
                columns = line.strip().split()

                if len(columns) == 3 and columns[1].startswith("Z") and columns[2].startswith("F"):
                    commands_file.write(f'printer_command(p, z={columns[1][1:]}, speed=700)\n')
                    z_height = columns[1][1:]

                elif ";Layer" in line and layer == 0:
                    z_height = Z_OFFSET + Z_TRAVEL
                    commands_file.write(f'\n#{line}\n')
                    commands_file.write(f'printer_command(p, z={z_height:.2f})\n')
                    layer += 1

                elif ";Layer" in line and layer > 0 and speed_bool == 1:
                    commands_file.write(f'\n#{line}\n')
                    layer += 1

                elif ";Layer" in line and layer > 0 and speed_bool == 0:
                    commands_file.write(f'wait_for_all_movements_to_complete(p)\n')
                    commands_file.write(f'teensy_command(teensy, feed="s")\n')
                    commands_file.write(f'\n#{line}\n')
                    speed_bool = 1
                    extrusion_bool = 1
                    layer += 1

                elif len(columns) == 4 and columns[1].startswith("X") and columns[2].startswith("Y") and columns[3].startswith("F") and speed_bool == 1:
                    x_val = float(columns[1][1:])
                    y_val = float(columns[2][1:])
                    z_val = float(z_height) + (X_SLOPE * x_val + Y_SLOPE * y_val + Z_INT)

                    commands_file.write(f'printer_command(p, x={columns[1][1:]}, y={columns[2][1:]}, z={z_val:.2f}, speed=700)\n')
                    speed_bool = 0

                elif len(columns) == 4 and columns[1].startswith("X") and columns[2].startswith("Y") and columns[3].startswith("F") and speed_bool == 0:
                    commands_file.write(f'wait_for_all_movements_to_complete(p)\n')
                    commands_file.write(f'teensy_command(teensy, feed="s")\n')
                    commands_file.write(f'printer_command(p, x={columns[1][1:]}, y={columns[2][1:]}, speed=700)\n')
                    commands_file.write(f'teensy_command(teensy, feed="b100")\n')
                    speed_bool = 1
                    extrusion_bool = 1

                elif len(columns) == 2 and columns[1].startswith("F") and layer < TOP_LAYER_START:
                    commands_file.write(f'printer_command(p, speed=48)\n')

                elif len(columns) == 2 and columns[1].startswith("F") and layer >= TOP_LAYER_START and layer < TOP_LAYER_STOP:
                    commands_file.write(f'printer_command(p, speed=40)\n')

                # elif len(columns) == 2 and columns[1].startswith("F") and layer >= 11 and layer < 12:
                #     commands_file.write(f'printer_command(p, speed=65)\n')

                elif len(columns) == 2 and columns[1].startswith("F") and layer >= TOP_LAYER_STOP:
                    commands_file.write(f'printer_command(p, speed=65)\n')

                elif len(columns) == 2 and columns[1].startswith("E") and speed_bool == 0:
                    commands_file.write(f'wait_for_all_movements_to_complete(p)\n')
                    commands_file.write(f'teensy_command(teensy, feed="s")\n')
                    speed_bool = 1
                    extrusion_bool = 1

                elif len(columns) == 2 and columns[1].startswith("Z"):
                    commands_file.write(f'printer_command(p, z={columns[1][1:]})\n')
                    z_height = columns[1][1:]

                elif len(columns) == 4 and columns[1].startswith("X") and columns[2].startswith("Y") and columns[3].startswith("E") and extrusion_bool == 1:
                    x_val = float(columns[1][1:])
                    y_val = float(columns[2][1:])
                    z_val = float(z_height) + (X_SLOPE * x_val + Y_SLOPE * y_val + Z_INT)
                    
                    commands_file.write(f'wait_for_all_movements_to_complete(p)\n')
                    commands_file.write(f'teensy_command(teensy, feed="f")\n')
                    commands_file.write(f'printer_command(p, x={columns[1][1:]}, y={columns[2][1:]}, z={z_val:.2f})\n')
                    extrusion_bool = 0
                    speed_bool = 0

                elif len(columns) == 4 and columns[1].startswith("X") and columns[2].startswith("Y") and columns[3].startswith("E") and extrusion_bool == 0:
                    x_val = float(columns[1][1:])
                    y_val = float(columns[2][1:])
                    z_val = float(z_height) + (X_SLOPE * x_val + Y_SLOPE * y_val + Z_INT)
                    
                    commands_file.write(f'printer_command(p, x={columns[1][1:]}, y={columns[2][1:]}, z={z_val:.2f})\n')

                elif len(columns) == 3 and columns[1].startswith("X"):
                    x_val = float(columns[1][1:])
                    y_val = float(columns[2][1:])
                    z_val = float(z_height) + (X_SLOPE * x_val + Y_SLOPE * y_val + Z_INT)

                    commands_file.write(f'printer_command(p, x={columns[1][1:]}, y={columns[2][1:]}, z={z_val:.2f})\n')

                elif len(columns) == 5 and columns[1].startswith("X") and columns[2].startswith("Y") and columns[3].startswith("Z") and columns[4].startswith("F"):
                    commands_file.write(f'wait_for_all_movements_to_complete(p)\n')
                    commands_file.write(f'teensy_command(teensy, feed="s")\n')
                    commands_file.write(f'printer_command(p, x={columns[1][1:]}, y={columns[2][1:]}, z={columns[3][1:]}, speed=700)\n')
                    commands_file.write(f'teensy_command(teensy, feed="b100")\n')
                    speed_bool = 1
                    extrusion_bool = 1
                    z_height = columns[3][1:]

def parse_printer_commands(file_path):
    """Parse printer commands from the file and extract relevant details."""
    printer_commands = []
    current_position = {'x': 0, 'y': 0, 'z': 0}  # Assume starting at origin
    current_speed = 700  # Default initial speed
    current_layer = 0  # Track current layer

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Detect layer change
            if line.startswith("#;Layer"):
                match = re.search(r"#;Layer (\d+)", line)
                if match:
                    current_layer = int(match.group(1))
                    continue

            # Parse printer commands
            if line.startswith("printer_command"):
                # Extract parameters using regex
                params = re.findall(r"(\w+)=(\d+\.?\d*)", line)
                command = {key: float(value) for key, value in params}

                # Use the current speed if not specified
                command['speed'] = command.get('speed', current_speed)
                command['layer'] = current_layer  # Associate with the current layer
                
                # Include the last known position if not specified
                command['x'] = command.get('x', current_position['x'])
                command['y'] = command.get('y', current_position['y'])
                command['z'] = command.get('z', current_position['z'])
                
                # Add the command to the list
                printer_commands.append(command)

                # Update the current position and speed
                current_position = {'x': command['x'], 'y': command['y'], 'z': command['z']}
                current_speed = command['speed']
    
    return printer_commands

def calculate_time_by_layer(commands):
    """Calculate total printing time and time per layer based on commands."""
    total_time = 0.0  # Total time in seconds
    layer_times = {}  # Dictionary to store time for each layer
    last_position = {'x': 0, 'y': 0, 'z': 0}  # Assume starting at origin

    for command in commands:
        # Extract movement parameters
        x, y, z = command['x'], command['y'], command['z']
        speed = command['speed']  # Speed is always set in parse_printer_commands
        layer = command['layer']

        # Calculate distance using 3D formula
        distance = math.sqrt(
            (x - last_position['x'])**2 +
            (y - last_position['y'])**2 +
            (z - last_position['z'])**2
        )

        # Convert speed to mm/sec and calculate time
        speed_mm_per_sec = speed / 60  # Convert mm/min to mm/sec
        if speed_mm_per_sec > 0:  # Avoid division by zero
            time = distance / speed_mm_per_sec
            total_time += time

            # Add time to the current layer
            if layer not in layer_times:
                layer_times[layer] = 0
            layer_times[layer] += time

        # Update the last known position
        last_position = {'x': x, 'y': y, 'z': z}
    
    return total_time, layer_times

def update_extrusion_power(input_file_path, commands_file_path, layer_times):
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()

        with open(commands_file_path, 'w') as commands_file:

            layer = 0
            part_powers = [INITIAL_POWER]  # Initial powers for part 0 and part 1
            part_elapsed_times = [0]  # Elapsed times for power adjustment
            sorted_layer_times = sorted(layer_times.items())

            for i, line in enumerate (lines):
                columns = line.strip().split()

                if ";Layer" in line:
                    # Update elapsed time and adjust power for each part
                    for part_index in range(len(part_powers)):
                        part_elapsed_times[part_index] += sorted_layer_times[layer][1] / 3600
                        part_powers[part_index] = calculate_power(
                            initial_power=INITIAL_POWER + part_index,  # Base initial power
                            elapsed_time=part_elapsed_times[part_index],
                            time_constant=TIME_CONST
                        )
                    layer += 1
                    commands_file.write(line)
                    commands_file.write(f'\nteensy_command(teensy, power={int(part_powers[0])})')

                else:
                    commands_file.write(line)

def parse_commands(input_file, output_file):
    # Open and read the input file
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Initialize an empty list for the converted commands
    commands = []

    # Regex patterns for matching different types of commands
    printer_command_pattern = re.compile(r'printer_command\(p(?:, (.*?))?\)')
    teensy_command_pattern = re.compile(r'teensy_command\(teensy, (.*?)\)')
    wait_command_pattern = re.compile(r'wait_for_all_movements_to_complete\(p\)')
    layer_pattern = re.compile(r'#;Layer (\d+)')

    for line in lines:
        line = line.strip()
        
        # Check for printer_command
        match = printer_command_pattern.match(line)
        if match:
            params_str = match.group(1)
            params = {}
            if params_str:
                # Parse key-value pairs in the params
                for param in params_str.split(', '):
                    key, value = param.split('=')
                    if key in {'x', 'y', 'z', 'speed'}:
                        params[key] = float(value) if '.' in value else int(value)
            commands.append({'type': 'printer', 'params': params})
            continue
        
        # Check for teensy_command
        match = teensy_command_pattern.match(line)
        if match:
            params_str = match.group(1)
            params = {}
            # Parse key-value pairs in the params
            for param in params_str.split(', '):
                key, value = param.split('=')
                params[key] = int(value) if key == 'power' else value.strip('"')
            commands.append({'type': 'teensy', 'params': params})
            continue

        # Check for wait_for_all_movements_to_complete
        if wait_command_pattern.match(line):
            commands.append({'type': 'wait'})
            continue

        # Check for layer comment
        match = layer_pattern.match(line)
        if match:
            layer_number = int(match.group(1))
            commands.append({'type': 'comment', 'content': f'Layer {layer_number}'})
            continue

    # Write the converted commands to the output file in JSON format
    with open(output_file, 'w') as outfile:
        json.dump(commands, outfile, indent=4)

    print(f"Converted commands written to {output_file}")

def check_or_create_files(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w'):
            pass
        print(f"File '{file_path}' created.")
    else:
        print(f"File '{file_path}' already exists.")

def check_output_files(file_1, file_2, file_3, file_4):
    check_or_create_files(file_1)
    check_or_create_files(file_2)
    check_or_create_files(file_3)
    check_or_create_files(file_4)

if __name__ == "__main__":
    # input_file_path = r"C:\Users\delan\Desktop\nih\02032025\150w_180h_zz.gcode"
    # extract_file_path = r"C:\Users\delan\Desktop\nih\02032025\150w_180h_zz_extract_v2.gcode"
    # output_file_path = r"C:\Users\delan\Desktop\nih\02032025\150w_180h_zz_output_v2.txt"
    # updated_file_path = r"C:\Users\delan\Desktop\nih\02032025\150w_180h_zz_updated_v2.txt"
    # json_file_path = r"C:\Users\delan\Desktop\nih\02032025\150w_180h_zz_converted_commands_v2.json"
    # output_json_file_path = r"C:\Users\delan\Desktop\nih\01132025\150w_180h_zz_repeated_top_layer.json"

    # input_file_path = r"C:\Users\delan\Desktop\nih\02032025\circle_cal.gcode"
    # extract_file_path = r"C:\Users\delan\Desktop\nih\02032025\circle_cal_extract_v1.gcode"
    # output_file_path = r"C:\Users\delan\Desktop\nih\02032025\circle_cal_output_v1.txt"
    # updated_file_path = r"C:\Users\delan\Desktop\nih\02032025\circle_cal_updated_v1.txt"
    # json_file_path = r"C:\Users\delan\Desktop\nih\02032025\circle_cal_converted_commands_v1.json"

    # Define variables
    base_dir = r"C:\Users\delan\Desktop\nih"
    date_folder = "02072025"
    base_name = "150w_180h_zz"
    version = "v1"

    # Construct file paths dynamically
    folder_path = os.path.join(base_dir, date_folder)

    input_file_path = os.path.join(folder_path, f"{base_name}.gcode")
    extract_file_path = os.path.join(folder_path, f"{base_name}_extract_{version}.gcode")
    output_file_path = os.path.join(folder_path, f"{base_name}_output_{version}.txt")
    updated_file_path = os.path.join(folder_path, f"{base_name}_updated_{version}.txt")
    json_file_path = os.path.join(folder_path, f"{base_name}_converted_commands_{version}.json")

    check_output_files(extract_file_path, output_file_path, updated_file_path, json_file_path)
    extract_lines(input_file_path, extract_file_path)
    modify_file_with_conditions(extract_file_path, output_file_path)
    
    # Parse the file and extract printer commands
    commands = parse_printer_commands(output_file_path)
    # Calculate the total printing time and time per layer
    total_time, layer_times = calculate_time_by_layer(commands)

    update_extrusion_power(output_file_path, updated_file_path, layer_times)
    parse_commands(updated_file_path, json_file_path)

    # Call the function to repeat the topmost layer
    # repeat_top_layer(json_file_path, output_json_file_path)

    cum_time = 0
    print("\nTime Estimate Per Layer:")
    for layer, time in sorted(layer_times.items()):
        minutes, seconds = divmod(time, 60)
        cum_time += time
        cum_min, cum_sec = divmod(cum_time, 60)
        print(f"Layer {layer}: {int(minutes)} minutes and {int(seconds)} seconds")
        print(f"Cumulative time: {int(cum_min)} minutes and {int(cum_sec)} seconds")
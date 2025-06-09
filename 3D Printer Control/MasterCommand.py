from printrun.printcore import printcore
import time
import json
import serial

# Global variable to track if the printer is ready
printer_ready = False

# Function to handle printer responses and check for 'ok'
def handle_printer_response(line):
    global printer_ready
    if "ok" in line:
        printer_ready = True
    # print(f"Received printer response: {line.strip()}")

# Function to send commands to the printer
def printer_command(p, x=None, y=None, z=None, speed=None):
    command = 'G1'
    if x is not None:
        command += f' X{x}'
    if y is not None:
        command += f' Y{y}'
    if z is not None:
        command += f' Z{z}'
    if speed is not None:
        command += f' F{speed}'
    
    # print(f"Sending printer command: {command}")
    p.send(command)

    global printer_ready
    printer_ready = False
    while not printer_ready:
        time.sleep(0.01)

# Function to send an M400 to ensure all commands are processed
def wait_for_all_movements_to_complete(p):
    # print("Waiting for all movements to complete...")
    p.send('M400')
    global printer_ready
    printer_ready = False
    while not printer_ready:
        time.sleep(0.01)

# Function to control the Teensy
def teensy_command(teensy, feed=None, power=None):
    if power is not None:
        if 0 <= power <= 255:
            command = f"p{power}\n"
            teensy.write(command.encode())
            # print(f"Power set to: {power}")
        else:
            print("Invalid power value. Must be between 0 and 255.")
        return

    if feed is not None and len(feed) > 1:
        direction = feed[0]
        try:
            duration = int(feed[1:])
        except ValueError:
            print(f"Invalid duration specified in feed: {feed}")
            return

        if direction in {'f', 'b'}:
            command = f"{direction}\n"
            teensy.write(command.encode())
            # print(f"Sent command: {command.strip()} for {duration}ms")
            time.sleep(duration / 1000)
            teensy.write(b's\n')
            # print("Sent stop command: s")
        else:
            print(f"Invalid direction specified in feed: {feed}")
    elif feed in {'f', 'b', 's'}:
        command = f"{feed}\n"
        teensy.write(command.encode())
        # print(f"Sent simple command: {command.strip()}")
    else:
        print(f"Invalid feed command: {feed}")

# Main program
if __name__ == "__main__":
    # Load commands from JSON file
    input_file = r"C:\Users\delan\Desktop\nih\12112024\240w_240h_4quad_converted_commands.json"  # Replace with your JSON file name
    with open(input_file, 'r') as infile:
        commands = json.load(infile)

    # Initialize printer and Teensy connections
    p = printcore('COM6', 115200)
    teensy = serial.Serial(port='COM5', baudrate=115200, timeout=0.1)
    p.recvcb = handle_printer_response

    while not p.online:
        time.sleep(0.1)

    # Execute commands from the JSON file
    for cmd in commands:
        if cmd['type'] == 'printer':
            printer_command(p, **cmd['params'])
        elif cmd['type'] == 'teensy':
            teensy_command(teensy, **cmd['params'])
        elif cmd['type'] == 'wait':
            wait_for_all_movements_to_complete(p)
        elif cmd['type'] == 'comment':
            print(f"Comment: {cmd['content']}")

    printer_command(p, x=130, y=30, z=50, speed=700)

    print("\nAll movements completed.")
    p.disconnect()

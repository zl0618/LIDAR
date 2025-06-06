import serial
import time

port = 'COM4'  # Your LIDAR port
baudrate = 115200

try:
    ser = serial.Serial(port, baudrate, timeout=1)
    print(f"Connected to {port} at {baudrate} baud")
    
    # Try to initialize LIDAR - some need specific commands
    # Common initialization commands for various LIDAR models:
    init_commands = [
        b'\xA5\x60',       # RPLidar start scan command
        b'\xA5\x20',       # RPLidar force scan command
        b'\x02',           # Some LIDAR models
        b'\x42\x57\x02\x00'  # YDLIDAR init sequence
    ]
    
    # Send each init command with delay
    for cmd in init_commands:
        print(f"Sending init command: {' '.join([f'{b:02X}' for b in cmd])}")
        ser.write(cmd)
        time.sleep(1)
    
    # Monitor incoming data
    print("Monitoring incoming data for 10 seconds...")
    start_time = time.time()
    total_bytes = 0
    
    while (time.time() - start_time) < 10:
        if ser.in_waiting > 0:
            data = ser.read(ser.in_waiting)
            total_bytes += len(data)
            print(f"Received {len(data)} bytes: {' '.join([f'{b:02X}' for b in data[:20]])}")
            if len(data) > 20:
                print("...")
        time.sleep(0.1)
    
    print(f"Total bytes received: {total_bytes}")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
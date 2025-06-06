import serial
import serial.tools.list_ports
import time

def list_available_ports():
    """List all available serial ports"""
    ports = serial.tools.list_ports.comports()
    available_ports = []
    
    print("Available serial ports:")
    for port, desc, hwid in sorted(ports):
        print(f"  {port}: {desc}")
        available_ports.append(port)
    
    return available_ports

def test_port(port_name, baudrates=[115200, 9600, 57600, 38400, 19200]):
    """Test communication with a specific port at different baud rates"""
    print(f"\nTesting port {port_name}...")
    
    for baudrate in baudrates:
        try:
            print(f"  Trying baudrate {baudrate}...", end="")
            with serial.Serial(port_name, baudrate, timeout=2) as ser:
                time.sleep(0.1)  # Wait for connection to stabilize
                
                # Try to read some data
                if ser.in_waiting > 0:
                    data = ser.read(ser.in_waiting)
                    print(f" SUCCESS - Received {len(data)} bytes")
                    print(f"    Data preview: {data[:50]}")
                    return True, baudrate
                else:
                    # Send a test command (common LIDAR commands)
                    test_commands = [b'\x00', b'\xA5\x20\x00\x00\x00\x00\x02', b'b']
                    for cmd in test_commands:
                        ser.write(cmd)
                        time.sleep(0.1)
                        if ser.in_waiting > 0:
                            data = ser.read(ser.in_waiting)
                            print(f" SUCCESS - Response to command")
                            print(f"    Data preview: {data[:50]}")
                            return True, baudrate
                    
                    print(" No response")
                    
        except serial.SerialException as e:
            print(f" ERROR - {e}")
        except Exception as e:
            print(f" ERROR - {e}")
    
    return False, None

def main():
    print("LIDAR Port Detection Tool")
    print("=" * 30)
    
    # List available ports
    available_ports = list_available_ports()
    
    if not available_ports:
        print("No serial ports found!")
        return
    
    print(f"\nFound {len(available_ports)} port(s)")
    
    # Test each port
    successful_ports = []
    
    for port in available_ports:
        success, baudrate = test_port(port)
        if success:
            successful_ports.append((port, baudrate))
    
    print("\n" + "=" * 30)
    print("RESULTS:")
    
    if successful_ports:
        print("Potential LIDAR ports found:")
        for port, baudrate in successful_ports:
            print(f"  ✓ {port} at {baudrate} baud")
        
        print(f"\nRecommendation: Try port {successful_ports[0][0]} first")
    else:
        print("No responsive ports found.")
        print("Your LIDAR might be:")
        print("  - Using a different baud rate")
        print("  - Not properly connected")
        print("  - Requiring specific initialization")

if __name__ == "__main__":
    main()
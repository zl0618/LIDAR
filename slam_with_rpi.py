import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import math

class MS200Decoder:
    def __init__(self):
        self.buffer = bytearray()
        self.header_byte = 0x54
        self.packet_count = 0
        # Add timing for delay monitoring
        self.last_packet_time = time.time()
        
    def add_data(self, data):
        """Add raw data to the buffer"""
        self.buffer.extend(data)
        # Reduce buffer size to prevent delay accumulation
        if len(self.buffer) > 2048:  # Reduced from 4096
            self.buffer = self.buffer[-1024:]  # Keep less data
        return self._process_buffer()
    
    def _process_buffer(self):
        """Process buffer and return list of points"""
        points = []
        
        while len(self.buffer) > 14:
            # Find header
            header_pos = self.buffer.find(bytes([self.header_byte]))
            if header_pos == -1:
                self.buffer.clear()
                break
            elif header_pos > 0:
                self.buffer = self.buffer[header_pos:]
                
            if len(self.buffer) < 11:
                break
            
            # Get point count
            n_points = self.buffer[1] & 0x1F
            if n_points < 1 or n_points > 20:
                self.buffer = self.buffer[1:]
                continue
                
            # Calculate packet length
            expected_length = 11 + 3 * n_points
            if len(self.buffer) < expected_length:
                break
                
            packet = self.buffer[:expected_length]
            
            # Parse packet
            start_angle = (packet[4] | (packet[5] << 8)) / 100.0
            end_angle_pos = 6 + 3 * n_points
            end_angle = (packet[end_angle_pos] | (packet[end_angle_pos+1] << 8)) / 100.0
            
            # Extract points
            for i in range(n_points):
                point_pos = 6 + i * 3
                distance = packet[point_pos] | (packet[point_pos+1] << 8)
                intensity = packet[point_pos+2]
                
                if 50 <= distance <= 8000 and intensity > 15:
                    if n_points > 1:
                        angle = start_angle + i * (end_angle - start_angle) / (n_points - 1)
                    else:
                        angle = start_angle
                    
                    # Convert to cartesian immediately
                    angle_rad = math.radians(angle)
                    x = distance * math.sin(angle_rad)
                    y = distance * math.cos(angle_rad)
                    points.append([x, y, angle, distance])
            
            self.buffer = self.buffer[expected_length:]
            self.packet_count += 1
            self.last_packet_time = time.time()
        
        return points

def main():
    decoder = MS200Decoder()
    
    # Reduce data storage for real-time performance
    scan_points = deque(maxlen=800)  # Reduced from 2000 for faster processing
    frame_count = 0
    last_update_time = time.time()
    
    # Setup visualization - just polar plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'), facecolor='black')  # Smaller figure
    
    # Configure polar plot
    ax.set_facecolor('black')
    ax.set_title('MS200 LIDAR Live Scan - Real-time', color='white', pad=20, fontsize=14)
    ax.set_ylim(0, 3000)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.grid(True, alpha=0.3, color='gray')
    
    # Simplified background - remove heavy circles
    ax.set_rticks([500, 1000, 1500, 2000, 2500, 3000])
    ax.set_rmax(3000)
    ax.set_rlabel_position(45)
    
    # Simplified legend
    legend_text = ax.text(0.75, 0.02, 'Red: <300mm\nLime: 300-500mm\nCyan: 500-1500mm\nYellow: >1500mm', 
                         transform=ax.transAxes, fontsize=8, color='white', alpha=0.8,
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Status text
    status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         verticalalignment='top', fontsize=10, color='white',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Serial connection with optimized settings
    ser = None
    
    try:
        ser = serial.Serial('/dev/ttyUSB0', 230400, timeout=0.01)  # Reduced timeout
        ser.reset_input_buffer()  # Clear any existing data
        print(f"Connected to MS200 LIDAR on /dev/ttyUSB0")
        time.sleep(0.5)  # Reduced sleep time
    except serial.SerialException as e:
        if "Permission denied" in str(e):
            print(f"Permission denied accessing /dev/ttyUSB0")
            print("Try running: sudo usermod -a -G dialout $USER")  
            print("Then log out and back in, or run with sudo")
        else:
            print(f"Could not connect to LIDAR on /dev/ttyUSB0: {e}")
        return
    except Exception as e:
        print(f"Error connecting to LIDAR: {e}")
        return
    
    def update(frame):
        nonlocal frame_count, last_update_time
        frame_count += 1
        current_time = time.time()
        
        if ser and ser.is_open:
            try:
                # Read all available data at once for better throughput
                if ser.in_waiting > 0:
                    # Read larger chunks but limit to prevent buffer overflow
                    data_size = min(ser.in_waiting, 4096)
                    data = ser.read(data_size)
                    new_points = decoder.add_data(data)
                    
                    if new_points:
                        # Clear old points more aggressively for real-time display
                        if len(scan_points) > 600:
                            # Remove older points to keep only recent data
                            for _ in range(len(scan_points) - 400):
                                scan_points.popleft()
                        
                        # Add new points
                        scan_points.extend(new_points)
                        
                        # Calculate actual FPS and delay
                        fps = 1.0 / (current_time - last_update_time) if last_update_time else 0
                        data_delay = current_time - decoder.last_packet_time
                        
                        # Update status with delay information
                        status = f"Port: /dev/ttyUSB0\n"
                        status += f"Points: {len(scan_points)}\n"
                        status += f"FPS: {fps:.1f}\n"
                        status += f"Delay: {data_delay*1000:.1f}ms\n"
                        status += f"Buffer: {len(decoder.buffer)}B"
                        status_text.set_text(status)
                        
                        # Update visualization more frequently but with less data
                        if scan_points:
                            angles = [math.radians(p[2]) for p in scan_points] 
                            distances = [p[3] for p in scan_points]
                            
                            # Simplified color coding for performance
                            colors = ['red' if d < 300 else 'lime' if d < 500 else 'cyan' if d < 1500 else 'yellow' for d in distances]
                            
                            # Clear and redraw - optimized
                            ax.clear()
                            ax.set_facecolor('black')
                            ax.set_title('MS200 LIDAR Live Scan - Real-time', color='white', pad=20, fontsize=14)
                            ax.set_ylim(0, 3000)
                            ax.set_theta_zero_location('N')
                            ax.set_theta_direction(-1)
                            ax.grid(True, alpha=0.3, color='gray')
                            ax.set_rticks([500, 1000, 1500, 2000, 2500, 3000])
                            ax.set_rmax(3000)
                            ax.set_rlabel_position(45)
                            
                            # Plot with uniform size for performance
                            ax.scatter(angles, distances, s=3, c=colors, alpha=0.8, edgecolors='none')
                        
                        last_update_time = current_time
                        
            except Exception as e:
                print(f"Update error: {e}")
        else:
            status_text.set_text("No LIDAR connection")
    
    # Faster animation interval for reduced delay
    ani = FuncAnimation(fig, update, interval=20, cache_frame_data=False)  # 50fps instead of 20fps
    
    try:
        plt.tight_layout()
        plt.show()
    finally:
        if ser and ser.is_open:
            ser.close()
            print("Serial port closed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

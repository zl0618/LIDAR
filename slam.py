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
        
    def add_data(self, data):
        """Add raw data to the buffer"""
        self.buffer.extend(data)
        if len(self.buffer) > 4096:
            self.buffer = self.buffer[-2048:]
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
        
        return points

def main():
    decoder = MS200Decoder()
    
    # Setup visualization - just polar plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'), facecolor='black')
    
    # Configure polar plot
    ax.set_facecolor('black')
    ax.set_title('MS200 LIDAR Live Scan', color='white', pad=30, fontsize=16)
    ax.set_ylim(0, 3000)  # 0 to 3000mm range
    ax.set_theta_zero_location('N')  # 0 degrees at top
    ax.set_theta_direction(-1)  # Clockwise
    ax.grid(True, alpha=0.4, color='gray')
    
    # Add white background circle using plot instead of fill_between
    theta_circle = np.linspace(0, 2*np.pi, 1000)
    for radius in range(100, 3000, 100):  # Multiple circles for background
        ax.plot(theta_circle, np.full(1000, radius), color='white', alpha=0.02, linewidth=0.5)
    
    # Range circles
    ax.set_rticks([500, 1000, 1500, 2000, 2500, 3000])
    ax.set_rmax(3000)
    
    # Color the range rings
    ax.set_rlabel_position(45)  # Move radial labels to 45 degrees
    
    # Polar scatter plot - using intensity for better visibility
    scan_plot = ax.scatter([], [], s=5, c='cyan', alpha=0.9, edgecolors='white', linewidths=0.1)
    
    # Add a small legend explaining the points
    legend_text = ax.text(0.77, 0.02, 'White: Scanning Space\nCyan: Walls/Obstacles', transform=ax.transAxes,
                         fontsize=9, color='white', alpha=0.9,
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    # Status text
    status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         verticalalignment='top', fontsize=11, color='white',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='cyan'))
    
    # Serial connection
    try:
        ser = serial.Serial('COM4', 230400, timeout=0.05)
        print("Connected to MS200 LIDAR on COM4")
        time.sleep(1)
    except Exception as e:
        print(f"Serial connection error: {e}")
        return
    
    def update(frame):
        if ser and ser.is_open:
            try:
                # Read available data
                if ser.in_waiting > 0:
                    data = ser.read(min(ser.in_waiting, 2048))
                    new_points = decoder.add_data(data)
                    
                    if new_points:
                        # Update status
                        status = f"Packets: {decoder.packet_count}\n"
                        status += f"Points: {len(new_points)}\n"
                        status += f"Buffer: {len(decoder.buffer)} bytes\n"
                        status += f"Range: {min(p[3] for p in new_points):.0f}-{max(p[3] for p in new_points):.0f}mm"
                        status_text.set_text(status)
                        
                        # Update polar scan plot - keep original orientation
                        angles = [math.radians(p[2]) for p in new_points] 
                        distances = [p[3] for p in new_points]
                        
                        # Color coding based on distance (optional)
                        colors = ['cyan' if d > 1000 else '#00FFFF' for d in distances]  # Brighter cyan for close objects
                        
                        scan_plot.set_offsets(np.column_stack([angles, distances]))
                        
            except Exception as e:
                print(f"Update error: {e}")
    
    # Start animation
    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    
    if ser and ser.is_open:
        ser.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

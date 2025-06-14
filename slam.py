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
        # Enhanced filtering parameters - optimized for real-time
        self.min_distance = 50  # Back to original for better detection
        self.max_distance = 8000
        self.min_intensity = 15  # Balanced for performance
        self.angle_tolerance = 0.5
        self.previous_points = deque(maxlen=30)  # Reduced for performance
        # Add timing for delay monitoring
        self.last_packet_time = time.time()
        
    def add_data(self, data):
        """Add raw data to the buffer"""
        self.buffer.extend(data)
        # Reduce buffer size to prevent delay accumulation
        if len(self.buffer) > 2048:  # Reduced from 4096
            self.buffer = self.buffer[-1024:]  # Keep less data
        return self._process_buffer()
    
    def _validate_distance_intensity(self, distance, intensity, angle):
        """Enhanced but performance-optimized validation"""
        # Basic range check
        if not (self.min_distance <= distance <= self.max_distance):
            return False
        
        # Simplified intensity-based filtering for performance
        if intensity < self.min_intensity:
            return False
            
        # Simplified outlier detection for real-time performance
        if len(self.previous_points) > 10:
            nearby_points = [p for p in list(self.previous_points)[-10:] 
                           if abs(p[2] - angle) < 15]  # Relaxed angle check
            if nearby_points and len(nearby_points) > 2:
                distances = [p[3] for p in nearby_points]
                median_dist = np.median(distances)
                # Less aggressive outlier rejection for real-time
                if abs(distance - median_dist) > median_dist * 0.8 and distance < 150:
                    return False
        
        return True
    
    def _interpolate_angle(self, start_angle, end_angle, point_index, total_points):
        """Enhanced angle interpolation with boundary handling"""
        if total_points == 1:
            return start_angle
            
        # Handle angle wraparound (359° to 1°)
        angle_diff = end_angle - start_angle
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
            
        # Linear interpolation with improved precision
        interpolated = start_angle + (angle_diff * point_index / (total_points - 1))
        
        # Normalize angle to 0-360 range
        while interpolated < 0:
            interpolated += 360
        while interpolated >= 360:
            interpolated -= 360
            
        return interpolated
    
    def _process_buffer(self):
        """Process buffer and return list of points with enhanced accuracy"""
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
            
            # Get point count with validation
            n_points = self.buffer[1] & 0x1F
            if n_points < 1 or n_points > 20:
                self.buffer = self.buffer[1:]
                continue
                
            # Calculate packet length
            expected_length = 11 + 3 * n_points
            if len(self.buffer) < expected_length:
                break
                
            packet = self.buffer[:expected_length]
            
            # Enhanced angle parsing with error checking
            try:
                start_angle = (packet[4] | (packet[5] << 8)) / 100.0
                end_angle_pos = 6 + 3 * n_points
                end_angle = (packet[end_angle_pos] | (packet[end_angle_pos+1] << 8)) / 100.0
                
                # Validate angle range
                if not (0 <= start_angle <= 360 and 0 <= end_angle <= 360):
                    self.buffer = self.buffer[expected_length:]
                    continue
                    
            except IndexError:
                self.buffer = self.buffer[1:]
                continue
            
            # Extract points with enhanced but optimized processing
            valid_points = []
            for i in range(n_points):
                point_pos = 6 + i * 3
                distance = packet[point_pos] | (packet[point_pos+1] << 8)
                intensity = packet[point_pos+2]
                
                # Enhanced angle calculation
                angle = self._interpolate_angle(start_angle, end_angle, i, n_points)
                
                # Enhanced validation
                if self._validate_distance_intensity(distance, intensity, angle):
                    # Convert to cartesian with higher precision
                    angle_rad = math.radians(angle)
                    x = distance * math.sin(angle_rad)
                    y = distance * math.cos(angle_rad)
                    
                    point_data = [x, y, angle, distance, intensity]
                    valid_points.append(point_data)
                    points.append(point_data)
            
            # Store valid points for future reference
            self.previous_points.extend(valid_points)
            
            self.buffer = self.buffer[expected_length:]
            self.packet_count += 1
            self.last_packet_time = time.time()
        
        return points

def main():
    decoder = MS200Decoder()
    
    # Reduce data storage for real-time performance
    scan_points = deque(maxlen=600)  # Optimized for real-time display
    frame_count = 0
    last_update_time = time.time()
    
    # Setup optimized visualization
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'), facecolor='black')
    
    # Configure polar plot with optimized settings
    ax.set_facecolor('black')
    ax.set_title('MS200 LIDAR Live Scan - Enhanced Real-time', color='white', pad=20, fontsize=14)
    ax.set_ylim(0, 3000)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.grid(True, alpha=0.3, color='gray')
    
    # Optimized range settings
    ax.set_rticks([500, 1000, 1500, 2000, 2500, 3000])
    ax.set_rmax(3000)
    ax.set_rlabel_position(45)
    
    # Simplified legend for performance
    legend_text = ax.text(0.75, 0.02, 'Red: <300mm\nLime: 300-500mm\nCyan: 500-1500mm\nYellow: >1500mm', 
                         transform=ax.transAxes, fontsize=8, color='white', alpha=0.8,
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Status text with enhanced metrics
    status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         verticalalignment='top', fontsize=10, color='white',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Serial connection with optimized settings
    ser = None
    
    try:
        ser = serial.Serial('COM6', 230400, timeout=0.01)  # Reduced timeout for responsiveness
        ser.reset_input_buffer()  # Clear any existing data
        print("Connected to MS200 LIDAR on COM6")
        time.sleep(0.5)  # Reduced sleep time
    except Exception as e:
        print(f"Serial connection error: {e}")
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
                        if len(scan_points) > 500:
                            # Remove older points to keep only recent data
                            for _ in range(len(scan_points) - 300):
                                scan_points.popleft()
                        
                        # Add new points
                        scan_points.extend(new_points)
                        
                        # Calculate actual FPS and delay
                        fps = 1.0 / (current_time - last_update_time) if last_update_time else 0
                        data_delay = current_time - decoder.last_packet_time
                        
                        # Enhanced status with accuracy metrics
                        distances = [p[3] for p in new_points]
                        intensities = [p[4] for p in new_points]
                        
                        status = f"Port: COM6\n"
                        status += f"Points: {len(scan_points)} ({len(new_points)} new)\n"
                        status += f"FPS: {fps:.1f}\n"
                        status += f"Delay: {data_delay*1000:.1f}ms\n"
                        status += f"Range: {min(distances):.0f}-{max(distances):.0f}mm\n"
                        status += f"Avg Intensity: {np.mean(intensities):.1f}"
                        status_text.set_text(status)
                        
                        # Update visualization with enhanced color coding
                        if scan_points:
                            angles = [math.radians(p[2]) for p in scan_points] 
                            distances = [p[3] for p in scan_points]
                            
                            # Enhanced color coding for better object distinction
                            colors = ['red' if d < 300 else 'lime' if d < 500 else 'cyan' if d < 1500 else 'yellow' for d in distances]
                            
                            # Clear and redraw - optimized
                            ax.clear()
                            ax.set_facecolor('black')
                            ax.set_title('MS200 LIDAR Live Scan - Enhanced Real-time', color='white', pad=20, fontsize=14)
                            ax.set_ylim(0, 3000)
                            ax.set_theta_zero_location('N')
                            ax.set_theta_direction(-1)
                            ax.grid(True, alpha=0.3, color='gray')
                            ax.set_rticks([500, 1000, 1500, 2000, 2500, 3000])
                            ax.set_rmax(3000)
                            ax.set_rlabel_position(45)
                            
                            # Plot with optimized settings
                            ax.scatter(angles, distances, s=4, c=colors, alpha=0.8, edgecolors='none')
                        
                        last_update_time = current_time
                        
            except Exception as e:
                print(f"Update error: {e}")
        else:
            status_text.set_text("No LIDAR connection")
    
    # Faster animation interval for reduced delay
    ani = FuncAnimation(fig, update, interval=20, cache_frame_data=False)  # 50fps for real-time
    
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

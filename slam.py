import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from collections import deque

class MS200Decoder:
    def __init__(self):
        self.buffer = bytearray()
        self.last_scan = None
        self.header_byte = 0x54  # MS200 header byte (84 decimal) as shown in documentation
        self.debug = False  # Reduced debug output for performance
        self.packet_count = 0
        self.error_count = 0
        self.scan_history = deque(maxlen=10)  # Keep recent scans for filtering
        
    def add_data(self, data):
        """Add raw data to the buffer"""
        self.buffer.extend(data)
        # Limit buffer size to prevent memory issues
        if len(self.buffer) > 8192:
            self.buffer = self.buffer[-4096:]
        self._process_buffer()
    
    def _process_buffer(self):
        """Process the buffer using MS200k format specification"""
        min_packet_size = 14  # More conservative minimum size
        
        while len(self.buffer) > min_packet_size:
            # Look for header byte
            header_pos = self.buffer.find(bytes([self.header_byte]))
            if header_pos == -1:
                # No header found, clear buffer
                self.buffer.clear()
                return
            elif header_pos > 0:
                # Remove data before header
                self.buffer = self.buffer[header_pos:]
                
            # Check if we have enough data to read point count
            if len(self.buffer) < 2:
                return
            
            # According to doc: low 5 bits indicate point count    
            n_points = self.buffer[1] & 0x1F
            
            # MS200k typically has 12 points per packet per documentation
            if n_points < 1 or n_points > 20:  # More restrictive range
                if self.debug:
                    print(f"Invalid point count: {n_points}, skipping packet")
                self.buffer = self.buffer[1:]
                self.error_count += 1
                continue
                
            # Calculate expected packet length
            expected_length = 11 + 3 * n_points
            
            if len(self.buffer) < expected_length:
                return
                
            packet = self.buffer[:expected_length]
            
            # Verify packet integrity with simple checksum
            if not self._verify_packet(packet):
                self.buffer = self.buffer[1:]
                self.error_count += 1
                continue
            
            # Extract data
            speed = (packet[2] | (packet[3] << 8)) / 100.0
            start_angle = (packet[4] | (packet[5] << 8)) / 100.0
            
            end_angle_pos = 6 + 3 * n_points
            if end_angle_pos + 1 >= len(packet):
                self.buffer = self.buffer[1:]
                continue
                
            end_angle = (packet[end_angle_pos] | (packet[end_angle_pos+1] << 8)) / 100.0
            
            # Parse point data with improved filtering
            points_data = self._parse_points(packet, n_points, start_angle, end_angle)
            
            if len(points_data['angles']) > 0:
                self.last_scan = {
                    'angles': np.array(points_data['angles']),
                    'distances': np.array(points_data['distances']),
                    'intensities': np.array(points_data['intensities']),
                    'timestamp': time.time(),
                    'speed': speed,
                    'quality': self._calculate_scan_quality(points_data)
                }
                
                # Add to history for filtering
                self.scan_history.append(self.last_scan)
                self.packet_count += 1
                
                if self.debug or (self.packet_count % 50 == 0):
                    print(f"Processed scan #{self.packet_count}: {len(points_data['angles'])} points, "
                          f"speed: {speed:.1f}°/s, quality: {self.last_scan['quality']:.2f}")
            
            self.buffer = self.buffer[expected_length:]
    
    def _verify_packet(self, packet):
        """Basic packet verification"""
        if len(packet) < 11:
            return False
        
        # Check if speed is reasonable (0-1000 deg/s)
        speed = (packet[2] | (packet[3] << 8)) / 100.0
        if speed < 0 or speed > 1000:
            return False
            
        return True
    
    def _parse_points(self, packet, n_points, start_angle, end_angle):
        """Parse point data with improved filtering"""
        distances = []
        intensities = []
        angles = []
        
        for i in range(n_points):
            point_pos = 6 + i * 3
            
            if point_pos + 2 >= len(packet):
                break
            
            distance = packet[point_pos] | (packet[point_pos+1] << 8)
            intensity = packet[point_pos+2]
            
            # Improved filtering
            if (intensity > 15 and  # Valid intensity per documentation
                50 <= distance <= 12000 and  # Reasonable range (5cm to 12m)
                intensity < 255):  # Avoid saturated readings
                
                # Calculate angle
                if n_points > 1:
                    angle = start_angle + i * (end_angle - start_angle) / (n_points - 1)
                else:
                    angle = start_angle
                    
                angle = angle % 360
                
                distances.append(distance)
                intensities.append(intensity)
                angles.append(angle)
        
        return {'distances': distances, 'intensities': intensities, 'angles': angles}
    
    def _calculate_scan_quality(self, points_data):
        """Calculate scan quality based on number of points and intensity distribution"""
        if len(points_data['distances']) == 0:
            return 0.0
        
        # Quality factors
        point_density = min(1.0, len(points_data['distances']) / 12.0)  # Expect ~12 points
        intensity_variance = np.var(points_data['intensities']) / 10000.0  # Normalize variance
        distance_consistency = 1.0 / (1.0 + np.std(np.diff(points_data['distances'])) / 1000.0)
        
        return (point_density * 0.5 + intensity_variance * 0.2 + distance_consistency * 0.3)
    
    def get_cartesian_points(self):
        """Convert polar coordinates to cartesian with filtering"""
        if self.last_scan is None:
            return None
        
        # Only return high-quality scans
        if self.last_scan['quality'] < 0.3:
            return None
            
        angles_rad = np.radians(self.last_scan['angles'])
        distances = self.last_scan['distances']
        
        # Apply median filter to reduce noise
        if len(distances) > 3:
            distances = self._median_filter(distances, window=3)
        
        # Convert to cartesian coordinates
        x = distances * np.sin(angles_rad)  
        y = distances * np.cos(angles_rad)  
        
        return np.column_stack((x, y))
    
    def _median_filter(self, data, window=3):
        """Simple median filter for noise reduction"""
        filtered = []
        data = list(data)
        
        for i in range(len(data)):
            start = max(0, i - window//2)
            end = min(len(data), i + window//2 + 1)
            window_data = data[start:end]
            filtered.append(np.median(window_data))
        
        return filtered
    
    def get_stats(self):
        """Get decoder statistics"""
        total_packets = self.packet_count + self.error_count
        success_rate = self.packet_count / max(1, total_packets) * 100
        return {
            'packets_processed': self.packet_count,
            'errors': self.error_count,
            'success_rate': success_rate
        }

class OccupancyGrid:
    def __init__(self, resolution=0.05, size=20.0):
        self.resolution = resolution
        self.size = size
        self.grid_size = int(size / resolution)
        self.grid = np.ones((self.grid_size, self.grid_size), dtype=np.float32) * 0.5
        self.center = self.grid_size // 2
        self.update_count = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
    @property
    def grid_for_display(self):
        """Return the grid with enhanced visualization"""
        # Apply some smoothing for better visualization
        from scipy import ndimage
        try:
            smoothed = ndimage.gaussian_filter(self.grid, sigma=0.5)
            return smoothed
        except ImportError:
            return self.grid
    
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        gx = int(self.center + x / self.resolution)
        gy = int(self.center + y / self.resolution)
        
        gx = max(0, min(gx, self.grid_size - 1))
        gy = max(0, min(gy, self.grid_size - 1))
        
        return gx, gy
    
    def update_cell(self, x, y, occupied, confidence=1.0):
        """Update cell with confidence weighting"""
        gx, gy = self.world_to_grid(x, y)
        
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            self.update_count[gy, gx] += 1
            
            # Use confidence to weight updates
            if occupied:
                update_amount = 0.1 * confidence
                self.grid[gy, gx] = min(1.0, self.grid[gy, gx] + update_amount)
            else:
                update_amount = 0.05 * confidence
                self.grid[gy, gx] = max(0.0, self.grid[gy, gx] - update_amount)

class SimpleSlam:
    def __init__(self):
        self.pose = [0, 0, 0]  # x, y, theta in mm and degrees
        self.map = OccupancyGrid(resolution=0.05, size=25.0)  # Larger map
        self.prev_scan = None
        self.scan_count = 0
        self.pose_history = deque(maxlen=100)
        
    def process_scan(self, points):
        """Process scan with improved SLAM"""
        if points is None or len(points) == 0:
            return
            
        self.scan_count += 1
        
        # Simple odometry update (would be improved with actual odometry data)
        if self.prev_scan is not None and len(self.prev_scan) > 5 and len(points) > 5:
            self._estimate_motion(points)
        
        # Update map
        self.update_map(points)
        self.prev_scan = points.copy()
        
        # Store pose history
        self.pose_history.append(self.pose.copy())
    
    def _estimate_motion(self, current_points):
        """Simple scan matching for pose estimation"""
        try:
            # Very basic scan matching - in practice, use ICP or similar
            if len(current_points) > 3 and len(self.prev_scan) > 3:
                # Calculate centroids
                current_centroid = np.mean(current_points, axis=0)
                prev_centroid = np.mean(self.prev_scan, axis=0)
                
                # Estimate translation (very basic)
                dx = current_centroid[0] - prev_centroid[0]
                dy = current_centroid[1] - prev_centroid[1]
                
                # Apply small motion update (conservative)
                if abs(dx) < 100 and abs(dy) < 100:  # Sanity check
                    self.pose[0] += dx * 0.1  # Reduce motion estimate
                    self.pose[1] += dy * 0.1
        except Exception as e:
            pass  # Ignore motion estimation errors
    
    def update_map(self, points):
        """Enhanced map update with confidence"""
        robot_x = self.pose[0] / 1000.0
        robot_y = self.pose[1] / 1000.0
        robot_theta = np.radians(self.pose[2])
        
        for i, point in enumerate(points):
            point_x = point[0] / 1000.0
            point_y = point[1] / 1000.0
            
            # Transform to world coordinates
            world_x = robot_x + point_x * np.cos(robot_theta) - point_y * np.sin(robot_theta)
            world_y = robot_y + point_x * np.sin(robot_theta) + point_y * np.cos(robot_theta)
            
            # Calculate confidence based on distance and consistency
            distance = np.sqrt(point_x**2 + point_y**2)
            confidence = max(0.3, 1.0 - distance / 10.0)  # Closer points are more reliable
            
            # Mark as occupied
            self.map.update_cell(world_x, world_y, True, confidence)
            
            # Clear ray with distance-based confidence
            self.clear_ray(robot_x, robot_y, world_x, world_y, confidence * 0.5)
    
    def clear_ray(self, x0, y0, x1, y1, confidence=1.0):
        """Enhanced ray tracing with confidence"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        cx, cy = self.map.world_to_grid(x0, y0)
        tx, ty = self.map.world_to_grid(x1, y1)
        
        max_steps = int(self.map.grid_size * 1.5)
        step_count = 0
        
        while (cx != tx or cy != ty) and step_count < max_steps:
            # Mark cell as free with confidence
            if 0 <= cx < self.map.grid_size and 0 <= cy < self.map.grid_size:
                current_confidence = confidence * (1.0 - step_count / max_steps)
                self.map.grid[cy, cx] = max(0.0, self.map.grid[cy, cx] - 0.01 * current_confidence)
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                new_cx = cx + sx
                if 0 <= new_cx < self.map.grid_size:
                    cx = new_cx
            if e2 < dx:
                err += dx
                new_cy = cy + sy
                if 0 <= new_cy < self.map.grid_size:
                    cy = new_cy
                    
            step_count += 1

def main():
    decoder = MS200Decoder()
    slam = SimpleSlam()
    
    # Enhanced visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Enhanced MS200 LIDAR SLAM Visualization')
    
    # LIDAR scan plot with polar grid
    scan_plot = ax1.scatter([], [], s=3, c='red', alpha=0.7)
    ax1.set_xlim(-6000, 6000)
    ax1.set_ylim(-6000, 6000)
    ax1.set_title('Current Scan (Local Frame)')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Add range circles
    for radius in [1000, 2000, 3000, 4000, 5000]:
        circle = plt.Circle((0, 0), radius, fill=False, color='gray', alpha=0.3, linestyle='--')
        ax1.add_patch(circle)
    
    # Add orientation markers
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        rad = np.radians(angle)
        ax1.plot([0, 5500*np.sin(rad)], [0, 5500*np.cos(rad)], 'gray', alpha=0.3, linestyle=':')
        ax1.text(5800*np.sin(rad), 5800*np.cos(rad), f"{angle}°", 
                ha='center', va='center', fontsize=8, alpha=0.7)
    
    # Enhanced map plot
    map_plot = ax2.imshow(slam.map.grid_for_display, origin='lower', 
                         extent=[-slam.map.size/2, slam.map.size/2, -slam.map.size/2, slam.map.size/2],
                         cmap='RdYlBu_r', vmin=0, vmax=1)
    
    robot_marker = ax2.plot([], [], 'go', markersize=8, markeredgecolor='black', markeredgewidth=2)[0]
    robot_direction = ax2.quiver([], [], [], [], color='green', scale=15, width=0.003)
    
    # Add trajectory line
    trajectory_line = ax2.plot([], [], 'g-', alpha=0.5, linewidth=1)[0]
    
    ax2.set_title('Occupancy Grid Map')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Enhanced status display
    status_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                          verticalalignment='top', fontsize=9, 
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Serial connection
    port = 'COM4'
    baudrate = 115200
    
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"Connected to {port} at {baudrate} baud")
        
        # Initialization commands
        init_commands = [
            b'\xA5\xF5\xA2\xC1\x01\x81\xB2\x31\xF2',
            b'\xA5\x20',
            b'\xA5\x60',
            b'\x42\x57\x02\x00',
        ]
        
        for cmd in init_commands:
            print(f"Sending: {' '.join([f'{b:02X}' for b in cmd])}")
            ser.write(cmd)
            time.sleep(0.2)
            
            if ser.in_waiting > 0:
                response = ser.read(ser.in_waiting)
                print(f"Response: {' '.join([f'{b:02X}' for b in response])}")
                
    except Exception as e:
        print(f"Serial error: {e}")
        return
    
    def update(_):
        nonlocal ser
        if ser and ser.is_open:
            try:
                if ser.in_waiting > 0:
                    data = ser.read(ser.in_waiting)
                    if len(data) > 0:
                        decoder.add_data(data)
                
                points = decoder.get_cartesian_points()
                
                if points is not None and len(points) > 0:
                    # Update status with enhanced information
                    stats = decoder.get_stats()
                    if decoder.last_scan is not None:
                        status = f"Scan #{slam.scan_count}\n"
                        status += f"Speed: {decoder.last_scan['speed']:.1f}°/s\n"
                        status += f"Points: {len(points)}\n"
                        status += f"Quality: {decoder.last_scan['quality']:.2f}\n"
                        status += f"Range: {np.min(decoder.last_scan['distances']):.0f}-{np.max(decoder.last_scan['distances']):.0f}mm\n"
                        status += f"Success: {stats['success_rate']:.1f}%\n"
                        status += f"Position: ({slam.pose[0]:.0f}, {slam.pose[1]:.0f})mm"
                        status_text.set_text(status)
                    
                    # Update SLAM
                    slam.process_scan(points)
                    
                    # Update visualizations
                    scan_plot.set_offsets(points)
                    map_plot.set_data(slam.map.grid_for_display)
                    
                    # Update robot marker and trajectory
                    robot_x = slam.pose[0] / 1000.0
                    robot_y = slam.pose[1] / 1000.0
                    robot_marker.set_data([robot_x], [robot_y])
                    
                    # Update direction arrow
                    rad = np.radians(slam.pose[2])
                    dx, dy = 0.5 * np.sin(rad), 0.5 * np.cos(rad)
                    robot_direction.set_offsets(np.array([[robot_x, robot_y]]))
                    robot_direction.set_UVC([dx], [dy])
                    
                    # Update trajectory
                    if len(slam.pose_history) > 1:
                        traj_x = [p[0]/1000.0 for p in slam.pose_history]
                        traj_y = [p[1]/1000.0 for p in slam.pose_history]
                        trajectory_line.set_data(traj_x, traj_y)
                    
            except Exception as e:
                print(f"Update error: {e}")
                import traceback
                traceback.print_exc()
                
        return scan_plot, map_plot, robot_marker, robot_direction, trajectory_line, status_text
    
    # Start animation with better performance
    ani = FuncAnimation(fig, update, interval=50, blit=True, save_count=200)
    plt.tight_layout()
    plt.show()
    
    if ser and ser.is_open:
        ser.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
# MS200 LIDAR Visualization - No ROS

A real-time LIDAR visualization system for the MS200 sensor without requiring ROS (Robot Operating System).

## Features

- **Real-time polar visualization** of MS200 LIDAR data
- **Automatic packet decoding** with proper MS200 protocol handling
- **Clean dark theme interface** with range circles and status display
- **Distance filtering** (50mm - 8000mm range) with intensity thresholding
- **High-performance visualization** with 50ms update intervals
- **Plug-and-play operation** - MS200 starts automatically after power-on

## Hardware

- MS200 LIDAR sensor
- USB/Serial connection (COM4, 230400 baud)

## Usage

```bash
python sensors/LIDAR/slam.py
```

## Files

- `sensors/LIDAR/slam_with_rpi.py` - Main LIDAR visualization with polar plot using raspberry pi 5
- `sensors/LIDAR/rawdata.py` - Raw data monitoring and debugging tool

## Dependencies

- Python 3.x
- pyserial
- matplotlib
- numpy

Perfect for robotics projects, mapping applications, and LIDAR sensor testing without the complexity of ROS installation.


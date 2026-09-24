# ZED-2i Camera Setup (Orin Nano)

## Prerequisites

* JetPack 7.2 flashed (Ubuntu 24.04 from Nvidia SDK manager) and then ROS 2 Jazzy installed (from apt)
* ZED-2i connected to a USB 3.0 port on the Orin

## Install

1. **ZED SDK 5.4** — download the Jetson/JetPack 7.2 installer from [stereolabs.com/developers/release](https://www.stereolabs.com/developers/release/), then:
   ```bash
   chmod +x ZED_SDK_Installer.run
   ./ZED_SDK_Installer.run
   ```
   Decline the full AI model optimization prompt (Y/n) unless you need every depth/detection mode — it can take hours. The NEURAL depth model needed for normal use is optimized automatically during install.

2. **zed-ros2-wrapper** — built from source (no Jazzy apt binary yet):
   ```bash
   mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
   git clone https://github.com/stereolabs/zed-ros2-wrapper.git
   cd ~/ros2_ws
   rosdep install --from-paths src --ignore-src -r -y
   colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release --parallel-workers $(nproc)
   ```

3. **rosboard** — browser-based topic viewer, self-hosted (no external CDN dependency and eliminates the need for connecting the display to Orin):
   ```bash
   git clone https://github.com/dheera/rosboard.git ~/rosboard
   sudo pip3 install --break-system-packages tornado simplejpeg
   ```
   Two patches are required against the numpy 2.x installed by the ZED SDK's Python API, without them rosboard crashes were observed on `/point_cloud/cloud_registered`:
   ```bash
   # np.nbytes removed in numpy 2.0
   sed -i 's/np\.nbytes\[field_np_datatype\]/np.dtype(field_np_datatype).itemsize/' ~/rosboard/rosboard/compression.py

   # tolerate point cloud frames with zero points after NaN filtering
   sed -i 's/^        ros_msg_dict = ros2dict(msg)$/        try:\n            ros_msg_dict = ros2dict(msg)\n        except Exception as e:\n            self.get_logger().warn("failed to convert message on topic %s: %s" % (topic_name, e))\n            return/' ~/rosboard/rosboard/rosboard.py
   ```

## Visualize

Run [`launch_zed.sh`](launch_zed.sh) to start the camera node and rosboard together (Ctrl-C stops both):
```bash
./launch_zed.sh
```

It passes [`config/cfr_zed2i.yaml`](config/cfr_zed2i.yaml) as `ros_params_override_path`, the race configuration, which pins everything that decides the pose rather than inheriting whatever the installed wrapper defaults to. `formula_one_node` and `lap_counter_node` both drive on `/zed/zed_node/pose` — the **map** frame pose, which the SDK corrects when it closes a loop — so the file pins the grab rate (HD720 at 60 fps, since the grab rate is the tracking rate), the tracking mode (`GEN_3`, which is what `AUTO` resolves to on SDK 5.4), loop closure (`area_memory`), `two_d_mode`, and camera-time stamps. The file explains each one. `jetson/scripts/launch.sh` loads the identical copy installed with `cfr_arduino_bridge`.

Confirm it took, on the car:
```bash
ros2 param get /zed/zed_node pos_tracking.pos_tracking_mode   # GEN_3
ros2 param get /zed/zed_node pos_tracking.area_memory         # true
ros2 topic hz /zed/zed_node/pose                              # ~60 Hz; well below means the Orin cannot keep up
```

`reset_odom_with_loop_closure` is turned **off**, against the wrapper's default. With it on, every loop closure resets `/zed/zed_node/odom` to the origin — a jump as long as the distance travelled since the last reset. With it off, `/odom` is continuous visual-inertial odometry and only `/pose` carries corrections, so a jump in `/odom` means tracking failed and a jump in `/pose` alone means a loop closed.

Or start them manually in separate terminals:

1. Launch the camera node:
   ```bash
   source ~/ros2_ws/install/setup.bash
   ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i \
       ros_params_override_path:=$HOME/cfr/zed/config/cfr_zed2i.yaml   # wherever this repo's zed/ is
   ```
2. In a second terminal, launch rosboard:
   ```bash
   cd ~/rosboard
   source /opt/ros/jazzy/setup.bash
   ./run
   ```

Either way, from a browser on a machine networked to the Orin, go to `http://<orin-ip>:8888` and select a topic (e.g. `/zed/zed_node/point_cloud/cloud_registered` or `/zed/zed_node/rgb/color/rect/image`) to visualize it.

![alt text](image.png)
## ROS 2 Topics

These are all the topics observed from ZED:

| Topic | Description |
| ----- | ----------- |
| `/zed/joint_states` | Joint states for the camera's URDF model |
| `/zed/zed_description` | Robot description (URDF) |
| `/zed/zed_node/depth/camera_info` | Depth camera intrinsics |
| `/zed/zed_node/depth/depth_registered` | Depth image registered to RGB frame |
| `/zed/zed_node/depth/depth_registered/camera_info` | Camera info for registered depth |
| `/zed/zed_node/depth/depth_registered/compressedDepth` | Compressed depth image |
| `/zed/zed_node/depth/depth_registered/zstd` | zstd-compressed depth image |
| `/zed/zed_node/imu/data` | IMU data |
| `/zed/zed_node/odom` | Visual-inertial odometry; never corrected, continuous with `cfr_zed2i.yaml` (reset to the origin on each loop closure without it) |
| `/zed/zed_node/point_cloud/cloud_registered` | Registered colored point cloud |
| `/zed/zed_node/pose` | Camera pose in the map frame, loop closure applied; what `lap_counter_node` reads |
| `/zed/zed_node/pose/status` | Positional tracking status |
| `/zed/zed_node/rgb/color/rect/camera_info` | RGB camera intrinsics |
| `/zed/zed_node/rgb/color/rect/image` | Rectified RGB image |
| `/zed/zed_node/rgb/color/rect/image/camera_info` | Camera info for rectified RGB image |
| `/zed/zed_node/rgb/color/rect/image/compressed` | JPEG-compressed rectified RGB image |
| `/zed/zed_node/rgb/color/rect/image/theora` | Theora-compressed rectified RGB image |
| `/zed/zed_node/rgb/color/rect/image/zstd` | zstd-compressed rectified RGB image |
| `/zed/zed_node/status/health` | Node/camera health status |
| `/zed/zed_node/status/heartbeat` | Node heartbeat |

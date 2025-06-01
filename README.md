# 🧠 ROS 2 Monocular 3D Perception using YOLOv8 and MiDaS

This project implements a complete **monocular 3D perception system** using **ROS 2**, a **Raspberry Pi camera**, and **Edge AI models (YOLOv8 + MiDaS)**.  
The system captures camera data on a Raspberry Pi using ROS 2, streams it to a laptop, performs **object detection**, **depth estimation**, and generates a **live 3D point cloud** — all from a **single camera**.

---

![0601](https://github.com/user-attachments/assets/66a52f97-7b35-4f38-94fe-fe95d9f31631)


## 📦 Packages

- `camera_calibration_pkg` – For intrinsic calibration of the Pi camera.
- `depth_estimation_pkg` – Runs MiDaS on laptop to estimate depth from ROS camera feed.
- `yolo_subscriber` – Performs real-time YOLOv8 object detection on the same feed.

---

## 🖥️ System Architecture

![Blank diagram](https://github.com/user-attachments/assets/b8d6e82e-a5bc-4461-8f3a-511554c09590)

## 📋 Requirements

### On **Raspberry Pi**:
- ROS 2 Humble
- Pi Camera enabled
- `v4l2_camera` or custom camera node (publishing to `/camera/image_raw`)

### On **Laptop**:
- Ubuntu 22.04 or 24.04
- ROS 2 Humble
- Python 3.8+
- [PyTorch](https://pytorch.org/get-started/locally/)
- OpenCV
- Open3D
- cv_bridge

---

## 🛠️ Installation

### 1. Clone the repository:
First, navigate to your ROS 2 workspace and clone the repository into the src directory:
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/Ureed-Hussain/ros2-monocular-depth-yolov8-reconstruction.git
```
### 2. Build your workspace:
```
cd ~/ros2_ws
colcon build
source install/setup.bash
```
### 3. Raspberry Pi Files

Inside the repository, you’ll find a file named Raspberry Pi.zip which contains all necessary files and packages for Raspberry Pi setup, including the cam_pub node that publishes camera data.

Unzip this file and transfer it to your Raspberry Pi.
### 4. Install Python dependencies (on laptop):
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python open3d

```
## 🎥 How to Run

### On Raspberry Pi (camera publisher):
```
ros2 run cam_publisher cam_pub
```
### On Laptop
For yolo:
```
source ~/ros2_ws/install/setup.bash
ros2 run yolo_subscriber yolo_node
```
For MiDaS depth + 3D point cloud
```
source ~/ros2_ws/install/setup.bash
ros2 run depth_estimation_pkg depth_subscriber
```

## 📁 Directory Structure

ros2-yolo-midas-monocular-3d/

├── camera_calibration_pkg/

├── depth_estimation_pkg/

├── yolo_subscriber/

├── README.md

└── assets/ (optional images)

## 🧠 Models Used

    YOLOv8 – Real-time object detection

    MiDaS – Monocular depth estimation

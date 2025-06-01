# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import torch
# import numpy as np

# class DepthSubscriber(Node):
#     def __init__(self):
#         super().__init__('depth_subscriber')

#         self.subscription = self.create_subscription(
#             Image,
#             'camera/image_raw',
#             self.callback,
#             10
#         )
#         self.bridge = CvBridge()

#         # ✅ Load MiDaS small model and transform
#         self.model_type = "MiDaS_small"
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
#         self.model.to(self.device)
#         self.model.eval()

#         midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
#         self.transform = midas_transforms.small_transform

#     def callback(self, msg):
#         # Convert ROS Image message to OpenCV format
#         input_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

#         # Apply MiDaS transform
#         input_batch = self.transform(input_image).to(self.device)

#         # Run inference
#         with torch.no_grad():
#             prediction = self.model(input_batch)
#             prediction = torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=input_image.shape[:2],
#                 mode='bicubic',
#                 align_corners=False,
#             ).squeeze()

#         # Postprocess to visualizable format
#         depth_map = prediction.cpu().numpy()
#         depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
#         depth_display = depth_display.astype(np.uint8)
#         depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_MAGMA)

#         # Display result
#         cv2.imshow('Depth Estimation (MiDaS_small)', depth_display)
#         cv2.waitKey(1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = DepthSubscriber()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()



# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import torch
# import numpy as np

# import open3d as o3d  # <-- Required for point cloud visualization

# # Camera intrinsics (from calibration)
# camera_matrix = np.array([
#     [423.93180248, 0., 283.55355037],
#     [0., 459.90497509, -4.31993347],
#     [0., 0., 1.]
# ])
# dist_coeffs = np.array([-2.51787323, 2.98381778, 0.23537903, -0.20795868, -2.03144261])


# class DepthSubscriber(Node):
#     def __init__(self):
#         super().__init__('depth_subscriber')

#         self.subscription = self.create_subscription(
#             Image,
#             'camera/image_raw',
#             self.callback,
#             10
#         )
#         self.bridge = CvBridge()

#         # ✅ Load MiDaS small model and transform
#         self.model_type = "MiDaS_small"
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
#         self.model.to(self.device)
#         self.model.eval()

#         midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
#         self.transform = midas_transforms.small_transform

#     def callback(self, msg):
#         input_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
#         h, w = input_image.shape[:2]

#         # Undistort image using calibration
#         new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1)
#         undistorted = cv2.undistort(input_image, camera_matrix, dist_coeffs, None, new_camera_matrix)

#         # MiDaS transform
#         input_batch = self.transform(undistorted).to(self.device)

#         # Predict depth
#         with torch.no_grad():
#             prediction = self.model(input_batch)
#             prediction = torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=undistorted.shape[:2],
#                 mode='bicubic',
#                 align_corners=False,
#             ).squeeze()

#         depth_map = prediction.cpu().numpy()

#         # Display depth
#         depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_MAGMA)
#         cv2.imshow('Depth Estimation (MiDaS_small)', depth_display)
#         cv2.waitKey(1)

#         # Generate and visualize point cloud
#         fx, fy = new_camera_matrix[0, 0], new_camera_matrix[1, 1]
#         cx, cy = new_camera_matrix[0, 2], new_camera_matrix[1, 2]

#         pcd = create_point_cloud(depth_map, undistorted, fx, fy, cx, cy)
#         o3d.visualization.draw_geometries([pcd])

# def create_point_cloud(depth_map, rgb_image, fx, fy, cx, cy):
#     h, w = depth_map.shape
#     i, j = np.meshgrid(np.arange(w), np.arange(h))

#     Z = depth_map.flatten()
#     X = (i.flatten() - cx) * Z / fx
#     Y = (j.flatten() - cy) * Z / fy

#     points = np.vstack((X, Y, Z)).T
#     colors = rgb_image.reshape(-1, 3) / 255.0

#     # Mask invalid depth values
#     mask = Z > 0
#     points = points[mask]
#     colors = colors[mask]

#     import open3d as o3d
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     return pcd


# def main(args=None):
#     rclpy.init(args=args)
#     node = DepthSubscriber()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()
#######################################final half##################

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import open3d as o3d

# Camera intrinsics (without using distortion coefficients)
camera_matrix = np.array([
    [423.93180248, 0., 283.55355037],
    [0., 459.90497509, -4.31993347],
    [0., 0., 1.]
])
# ⚠️ Not used:
# dist_coeffs = np.array([...])

class DepthSubscriber(Node):
    def __init__(self):
        super().__init__('depth_subscriber')

        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.callback,
            10
        )
        self.bridge = CvBridge()

        # ✅ Load MiDaS small model and transform
        self.model_type = "MiDaS_small"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

        # 👇 Open3D Visualizer for real-time point cloud updates
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Live 3D Point Cloud")
        self.pcd = o3d.geometry.PointCloud()
        self.added = False

    def callback(self, msg):
        input_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        # MiDaS input transform
        input_batch = self.transform(input_image).to(self.device)

        # Depth prediction
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=input_image.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Show depth image
        depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_MAGMA)
        cv2.imshow('Depth Estimation (MiDaS_small)', depth_display)
        cv2.waitKey(1)

        # Camera intrinsics
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        # Create point cloud from depth and RGB
        pcd = create_point_cloud(depth_map, input_image, fx, fy, cx, cy)

        # Visualize with Open3D (non-blocking)
        if not self.added:
            self.vis.add_geometry(pcd)
            self.pcd = pcd
            self.added = True
        else:
            self.pcd.points = pcd.points
            self.pcd.colors = pcd.colors
            self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()

def create_point_cloud(depth_map, rgb_image, fx, fy, cx, cy):
    h, w = depth_map.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))

    Z = depth_map.flatten()
    X = (i.flatten() - cx) * Z / fx
    Y = -(j.flatten() - cy) * Z / fy  # ✅ Invert Y-axis here

    points = np.vstack((X, Y, Z)).T
    colors = rgb_image.reshape(-1, 3) / 255.0

    # Mask invalid depth values
    mask = Z > 0
    points = points[mask]
    colors = colors[mask]

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def main(args=None):
    rclpy.init(args=args)
    node = DepthSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.vis.destroy_window()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import torch
# import numpy as np
# import open3d as o3d

# # Original camera intrinsics
# original_camera_matrix = np.array([
#     [423.93180248, 0., 283.55355037],
#     [0., 459.90497509, -4.31993347],
#     [0., 0., 1.]
# ])
# original_width = 320  # update if your camera is different
# original_height = 240

# # New lower resolution for speed
# target_width = 256
# target_height = 256
# scale_x = target_width / original_width
# scale_y = target_height / original_height

# # Adjust intrinsics for resized image
# fx = original_camera_matrix[0, 0] * scale_x
# fy = original_camera_matrix[1, 1] * scale_y
# cx = original_camera_matrix[0, 2] * scale_x
# cy = original_camera_matrix[1, 2] * scale_y

# class DepthSubscriber(Node):
#     def __init__(self):
#         super().__init__('depth_subscriber')

#         self.subscription = self.create_subscription(
#             Image,
#             'camera/image_raw',
#             self.callback,
#             qos_profile=10
#         )
#         self.bridge = CvBridge()

#         # ✅ Load MiDaS model
#         self.model_type = "MiDaS_small"
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
#         self.model.to(self.device)
#         self.model.eval()
#         self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

#         # ✅ Open3D Visualizer
#         self.vis = o3d.visualization.Visualizer()
#         self.vis.create_window(window_name="Live 3D Point Cloud", width=640, height=480)
#         self.pcd = o3d.geometry.PointCloud()
#         self.added = False

#     def callback(self, msg):
#         input_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
#         input_image = cv2.resize(input_image, (target_width, target_height))

#         input_batch = self.transform(input_image).to(self.device)

#         with torch.no_grad():
#             prediction = self.model(input_batch)
#             prediction = torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=(target_height, target_width),
#                 mode='bicubic',
#                 align_corners=False,
#             ).squeeze()

#         depth_map = prediction.cpu().numpy()

#         # Show depth
#         depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
#         cv2.imshow("MiDaS Depth", depth_vis)
#         cv2.waitKey(1)

#         # Point cloud
#         pcd = create_point_cloud(depth_map, input_image, fx, fy, cx, cy)

#         if not self.added:
#             self.vis.add_geometry(pcd)
#             self.pcd = pcd
#             self.added = True
#         else:
#             self.pcd.points = pcd.points
#             self.pcd.colors = pcd.colors
#             self.vis.update_geometry(self.pcd)

#         self.vis.poll_events()
#         self.vis.update_renderer()

# def create_point_cloud(depth_map, rgb_image, fx, fy, cx, cy):
#     h, w = depth_map.shape
#     i, j = np.meshgrid(np.arange(w), np.arange(h))

#     Z = depth_map.flatten()
#     X = (i.flatten() - cx) * Z / fx
#     Y = -(j.flatten() - cy) * Z / fy  # Inverted Y-axis

#     points = np.vstack((X, Y, Z)).T
#     colors = rgb_image.reshape(-1, 3) / 255.0

#     # Mask invalid depths
#     mask = Z > 0
#     points = points[mask]
#     colors = colors[mask]

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     return pcd

# def main(args=None):
#     rclpy.init(args=args)
#     node = DepthSubscriber()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.vis.destroy_window()
#         node.destroy_node()
#         rclpy.shutdown()
#         cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()


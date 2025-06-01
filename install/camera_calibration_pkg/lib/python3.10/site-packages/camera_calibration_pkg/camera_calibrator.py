import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

class CameraCalibrator(Node):
    def __init__(self):
        super().__init__('camera_calibrator')
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.callback,
            10
        )
        self.bridge = CvBridge()
        self.image_save_path = os.path.expanduser('~/calibration_images')
        os.makedirs(self.image_save_path, exist_ok=True)
        self.img_count = 0
        self.get_logger().info(f"Saving images to: {self.image_save_path}")

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.image_save_path, f"calib_{timestamp}.png")
        cv2.imwrite(filename, frame)
        self.img_count += 1
        self.get_logger().info(f"Saved image {self.img_count}: {filename}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

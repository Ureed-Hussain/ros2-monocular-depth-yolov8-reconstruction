import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
from ultralytics import YOLO

class YoloSubscriber(Node):
    def __init__(self):
        super().__init__('yolo_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.listener_callback,
            10)
        self.bridge = CvBridge()

        # Load smallest YOLOv8 model and move to GPU if available
        self.model = YOLO('yolov8n.pt')
        if torch.cuda.is_available():
            self.get_logger().info('Using GPU for YOLOv8')
            self.model.to('cuda')
        else:
            self.get_logger().info('Using CPU for YOLOv8')

    def listener_callback(self, msg):
        # Convert ROS image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Resize for faster inference
        resized_frame = cv2.resize(frame, (256, 256))

        # Run YOLOv8 with confidence threshold
        results = self.model(resized_frame, verbose=False, conf=0.5)[0]

        # Draw detections on original frame
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = self.model.names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow('YOLOv8 Detection', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

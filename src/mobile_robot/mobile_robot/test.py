import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class TestPublisher(Node):
    def __init__(self):
        super().__init__('test_node')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 0.5  # seconds (2 Hz)
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        controlVel = Twist()
        controlVel.linear.x = 4.0
        controlVel.angular.z = 8.0
        self.get_logger().info("Publishing control velocity")
        self.publisher_.publish(controlVel)


def main(args=None):
    rclpy.init(args=args)
    node = TestPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()





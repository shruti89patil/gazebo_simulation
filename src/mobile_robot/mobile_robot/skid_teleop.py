import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import sys, termios, tty

# Distance between left and right wheels
WHEEL_SEPARATION = 0.7  # meters
# Maximum wheel speed
MAX_SPEED = 2.0  # adjust if needed

class SkidSteerTeleop(Node):
    def __init__(self):
        super().__init__('skid_teleop')

        # Publishers for each wheel velocity
        self.pub_wheel1 = self.create_publisher(Float64, '/wheel1_joint_velocity_controller/commands', 10)
        self.pub_wheel2 = self.create_publisher(Float64, '/wheel2_joint_velocity_controller/commands', 10)
        self.pub_wheel3 = self.create_publisher(Float64, '/wheel3_joint_velocity_controller/commands', 10)
        self.pub_wheel4 = self.create_publisher(Float64, '/wheel4_joint_velocity_controller/commands', 10)

    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    def run(self):
        print("Use WASD keys to drive (q to quit):")
        linear = 0.0
        angular = 0.0

        while rclpy.ok():
            key = self.get_key()
            if key == 'w':
                linear = MAX_SPEED
                angular = 0.0
            elif key == 's':
                linear = -MAX_SPEED
                angular = 0.0
            elif key == 'a':
                linear = 0.0
                angular = MAX_SPEED
            elif key == 'd':
                linear = 0.0
                angular = -MAX_SPEED
            elif key == 'q':
                break
            else:
                linear = 0.0
                angular = 0.0

            # Compute left/right wheel speeds
            v_left = linear - (angular * WHEEL_SEPARATION / 2.0)
            v_right = linear + (angular * WHEEL_SEPARATION / 2.0)

            # Publish to all wheels
            self.pub_wheel1.publish(Float64(data=v_right))
            self.pub_wheel2.publish(Float64(data=v_left))
            self.pub_wheel3.publish(Float64(data=v_right))
            self.pub_wheel4.publish(Float64(data=v_left))


def main(args=None):
    rclpy.init(args=args)
    node = SkidSteerTeleop()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

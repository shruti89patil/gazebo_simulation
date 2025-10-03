# import numpy as np
# import math
# import time
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist
# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import LaserScan
# from tf_transformations import euler_from_quaternion


# topic1 = '/cmd_vel'
# topic2 = '/odom'
# topic3 = '/scan'


# class ControllerNode(Node):

#     def __init__(self, xdu, ydu, kau, kru, kthetau, gstaru, eps_orientu, eps_controlu):
#         super().__init__('controller_node')

#         # Goal position
#         self.xdp = xdu
#         self.ydp = ydu

#         # Control parameters
#         self.kap = kau
#         self.krp = kru
#         self.kthetap = kthetau
#         self.gstarp = gstaru
#         self.eps_orient = eps_orientu
#         self.eps_control = eps_controlu

#         # Messages
#         self.OdometryMsg = Odometry()
#         self.LidarMsg = LaserScan()
        
#         # New flag to ensure Odometry is received before starting control loop
#         self.odometry_received = False

#         # Timing
#         self.initialTime = time.time()
#         self.msgOdometryTime = time.time()
#         self.msgLidarTime = time.time()

#         # Control velocities
#         self.controlVel = Twist()
#         self.controlVel.linear.x = 0.0
#         self.controlVel.linear.y = 0.0
#         self.controlVel.linear.z = 0.0
#         self.controlVel.angular.x = 0.0
#         self.controlVel.angular.y = 0.0
#         self.controlVel.angular.z = 0.0

#         # ROS publishers/subscribers
#         self.ControlPublisher = self.create_publisher(Twist, topic1, 10)
#         # PoseSubscriber is crucial for setting the initial position
#         self.PoseSubscriber = self.create_subscription(Odometry, topic2, self.SensorCallbackPose, 10)
#         self.LidarSubscriber = self.create_subscription(LaserScan, topic3, self.SensorCallbackLidar, 10)

#         # Timer for control loop. It is initially set to None and will be created
#         # once the first Odometry message is received in SensorCallbackPose.
#         self.period = 0.05
#         self.timer = None

    

#     # ------------------------------------------------
#     # Orientation Error Calculation
#     # ------------------------------------------------
#     def orientationError(self, theta_, thetad_):
#         if (thetad_ > np.pi/2) and (thetad_ <= np.pi):
#             if (theta_ > -np.pi) and (theta_ <= -np.pi/2):
#                 theta_ = theta_ + 2*np.pi

#         if (theta_ > np.pi/2) and (theta_ <= np.pi):
#             if (thetad_ > -np.pi) and (thetad_ <= -np.pi/2):
#                 thetad_ = thetad_ + 2*np.pi

#         errorOrientation = thetad_ - theta_
#         return errorOrientation

#     # ------------------------------------------------
#     # Pose Callback (Updated to start the timer)
#     # ------------------------------------------------
#     def SensorCallbackPose(self, receivedMsg):
#         self.OdometryMsg = receivedMsg
#         self.msgOdometryTime = time.time()
        
#         # Start the control loop timer only after the first Odometry message is processed
#         if not self.odometry_received:
#             self.odometry_received = True
#             self.get_logger().info('First Odometry message received! Starting control timer.')
#             self.timer = self.create_timer(self.period, self.ControlFunction)

#     # ------------------------------------------------
#     # Lidar Callback
#     # ------------------------------------------------
#     def SensorCallbackLidar(self, receivedMsg):
#         self.LidarMsg = receivedMsg
#         self.msgLidarTime = time.time()

#     # ------------------------------------------------
#     # Main Control Function
#     # ------------------------------------------------
#     # def ControlFunction(self):
#     #     ka = self.kap
#     #     kr = self.krp
#     #     ktheta = self.kthetap
#     #     gstar = self.gstarp
#     #     xd = self.xdp
#     #     yd = self.ydp

#     #     # Current position
#     #     x = self.OdometryMsg.pose.pose.position.x
#     #     y = self.OdometryMsg.pose.pose.position.y

#     #     # Orientation from quaternion
#     #     quat = self.OdometryMsg.pose.pose.orientation
#     #     quat_list = [quat.x, quat.y, quat.z, quat.w]
#     #     (roll, pitch, theta) = euler_from_quaternion(quat_list)

#     #     # Desired vector
#     #     vectorD = np.array([[x - xd], [y - yd]])
#     #     gradUa = ka * vectorD
#     #     AF = -gradUa # Attractive force

#     #     # Process Lidar data
#     #     LidarRanges = np.array(self.LidarMsg.ranges)
#     #     angle_min = self.LidarMsg.angle_min
#     #     angle_increment = self.LidarMsg.angle_increment

#     #     indices_not_inf = np.where(~np.isinf(LidarRanges))[0]
#     #     obstacleYES = indices_not_inf.size > 0

#     #     RF = np.array([[0], [0]]) # Repulsive force

#     #     if obstacleYES:
#     #         # Angles of valid lidar beams
#     #         angles = angle_min + indices_not_inf * angle_increment + theta
#     #         distances = LidarRanges[indices_not_inf]

#     #         # Convert to world coordinates
#     #         xo = x + distances * np.cos(angles)
#     #         yo = y + distances * np.sin(angles)

#     #         # Find closest obstacles in clusters
#     #         min_distances = []
#     #         min_distances_angles = []

#     #         diff_array = np.diff(indices_not_inf)
#     #         split_indices = np.where(diff_array > 1)[0] + 1
#     #         partitioned_arrays = np.split(indices_not_inf, split_indices)

#     #         for part in partitioned_arrays:
#     #             tmpArray = LidarRanges[part]
#     #             min_index = np.argmin(tmpArray)
#     #             min_distances.append(tmpArray[min_index])
#     #             min_distances_angles.append(angle_min + angle_increment * part[min_index])

#     #         # Repulsive forces
#     #         for i in range(len(min_distances)):
#     #             g_val = np.sqrt((x - (x + min_distances[i] * np.cos(min_distances_angles[i] + theta)))**2 +
#     #                              (y - (y + min_distances[i] * np.sin(min_distances_angles[i] + theta)))**2)

#     #             if g_val <= gstar:
#     #                 pr = kr * ((1/gstar) - (1/g_val)) * (1 / (g_val**3))
#     #                 gradUr_i = pr * np.array([[x - (x + min_distances[i] * np.cos(min_distances_angles[i] + theta))],
#     #                                          [y - (y + min_distances[i] * np.sin(min_distances_angles[i] + theta))]])
#     #                 RF += gradUr_i

#     #         RF = -RF # Repulsive force

#     #     # Total force
#     #     if obstacleYES:
#     #         F = AF + RF
#     #     else:
#     #         F = AF

#     #     thetaD = math.atan2(F[1, 0], F[0, 0])
#     #     eorient = self.orientationError(theta, thetaD)

#     #     # # Control law
#     #     # if np.linalg.norm(vectorD, 2) < self.eps_control:
#     #     #     thetavel = 0.0
#     #     #     xvel = 0.0
#     #             # Control law

#     #     rho_goal = np.linalg.norm(vectorD, 2)
#     #     if rho_goal < self.eps_control: # <--- Use the clear distance variable
#     #         thetavel = 0.0
#     #         xvel = 0.0
#     #         if not self.goal_reached:
#     #             self.goal_reached = True
                
#     #             # 1. Publish Zero Velocity Command
#     #             self.controlVel.linear.x = 0.0
#     #             self.controlVel.angular.z = 0.0
#     #             self.ControlPublisher.publish(self.controlVel)
                
#     #             # 2. Log Termination
#     #             self.get_logger().info('Goal Reached! Stopping control timer and locking velocity at zero.')
                
#     #             # 3. Stop Control Timer
#     #             if self.timer is not None:
#     #                 self.timer.cancel()# Log only once
#     #                 return
#     #     else:
#     #         self.goal_reached = False
#     #         if abs(eorient) > self.eps_orient:
#     #             thetavel = ktheta * eorient
#     #             xvel = 0.0
#     #         else:
#     #             thetavel = ktheta * eorient
#     #             xvel = np.linalg.norm(F, 2)
#     #             if abs(xvel) > 2.6:
#     #                 xvel = 2.5

#     #     # Send command
#     #     self.controlVel.linear.x = xvel
#     #     self.controlVel.angular.z = thetavel

#     #     self.ControlPublisher.publish(self.controlVel)

#     #     # Debug info
#     #     print("Sending the control command")
#     #     timeDiff = self.msgOdometryTime - self.initialTime
#     #     print(f"Time, x, y, theta: ({timeDiff:.3f}, {x:.3f}, {y:.3f}, {theta:.3f})")


# # ------------------------------------------------
#     # Main Control Function (Corrected Logic Flow)
#     # ------------------------------------------------
#     def ControlFunction(self):
#         ka = self.kap
#         kr = self.krp
#         ktheta = self.kthetap
#         gstar = self.gstarp
#         xd = self.xdp
#         yd = self.ydp

#         # Check for Goal Reached (Termination Condition)
#         x = self.OdometryMsg.pose.pose.position.x
#         y = self.OdometryMsg.pose.pose.position.y
#         vectorD = np.array([[x - xd], [y - yd]])
#         rho_goal = np.linalg.norm(vectorD, 2)

#         if rho_goal < self.eps_control:
#             if not self.goal_reached:
#                 self.goal_reached = True
                
#                 # 1. Publish Zero Velocity Command
#                 self.controlVel.linear.x = 0.0
#                 self.controlVel.angular.z = 0.0
#                 self.ControlPublisher.publish(self.controlVel)
                
#                 # 2. Stop Control Timer
#                 if self.timer is not None:
#                     self.timer.cancel()
                    
#                 # 3. Log Termination
#                 self.get_logger().info('Goal Reached! Stopping control timer and locking velocity at zero.')
#             return # Exit the function if goal reached

#         # --- ARTIFICIAL POTENTIAL FIELD CALCULATION ---
        
#         # Orientation from quaternion
#         quat = self.OdometryMsg.pose.pose.orientation
#         quat_list = [quat.x, quat.y, quat.z, quat.w]
#         (roll, pitch, theta) = euler_from_quaternion(quat_list)
        
#         # Attractive Force (AF)
#         gradUa = ka * vectorD
#         AF = -gradUa 

#         # Process Lidar data
#         LidarRanges = np.array(self.LidarMsg.ranges)
#         angle_min = self.LidarMsg.angle_min
#         angle_increment = self.LidarMsg.angle_increment

#         indices_not_inf = np.where(~np.isinf(LidarRanges))[0]
#         obstacleYES = indices_not_inf.size > 0

#         RF = np.array([[0.0], [0.0]]) # Repulsive force

#         if obstacleYES:
#             # The angles array needs to be relative to the robot's current heading (theta)
#             # The angles in the LidarMsg are relative to the sensor frame.
            
#             # Angles of valid lidar beams (relative to world frame)
#             angles_world = angle_min + indices_not_inf * angle_increment + theta 
#             distances = LidarRanges[indices_not_inf]

#             # Find closest obstacles in clusters (This logic is correct for APF)
#             min_distances = []
#             min_distances_angles = []

#             diff_array = np.diff(indices_not_inf)
#             split_indices = np.where(diff_array > 1)[0] + 1
#             partitioned_arrays = np.split(indices_not_inf, split_indices)

#             for part in partitioned_arrays:
#                 tmpArray = LidarRanges[part]
#                 min_index = np.argmin(tmpArray)
#                 min_distances.append(tmpArray[min_index])
#                 # The angle for RF calculation must be relative to the world frame
#                 min_distances_angles.append(angle_min + angle_increment * part[min_index] + theta) 

#             # Repulsive forces
#             for i in range(len(min_distances)):
#                 # x_obs and y_obs in world coordinates
#                 x_obs = x + min_distances[i] * np.cos(min_distances_angles[i])
#                 y_obs = y + min_distances[i] * np.sin(min_distances_angles[i])
                
#                 # g_val is the distance from robot (x, y) to the obstacle (x_obs, y_obs)
#                 # This calculation can be simplified since min_distances[i] is already g_val
#                 g_val = min_distances[i] 

#                 if g_val <= gstar:
#                     # Potential function derivative
#                     pr = kr * ((1/gstar) - (1/g_val)) * (1 / (g_val**2)) # Corrected power for r_dot
                    
#                     # Direction vector from obstacle to robot (x_obs -> x)
#                     direction_vector = np.array([[x - x_obs], [y - y_obs]])
                    
#                     # GradUr_i is pr * normalized(direction_vector)
#                     gradUr_i = pr * direction_vector / g_val 
                    
#                     RF += gradUr_i
            
#             # RF is the repulsive force
#             RF = -RF 
            
#         # Total force
#         F = AF + RF
        
#         # Calculate desired orientation (thetaD) and linear velocity (xvel)
#         thetaD = math.atan2(F[1, 0], F[0, 0])
#         eorient = self.orientationError(theta, thetaD)
        
#         # --- CONTROL LAW EXECUTION ---

#         if abs(eorient) > self.eps_orient:
#             # Only rotate to align with the desired force vector F
#             thetavel = ktheta * eorient
#             xvel = 0.0
#         else:
#             # Rotate and move forward in the direction of F
#             thetavel = ktheta * eorient
#             xvel = np.linalg.norm(F, 2)
            
#             # Clamp maximum linear velocity
#             if abs(xvel) > 2.6:
#                 xvel = 2.5
        
#         # Send command
#         self.controlVel.linear.x = xvel
#         self.controlVel.angular.z = thetavel

#         self.ControlPublisher.publish(self.controlVel)

#         # Debug info
#         self.get_logger().info(f"Time, x, y, theta: ({self.msgOdometryTime - self.initialTime:.3f}, {x:.3f}, {y:.3f}, {theta:.3f}) | Fx:{F[0, 0]:.2f}, Fy:{F[1, 0]:.2f} | V:{xvel:.2f}, W:{thetavel:.2f}")

# # ------------------------------------------------
# # Main Entry Point
# # ------------------------------------------------
# def main(args=None):
#     rclpy.init(args=args)

#     xd_u = 10
#     yd_u = -10
#     ka_u = 0.3
#     kr_u = 10
#     ktheta_u = 4
#     gstar_u = 4.0
#     eps_orient_u = np.pi / 10
#     eps_control_u = 0.2

#     testNode = ControllerNode(xd_u, yd_u, ka_u, kr_u, ktheta_u, gstar_u, eps_orient_u, eps_control_u)
#     rclpy.spin(testNode)

#     testNode.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()

import numpy as np
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion


topic1 = '/cmd_vel'
topic2 = '/odom'
topic3 = '/scan'


class ControllerNode(Node):

    def __init__(self, xdu, ydu, kau, kru, kthetau, gstaru, eps_orientu, eps_controlu):
        super().__init__('controller_node')

        # Goal position
        self.xdp = xdu
        self.ydp = ydu

        # Control parameters (TUNED FOR OBSTACLE AVOIDANCE)
        self.kap = kau       # Attractive Gain (Reduced from 0.3)
        self.krp = kru       # Repulsive Gain (Increased from 10)
        self.kthetap = kthetau
        self.gstarp = gstaru # Influence Radius (Kept at 4.0)
        self.eps_orient = eps_orientu
        self.eps_control = eps_controlu
        self.goal_reached = False # State flag for termination

        # Messages
        self.OdometryMsg = Odometry()
        self.LidarMsg = LaserScan()
        
        self.odometry_received = False

        # Timing
        self.initialTime = time.time()
        self.msgOdometryTime = time.time()
        self.msgLidarTime = time.time()

        # Control velocities
        self.controlVel = Twist()

        # ROS publishers/subscribers
        self.ControlPublisher = self.create_publisher(Twist, topic1, 10)
        self.PoseSubscriber = self.create_subscription(Odometry, topic2, self.SensorCallbackPose, 10)
        self.LidarSubscriber = self.create_subscription(LaserScan, topic3, self.SensorCallbackLidar, 10)

        self.period = 0.05
        self.timer = None

    
    # ------------------------------------------------
    # Orientation Error Calculation (ROBUST)
    # ------------------------------------------------
    def orientationError(self, theta_, thetad_):
        """Calculates the shortest angular difference in the range (-pi, pi]"""
        errorOrientation = thetad_ - theta_
        if errorOrientation > np.pi:
            errorOrientation -= 2 * np.pi
        elif errorOrientation < -np.pi:
            errorOrientation += 2 * np.pi
        return errorOrientation

    # ------------------------------------------------
    # Pose Callback
    # ------------------------------------------------
    def SensorCallbackPose(self, receivedMsg):
        self.OdometryMsg = receivedMsg
        self.msgOdometryTime = time.time()
        
        # Start the control loop timer only after the first Odometry message
        if not self.odometry_received:
            self.odometry_received = True
            self.get_logger().info('First Odometry message received! Starting control timer.')
            self.timer = self.create_timer(self.period, self.ControlFunction)

    # ------------------------------------------------
    # Lidar Callback
    # ------------------------------------------------
    def SensorCallbackLidar(self, receivedMsg):
        self.LidarMsg = receivedMsg
        self.msgLidarTime = time.time()

    # ------------------------------------------------
    # Main Control Function (APF with Corrections)
    # ------------------------------------------------
    def ControlFunction(self):
        ka = self.kap
        kr = self.krp
        ktheta = self.kthetap
        gstar = self.gstarp
        xd = self.xdp
        yd = self.ydp

        # Get Current Position and Orientation
        x = self.OdometryMsg.pose.pose.position.x
        y = self.OdometryMsg.pose.pose.position.y
        quat = self.OdometryMsg.pose.pose.orientation
        quat_list = [quat.x, quat.y, quat.z, quat.w]
        (_, _, theta) = euler_from_quaternion(quat_list)
        
        vectorD = np.array([[x - xd], [y - yd]])
        rho_goal = np.linalg.norm(vectorD, 2)

        # --- Check for Goal Reached (Termination) ---
        if rho_goal < self.eps_control:
            if not self.goal_reached:
                self.goal_reached = True
                self.controlVel.linear.x = 0.0
                self.controlVel.angular.z = 0.0
                self.ControlPublisher.publish(self.controlVel)
                
                if self.timer is not None:
                    self.timer.cancel()
                    
                self.get_logger().info('Goal Reached! Stopping control. 🏁')
            return 

        self.goal_reached = False
        
        # --- ARTIFICIAL POTENTIAL FIELD CALCULATION ---
        
        # 1. Attractive Force (AF)
        gradUa = ka * vectorD
        AF = -gradUa 
        AF.shape = (2, 1)

        # 2. Repulsive Force (RF)
        LidarRanges = np.array(self.LidarMsg.ranges)
        angle_min = self.LidarMsg.angle_min
        angle_increment = self.LidarMsg.angle_increment

        indices_not_inf = np.where(~np.isinf(LidarRanges))[0]
        obstacleYES = indices_not_inf.size > 0

        RF = np.array([[0.0], [0.0]]) # Initialize repulsive force

        if obstacleYES:
            
            # --- Obstacle Clustering and Finding Closest Points ---
            min_distances = []
            min_distances_angles_robot_frame = []

            # Partition Lidar indices based on gaps (clustering)
            diff_array = np.diff(indices_not_inf)
            split_indices = np.where(diff_array > 1)[0] + 1
            partitioned_arrays = np.split(indices_not_inf, split_indices)

            for part in partitioned_arrays:
                tmpArray = LidarRanges[part]
                min_index = np.argmin(tmpArray)
                min_distances.append(tmpArray[min_index])
                # Angle relative to the robot's frame
                min_distances_angles_robot_frame.append(angle_min + angle_increment * part[min_index]) 

            # --- Sum Repulsive Forces (F_R = -Grad(U_r)) ---
            for i in range(len(min_distances)):
                g_val = min_distances[i] # Distance to the closest point of cluster
                
                if g_val <= gstar:
                    
                    # 1. Magnitude of Repulsive Potential Derivative (pr)
                    # Correct APF formula uses 1/g^2 for distance weighting
                    pr = kr * ((1/gstar) - (1/g_val)) * (1 / (g_val**2)) 
                    
                    # 2. Obstacle Position in World Coordinates
                    angle_world = min_distances_angles_robot_frame[i] + theta
                    x_obs = x + g_val * np.cos(angle_world)
                    y_obs = y + g_val * np.sin(angle_world)
                    
                    # 3. Direction vector (Obstacle -> Robot)
                    # This is the direction of the potential gradient (Grad U_r)
                    direction_vector_obs_to_robot = np.array([[x - x_obs], [y - y_obs]])
                    
                    # 4. Repulsive Gradient (Grad U_r)
                    gradUr_i = pr * direction_vector_obs_to_robot / g_val 
                    
                    RF += gradUr_i
            
            # Repulsive Force F_R is the negative of the gradient
            RF = -RF 
            
        # 3. Total Force
        F = AF + RF
        
        # --- CONTROL LAW EXECUTION ---

        # Calculate desired orientation (thetaD) based on total force
        thetaD = math.atan2(F[1, 0], F[0, 0])
        eorient = self.orientationError(theta, thetaD)
        
        if abs(eorient) > self.eps_orient:
            # Only rotate to align with F (linear velocity = 0)
            thetavel = ktheta * eorient
            xvel = 0.0
        else:
            # Move forward in the direction of F
            thetavel = ktheta * eorient
            xvel = np.linalg.norm(F, 2)
            
            # Clamp maximum linear velocity (Tuned for safer avoidance)
            MAX_V = 1.0 
            if abs(xvel) > MAX_V:
                xvel = MAX_V
        
        # Send command
        self.controlVel.linear.x = xvel
        self.controlVel.angular.z = thetavel

        self.ControlPublisher.publish(self.controlVel)

        # Debug info
        self.get_logger().info(f"Time: {self.msgOdometryTime - self.initialTime:.2f}s | Pose: ({x:.2f}, {y:.2f}, {theta:.2f}) | Fx:{F[0, 0]:.2f}, Fy:{F[1, 0]:.2f} | V:{xvel:.2f}, W:{thetavel:.2f}")

# ------------------------------------------------
# Main Entry Point
# ------------------------------------------------
def main(args=None):
    rclpy.init(args=args)

    # FINAL TUNED PARAMETERS:
    xd_u = 10
    yd_u = -10
    ka_u = 0.15      # Attractive Gain (Reduced for better avoidance)
    kr_u = 50.0      # Repulsive Gain (Increased for better avoidance)
    ktheta_u = 4
    gstar_u = 4.0    # Influence radius for obstacles
    eps_orient_u = np.pi / 10
    eps_control_u = 0.2

    testNode = ControllerNode(xd_u, yd_u, ka_u, kr_u, ktheta_u, gstar_u, eps_orient_u, eps_control_u)
    rclpy.spin(testNode)

    testNode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
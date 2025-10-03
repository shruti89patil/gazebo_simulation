
import os
import xacro
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    robotXacroName = 'differential_drive_robot'
    namepackage = 'mobile_robot'
    modelFileRelativePath = 'model/robot.xacro'

    # Path to URDF
    pathModelFile = os.path.join(
        get_package_share_directory(namepackage),
        modelFileRelativePath
    )

    # Convert xacro → xml string
    robotDescription = xacro.process_file(pathModelFile).toxml()

    # Include Gazebo launch file
    gazebo_rosPackageLaunch = PythonLaunchDescriptionSource(
        os.path.join(
            get_package_share_directory('ros_gz_sim'),
            'launch', 'gz_sim.launch.py'
        )
    )

    world_file_name = 'my_world.sdf'  # <--- RENAME THIS TO YOUR FILE
    world_path = os.path.join(
        get_package_share_directory(namepackage),
        'worlds',
        world_file_name
    )

    gazeboLaunch = IncludeLaunchDescription(
        gazebo_rosPackageLaunch,
        launch_arguments={
            'gz_args': [f'-r -v -v4 {world_path}'],
            # 'gz_args': [f'-r -v -v4 empty.sdf'],
            'on_exit_shutdown': 'true'
        }.items(),
    )

    # Gazebo spawn node
    spawnModelNodeGazebo = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', robotXacroName,
            '-topic', 'robot_description',
        ],
        output='screen',
    )

    # Robot State Publisher Node
    nodeRobotStatePublisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robotDescription,
            'use_sim_time': True
        }]
    )

    # Load wheel controllers
    # wheel_controllers_file = os.path.join(
    #     get_package_share_directory('mobile_robot'),
    #     'parameters',
    #     'wheel_controllers.yaml'
    # )

    # load_controllers = Node(
    #     package="controller_manager",
    #     executable="spawner",
    #     arguments=[
    #         "wheel1_joint_velocity_controller",
    #         "wheel2_joint_velocity_controller",
    #         # "wheel3_joint_velocity_controller",
    #         # "wheel4_joint_velocity_controller",
    #         "--param-file", wheel_controllers_file
    #     ],
    #     output="screen",
    # )


    # Bridge for ROS2 <-> Gazebo topics
    bridge_params = os.path.join(
        get_package_share_directory(namepackage),
        'parameters',
        'bridge_parameters.yaml'
    )

    start_gazebo_ros_bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='parameter_bridge',
        output='screen',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan'
            # '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan'

    ]
)


    # Launch Description object
    launchDescriptionObject = LaunchDescription()
    launchDescriptionObject.add_action(gazeboLaunch)
    launchDescriptionObject.add_action(spawnModelNodeGazebo)
    launchDescriptionObject.add_action(nodeRobotStatePublisher)
    # launchDescriptionObject.add_action(load_controllers)
    launchDescriptionObject.add_action(start_gazebo_ros_bridge_cmd)

    return launchDescriptionObject

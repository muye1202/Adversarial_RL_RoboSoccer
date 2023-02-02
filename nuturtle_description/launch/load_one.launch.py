from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown, SetLaunchConfiguration
from launch.conditions import LaunchConfigurationEquals
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, Command, \
                                 PathJoinSubstitution, TextSubstitution
from launch_ros.substitutions import FindPackageShare, ExecutableInPackage


def generate_launch_description():
    urdf_path = FindPackageShare('nuturtle_description')
    default_model_path = PathJoinSubstitution([urdf_path, 'urdf/turtlebot3_burger.urdf.xacro'])

    return LaunchDescription([
        DeclareLaunchArgument(name='use_jsp', default_value='true',
                              choices=['true', 'false'],
                              description='determine how to publish joint states'),

        DeclareLaunchArgument(name='use_rviz', default_value='true',
                              choices=['true', 'false'],
                              description='determine whether to use rviz'),

        DeclareLaunchArgument(name='use_custom_rviz', default_value='false',
                              choices=['true', 'false'],
                              description='determine whether to use custom Rviz files'),

        # parent frame id
        DeclareLaunchArgument(name='world_frame', default_value='nusim/world',
                              description="name of the world frame"),

        DeclareLaunchArgument(name='xpos', default_value='0',
                              description="robot starting x position"),

        DeclareLaunchArgument(name='ypos', default_value='0',
                              description="robot starting y position"),

        DeclareLaunchArgument(name='zpos', default_value='0',
                              description="robot starting z position"),

        # declare color argument
        DeclareLaunchArgument(name='robot_color', default_value='purple',
                              choices=['purple', 'red', 'green', 'blue', ''],
                              description='determine the robot color'),

        SetLaunchConfiguration(name='rvizconfig', value=[urdf_path,
                                                         TextSubstitution(text='/rviz/robot_player.rviz')]),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace=LaunchConfiguration('robot_color'),
            parameters=[{
                'frame_prefix': PathJoinSubstitution([LaunchConfiguration('robot_color'), ' ']),
                'robot_description': Command([ExecutableInPackage('xacro', 'xacro'), ' ',
                                              default_model_path,
                                              ' color:=',
                                              LaunchConfiguration('robot_color')])}]
        ),

        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            namespace=LaunchConfiguration('robot_color'),
            condition=LaunchConfigurationEquals('use_jsp', 'true')
        ),

        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            namespace=LaunchConfiguration('robot_color'),
            condition=LaunchConfigurationEquals('use_jsp', 'false')
        ),

        # access rviz config files corresponding to different colors
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', LaunchConfiguration('rvizconfig')],
            namespace=LaunchConfiguration('robot_color'),
            condition=LaunchConfigurationEquals('use_rviz', 'true'),
            on_exit=Shutdown()   # Shut down launched system upon closing rviz.
        ),

        # publish static frame nusim/world
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     arguments=['--frame-id', LaunchConfiguration('world_frame'),
        #                '--child-frame-id', [LaunchConfiguration('robot_color'),
        #                                     '/base_footprint'],
        #                '--x', LaunchConfiguration('xpos'),
        #                '--y', LaunchConfiguration('ypos'),
        #                '--z', LaunchConfiguration('zpos')]
        # )
    ])

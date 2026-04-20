from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'serial',
            default_value='/dev/pts/4'
        ),
        Node(
            package='homework_bringup',
            executable='shooter',
            name='shooter',
            output='screen',
            parameters=[
                {'serial_port': LaunchConfiguration('serial')}
            ]
        )
    ])
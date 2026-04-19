from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('serial_num',default_value='0'),
        Node(
            package='simple_shooter',
            executable='shooter',
            name='shooter',
            output='screen',
            parameters=[
                {'serial_num': LaunchConfiguration('serial_num')}
            ]
        )
    ])

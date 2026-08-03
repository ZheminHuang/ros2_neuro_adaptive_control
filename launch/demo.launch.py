"""Launch the deterministic unknown-dynamics Cartesian NAC demo."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Create controller and plant processes with shared experiment settings."""
    config_file = PathJoinSubstitution(
        [FindPackageShare("neuro_adaptive_control"), "config", "default.yaml"]
    )
    reference_type = LaunchConfiguration("reference_type")
    adaptation_enabled = LaunchConfiguration("adaptation_enabled")
    external_wrench_enabled = LaunchConfiguration("external_wrench_enabled")
    duration_sec = LaunchConfiguration("duration_sec")
    control_rate_hz = LaunchConfiguration("control_rate_hz")
    output_directory = LaunchConfiguration("output_directory")

    arguments = [
        DeclareLaunchArgument(
            "reference_type",
            default_value="circle",
            description="circle, line, figure8, or fixed_point",
        ),
        DeclareLaunchArgument(
            "adaptation_enabled",
            default_value="true",
            description="Enable online RBF output-weight adaptation",
        ),
        DeclareLaunchArgument(
            "external_wrench_enabled",
            default_value="false",
            description="Apply the deterministic optional physical wrench",
        ),
        DeclareLaunchArgument(
            "duration_sec",
            default_value="12.0",
            description="Fixed simulated duration before an automatic zero stop",
        ),
        DeclareLaunchArgument(
            "control_rate_hz",
            default_value="500.0",
            description="Fixed-step target rate; not a hard real-time guarantee",
        ),
        DeclareLaunchArgument(
            "output_directory",
            default_value="",
            description="Optional directory for ROS demo metrics JSON",
        ),
    ]

    shared = {
        "control_rate_hz": ParameterValue(control_rate_hz, value_type=float),
        "duration_sec": ParameterValue(duration_sec, value_type=float),
    }
    controller = Node(
        package="neuro_adaptive_control",
        executable="nac_controller_node",
        name="nac_controller",
        output="screen",
        emulate_tty=True,
        parameters=[
            config_file,
            shared,
            {
                "trajectory.type": reference_type,
                "rbf.adaptation_enabled": ParameterValue(
                    adaptation_enabled, value_type=bool
                ),
            },
        ],
    )
    plant = Node(
        package="neuro_adaptive_control",
        executable="cartesian_demo_plant",
        name="cartesian_demo_plant",
        output="screen",
        emulate_tty=True,
        parameters=[
            config_file,
            shared,
            {
                "external_wrench_enabled": ParameterValue(
                    external_wrench_enabled, value_type=bool
                ),
                "output_directory": output_directory,
            },
        ],
    )

    stop_on_controller_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=controller,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason="NAC controller process exited")
                )
            ],
        )
    )
    stop_on_plant_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=plant,
            on_exit=[
                EmitEvent(event=Shutdown(reason="demo plant process exited"))
            ],
        )
    )
    return LaunchDescription(
        arguments
        + [controller, plant, stop_on_controller_exit, stop_on_plant_exit]
    )

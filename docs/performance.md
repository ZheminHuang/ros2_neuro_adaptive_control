# 500 Hz timing record

v0.1.0 uses a fixed algorithmic step of 0.002 s and targets 500 Hz. The
measurements below were collected on 2026-08-03 using Python 3.10 and ROS 2
Humble on a general-purpose Linux host.

## Pure controller core

Command:

```bash
python3 -m pytest -q -s test/test_performance.py
```

The test warms up 250 adaptive controller steps, then times 5,000 calls with
finite nonzero reference, error, external wrench, RBF inference, online weight
update, projection, and safety filtering:

| Statistic | Time | Equivalent serial rate |
|---|---:|---:|
| median | 0.106613 ms | 9379.7 Hz |
| p95 | 0.113096 ms | 8842.0 Hz |
| p99 | 0.126680 ms | 7893.9 Hz |

This measures controller computation only. It excludes ROS serialization,
DDS, callbacks, the plant, message synchronization, plotting, and OS
scheduling.

## End-to-end ROS demo

Command:

```bash
ros2 launch neuro_adaptive_control demo.launch.py
```

| Field | Observed value |
|---|---:|
| simulated steps | 6000 |
| fixed simulated duration | 12.0 s |
| wall duration | 12.022547 s |
| observed step rate | 499.062 Hz |
| missed wall deadlines | 4 |
| stamp mismatches | 0 |
| final state | `stopped` |
| process exit | clean |

The deterministic stamp handshake preserves the 0.002 s algorithmic step even
when a wall deadline is late. These figures are one release-host observation,
not a throughput guarantee. Python, `rclpy`, DDS, and the host operating
system are not configured or certified for hard real-time execution.

The exact emitted record is committed as
[`metrics/v0.1.0_ros_metrics.json`](metrics/v0.1.0_ros_metrics.json).

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

## v0.3 six-DoF candidate measurement

Measured on 2026-08-04 with Python 3.10.12, NumPy 1.24.4, MuJoCo 3.9.0,
and ROS 2 Humble. The pure 42D two-layer V/W controller test warms up 250
steps and times 5,000 complete core calls:

```bash
python3 -m pytest -q -s test/test_pose_performance.py
```

| Statistic | Time | Equivalent serial rate |
|---|---:|---:|
| median | 0.159540 ms | 6268.0 Hz |
| p95 | 0.168337 ms | 5940.5 Hz |
| p99 | 0.186302 ms | 5367.6 Hz |

The installed headless ROS launch completed 7,500 controller samples and
30,000 MuJoCo substeps for 15.0 s simulated time in 10.725 s wall time, an
observed 699.319 steps/s on this run:

```bash
ros2 launch neuro_adaptive_control payload_benchmark.launch.py \
  viewer:=false realtime:=false
```

Across the eight regenerated held-out/comparison artifact trials, every
controller completed; observed headless rates ranged from 710.103 to
914.004 steps/s. Exact per-trial rates remain in the machine-readable
artifact. They are throughput observations, not deadline guarantees. The
default viewer launch deliberately wall-paces the simulation, and Python,
ROS 2, DDS, MuJoCo rendering, and the general-purpose kernel are not
hard-real-time systems.

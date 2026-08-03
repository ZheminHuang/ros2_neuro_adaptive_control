# Deterministic demo results

The v0.1.0 comparison uses the same unknown coupled plant, zero initial state,
circle reference, 12 s duration, 0.002 s controller step, four plant substeps,
and seed 7. The only changed setting is RBF adaptation (`false` for baseline,
`true` for NAC).

| Metric | Frozen-weight baseline | NAC | Change |
|---|---:|---:|---:|
| Impedance tracking RMSE | 0.0102201 m | 0.00172787 m | 83.09% lower |
| Desired tracking RMSE | 0.0158431 m | 0.0114870 m | 27.50% lower |
| Impedance maximum error | 0.0140923 m | 0.0108417 m | 23.07% lower |
| Command RMS norm | 1.36121 N | 1.39635 N | 2.58% higher |
| Command maximum norm | 2.79301 N | 3.41573 N | 22.30% higher |
| Saturated samples | 0 / 6000 | 0 / 6000 | unchanged |

The machine-readable record is
[`metrics/v0.1.0_demo_metrics.json`](metrics/v0.1.0_demo_metrics.json), and the
plot is [`images/demo_results.png`](images/demo_results.png). From the
repository root, reproduce both:

```bash
python3 examples/run_deterministic_demo.py \
  --duration 12.0 \
  --output-directory results/v0.1.0
```

These are deterministic simulation results for the bundled scenario, not a
general performance claim, formal stability proof, hardware result, or hard
real-time measurement.

The exact default ROS launch was also run for its full 12 simulated seconds.
It processed 6,000 fixed steps, reported 499.06 Hz observed wall-clock
throughput, 4 missed wall deadlines, and 0 stamp mismatches, then stopped and
exited cleanly. That ROS measurement uses online telemetry decimation and is
separate from the full-history NumPy comparison above. Timing scope and the
reproduction commands are recorded in [`performance.md`](performance.md).

# Mathematical contract

This document fixes the signs, dimensions, approximation model, and discrete
implementation shared by the v0.1 Cartesian demo and the v0.2 MuJoCo feature
branch. Robot-specific dynamics and frame details are defined in
[mujoco_dynamics_contract.md](mujoco_dynamics_contract.md).

## Scope and meaning of model-free

The controller is designed around an unknown translational Cartesian plant

\[
M_C(q)\ddot x+C_C(q,\dot q)\dot x+F_C(q,\dot q)+G_C(q)
=f_c+K_h f_{ext},\qquad x,f_c,f_{ext}\in\mathbb R^3.
\]

*Model-free* means that the NAC and its force-to-torque mapper do not read or
evaluate the plant mass matrix, Coriolis term, friction, gravity term, MuJoCo
`qM`, `qfrc_bias`, contact parameters, or a ground-truth disturbance. MuJoCo
necessarily owns those quantities to simulate the plant, but they are not NAC
inputs.

The controller still requires measured or estimated joint state, forward
kinematics, a geometric Jacobian, Cartesian pose and twist, frame transforms,
and a robot-specific command mapping. These are kinematics and measurements,
not known rigid-body dynamics.

The adaptive controller produces a **three-dimensional translational force**.
The MuJoCo adapter adds an independent, non-adaptive orientation-hold moment
and joint damping. That auxiliary control does not turn the NAC into a
validated 6D controller.

## Impedance reference model

The prescribed translational response is

\[
M_m\ddot x_m+D_m\dot x_m+K_mx_m=K_hf_{ext}+f_a,
\]

with the full moving-reference auxiliary input

\[
f_a=M_m\ddot x_d+D_m\dot x_d+K_mx_d.
\]

For a fixed point, \(\dot x_d=\ddot x_d=0\), so the auxiliary input reduces to
\(K_mx_d\). The signs of the model-following variables are fixed as

\[
e_m=x_m-x,\qquad
\dot e_m=\dot x_m-\dot x,\qquad
r=\dot e_m+\Lambda e_m.
\]

Changing the error to \(x-x_m\) would require changing every dependent command
and adaptation sign.

## Lumped unknown dynamics

Under the ideal Cartesian representation, substitution gives

\[
M_C\dot r=-C_Cr+G(z)-f_c-K_hf_{ext},
\]

where the source derivation defines

\[
G(z)=M_C(\ddot x_m+\Lambda\dot e_m)
 +C_C(\dot x_m+\Lambda e_m)+F_C+G_C.
\]

The RBF output \(\hat G\in\mathbb R^3\) is therefore interpreted only as a
lumped Cartesian force estimate. It is not a joint-space gravity vector, a
mass-matrix estimate, or an orientation model.

## 21D and 27D regressor contracts

The fixed Gaussian RBF network accepts an adapter-selected dynamics context
followed by the same fifteen model/error features:

\[
z=[z_{dyn},x_m,\dot x_m,\ddot x_m,e_m,\dot e_m].
\]

The two supported contexts are deliberately separate:

| Mode | Dynamics context | Complete input | Dimension |
|---|---|---|---:|
| Cartesian deterministic demo | \([x,\dot x]\) | \([x,\dot x,x_m,\dot x_m,\ddot x_m,e_m,\dot e_m]\) | 21 |
| UR5e MuJoCo | \([q,\dot q]\), using the six arm joints only | \([q,\dot q,x_m,\dot x_m,\ddot x_m,e_m,\dot e_m]\) | 27 |

The gripper joints are not appended to the 27D vector. Gripper mass, motion,
payload, and contacts remain part of the unknown plant seen through arm and
Cartesian measurements. A 21D center matrix must never be silently reused for
the 27D controller.

For one MuJoCo control stamp \(k\), the runner supplies \(q_k,\dot q_k\).
The core first advances the impedance model once, then constructs the RBF
input from \(q_k,\dot q_k\), the returned model state, and errors relative to
the measured Cartesian state \(x_k,\dot x_k\). No wrapper may advance the
impedance model a second time.

## RBF approximation contract

The manuscript and historical implementations use adaptive sigmoid/tanh
networks, **not** an RBF network. This project deliberately implements a
fixed-basis Gaussian RBF specialization. For input dimension \(d\),

\[
\bar z_j=\operatorname{clip}(z_j/s_j,-c,c),\qquad c=3,
\]

\[
\phi_i(z)=\exp\!\left(-\frac{\lVert\bar z-c_i\rVert^2}{2b_i^2}\right),
\qquad
\hat G(z)=\hat W^T\phi(z)\in\mathbb R^3.
\]

Centers and widths are fixed; only \(\hat W\in\mathbb R^{N\times3}\) adapts.
Weights start at exactly zero. The reproducible defaults are:

| Mode | \(d\) | Bases \(N\) | Width | Seed | \(\gamma\) | \(\kappa\) | Weight limit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cartesian demo | 21 | 31 | 2.5 | 7 | 5.0 | 0.01 | 80 |
| UR5e MuJoCo | 27 | 45 | 3.5 | scenario seed, default 23 | 18.0 | 0.01 | 120 |

Generated centers are uniform in \([-0.75,0.75]^d\), with the first center
forced to zero. The MuJoCo feature scales, in regressor order, are six joint
position values at 2.0, six joint velocity values at 2.0, and three values per
Cartesian block at 0.60, 0.50, 3.0, 0.08, and 0.50 for
\(x_m,\dot x_m,\ddot x_m,e_m,\dot e_m\), respectively.

## NAC command, robust term, and adaptation

The common translational command is

\[
u_{rob}=(\lVert\hat W\rVert_F+b_r)K_rr,
\]

\[
\boxed{f_c^{raw}=\hat G+K_vr+u_{rob}-K_hf_{ext}},
\qquad
f_c=\operatorname{saturate}(f_c^{raw}).
\]

The effective robust contribution has a positive sign because the source form
writes \(-\nu\) with
\(\nu=-K_Z(\lVert\hat Z\rVert_F+Z_B)r\). The configurable robust bias is a
controller gain, not a certified bound on ideal neural weights.

Only the RBF output weights adapt:

\[
\dot{\hat W}=\gamma\left(\phi r^T
-\kappa\lVert r\rVert\hat W\right).
\]

The positive \(\phi r^T\) sign is paired with \(e_m=x_m-x\). With adaptation
disabled, the update is a no-op. The core applies per-axis Cartesian force
limits followed by a Euclidean-norm limit. That Cartesian saturation does not
currently undo the RBF update.

The MuJoCo runner additionally snapshots \(\hat W_k\) before the controller
step. If the later joint-torque mapper reports either torque-rate saturation
or absolute torque saturation, the runner restores that snapshot. Thus the
entire output-weight update for that sample is frozen; the already-computed
impedance step and bounded actuator command are not undone.

## External-force placement and sign

Here \(f_{ext}\) always means a three-dimensional force expressed in the
documented control frame and applied at the documented TCP point. A published
six-dimensional wrist wrench may also contain a moment, but that moment is not
an input to the translational NAC.

For the v0.1 synthetic plant, one coherent sample has the ideal `+ / + / -`
placement:

1. plant: \(+K_hf_{ext}\);
2. impedance model: \(+K_hf_{ext}\);
3. NAC command: \(-K_hf_{ext}\).

The MuJoCo controller fixes \(K_h=I_3\) and supports three modes:

- `none`: model, controller, and plant use zero external force;
- `injected`: the same sampled force enters the impedance model with `+`, the
  command with `-`, and is physically applied exactly once at the TCP with
  `+`; it is held across the four MuJoCo substeps;
- `virtual_ft`: the sampled environment-on-robot contact force enters the
  model with `+` and command with `-`, while the injected-force argument stays
  zero because MuJoCo contact dynamics already applied the physical force.

`virtual_ft` must never reapply the measured contact through `mj_applyFT`.
Contact forces can evolve during the following physics substeps, so the ideal
continuous cancellation is not a strict per-substep identity in this sampled
mode.

The raw wrist cut-wrench and contact-only wrench are different measurements.
The former includes load transmitted across the wrist sensor site; the latter
sums only external environment/robot contacts and shifts their moments to the
TCP. The current NAC runner uses contact-only force for `virtual_ft`. See the
robot-specific contract for the exact frame and sign.

## Discrete implementation

The impedance acceleration is evaluated from the old model state and advanced
with semi-implicit Euler:

\[
\begin{aligned}
\ddot x_{m,k}&=M_m^{-1}(K_hf_{ext,k}+f_{a,k}
-D_m\dot x_{m,k}-K_mx_{m,k}),\\
\dot x_{m,k+1}&=\dot x_{m,k}+\Delta t\,\ddot x_{m,k},\\
x_{m,k+1}&=x_{m,k}+\Delta t\,\dot x_{m,k+1}.
\end{aligned}
\]

Weights use explicit Euler followed by Frobenius-norm projection:

\[
\hat W^*=\hat W_k+\Delta t\,\dot{\hat W}_k,
\qquad
\hat W_{k+1}=\Pi_{\lVert W\rVert_F\le W_{max}}(\hat W^*).
\]

Both demos use a fixed controller period of 0.002 s. The v0.1 synthetic plant
uses four 0.0005 s semi-implicit substeps. The robot plant uses MuJoCo's
`implicitfast` integrator with the same 0.0005 s timestep and exactly four
zero-order-hold substeps per accepted command. Algorithmic stamps advance by
0.002 s; measured wall time is used only for performance reporting and ROS
watchdogs. This is a 500 Hz simulation/control target, not a hard real-time
guarantee.

## Safety-state contract

The pure controller retains its five states:

```text
start -> running -> stopping -> stopped
             \-> fault
```

NaN/Inf, invalid or reversed time, invalid \(\Delta t\), and watchdog expiry
latch the core fault and produce zero Cartesian force. Core reset restores the
caller-supplied impedance state, zero RBF weights, timing/watchdog history,
command/saturation history, and `start`.

The MuJoCo owner evaluates its state/contact guards before and after each
four-substep integration interval. `fault` is latched in both owner and core;
a stop request cannot transition either state machine out of fault. Only reset
clears the latch.

The MuJoCo experiment owner adds a transient `resetting` state:

```text
resetting -> start -> running -> stopping -> stopped
                         \-> fault
```

On simulation fault or normal stop, the owner resets torque-rate history,
replaces the last NAC output by the finite bounded command
\(\tau_{safe}=-D_q\dot q\), commands the gripper to hold its measured opening
with its finite effort limit, and clears all externally applied generalized
forces. No further physics step occurs, so the scene is then paused in that
safe-hold state rather than retaining the last NAC command or falling under
gravity. If measured velocity or gripper state is non-finite, its fallback is
zero actuator control. A deterministic reset restores MuJoCo
dynamic/actuator/applied-force state, arm/gripper/object initial state,
controller and mapper state, RBF weights, impedance state, command-rate
history, sequence counters, and the seeded generator before returning to
`start`.

## Source comparison and chosen resolution

| Item | Source material | v0.1 Cartesian demo | MuJoCo feature branch |
|---|---|---|---|
| Controlled dimension | n-D position examples | 3D translation | 3D translation; separate orientation PD |
| Error sign | \(x_m-x\) | same | same |
| Moving feedforward | full M/D/K terms | full terms | full terms |
| Neural model | two-layer sigmoid/tanh | fixed Gaussian RBF, 21D | fixed Gaussian RBF, 27D |
| Adaptation | adaptive hidden/output layers | output weights, projection | same; restore weights on downstream torque limiting |
| Command mapping | Cartesian force/wrench | synthetic Cartesian plant | \(J_v^Tf_c+J_\omega^Tm_R-D_q\dot q\) |
| External input | ideal `+ / + / -` | coherent synthetic sample | injected once, or measured contact without reinjection |
| Integration | continuous proof; integrator unspecified | fixed semi-implicit plant | MuJoCo `implicitfast`, four fixed substeps |
| Reset | not a software contract | deterministic core reset | deterministic plant/controller/mapper reset |

## Validity boundary

Tests verify dimensions, signs, deterministic reset, finite-value guards,
virtual work before nonlinear limits, numerical behavior, and repeatable
tracking/grasp metrics. They do **not** extend the manuscript's continuous UUB
result to this fixed-RBF, sampled, leaked, projected, force-saturated,
torque-rate-limited, torque-saturated, intermittently adaptation-frozen,
contacting rigid-body system. They also do not prove hard real-time behavior,
formal discrete-time stability, hardware safety, or validated 6D control.

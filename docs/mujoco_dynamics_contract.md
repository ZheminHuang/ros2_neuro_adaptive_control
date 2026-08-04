# MuJoCo dynamics contract

This document defines the implemented boundary between the v0.3 six-DoF NAC,
the retained legacy 3D NAC, the UR5e + Robotiq MuJoCo plant, and visualization.
MuJoCo computes dynamics, constraints, collision, and contact. RViz or the
native viewer only consumes state and never computes or feeds back dynamics.

The common NAC equations and error signs remain authoritative in
[math_contract.md](math_contract.md).

## Coordinates, frames, and state

The simulated UR5e base is fixed in the MuJoCo world. For this demo, the
MuJoCo world frame and controller base frame coincide. The arm state is

\[
q,\dot q\in\mathbb R^6.
\]

The Robotiq mechanism has eight additional articulated joints, but those
joints are not part of the arm NAC regressor. The TCP is the `gripper_pinch`
site. At every accepted control stamp, the plant returns

\[
x^B=\operatorname{FK}(q),\qquad
\dot x^B=J_v^B(q)\dot q,
\]

together with the TCP rotation \(R^B_T\), spatial translational Jacobian
\(J_v^B\in\mathbb R^{3\times6}\), and spatial angular Jacobian
\(J_\omega^B\in\mathbb R^{3\times6}\). Both Jacobians refer to the same TCP
point and are expressed in the base/world frame.

## Ownership of dynamics

One `MujocoUR5ePlant` instance owns one MuJoCo model/data pair. It contains the
UR5e arm, the dynamic Robotiq linkage and tendon actuator, collision geometry,
contact constraints, table, and free grasp object. MuJoCo alone advances the
joint, gripper, object, and contact dynamics.

The NAC and force-to-torque mapper do not read MuJoCo `qM`, `qfrc_bias`,
inverse dynamics, gravity compensation, contact model parameters, or the
ground-truth disturbance. They receive only state, FK/Jacobian kinematics, a
selected measured external force, and configured feedback/limit values.

The v0.3 payload benchmark fixes the NAC external wrench to zero. Its
controller observation is limited to the six arm positions and velocities,
TCP pose and twist, desired/impedance state, and geometric Jacobian. The
physical payload case, contact state, acquisition schedule, object identifier,
body mass, COM, inertia, `qM`, and `qfrc_bias` are not passed to the adaptive
controller. Contact is used only for safety/metrics and to freeze the explicit
comparison controller at the observed pickup event.

## v0.3 analytical state and torque path

Reset captures (R_0). Each coherent MuJoCo sample is converted to

\[
\rho=\operatorname{Log}(RR_0^T)^\vee,\qquad
\dot\rho=J_l^{-1}(\rho)\omega.
\]

The six-dimensional geometric Jacobian is

\[
J_g=\begin{bmatrix}J_v\\J_\omega\end{bmatrix},
\]

and the adapter forms

\[
\mathcal E=\operatorname{blkdiag}(I_3,J_l(\rho)),\qquad
J_a=\mathcal E^{-1}J_g.
\]

For the analytical NAC output (u_c\in\mathbb R^6), the adapter computes

\[
w_c=\mathcal E^{-T}u_c,
\]

\[
\boxed{\tau_{raw}=J_g^Tw_c=J_a^Tu_c}.
\]

The two forms are tested numerically for equality and virtual-work
consistency. There is no extra orientation PD or running joint damping in this
path. Torque-rate and absolute-torque limits are applied after the mapping.
Only stopping/fault handling may replace the command with bounded
(-D_{q,safe}\dot q).

## Physical payload acquisition

The free object is present from the first physics step and initially rests on
the table. A payload case changes the MuJoCo body's mass, inertial-frame COM,
and diagonal inertia before reset; `mj_setConst` recomputes model constants.
No mass is hot-switched at pickup. The dynamics change seen by the arm occurs
naturally when bilateral finger contact lifts the object and support transfers
from the table to the articulated gripper.

The canonical event requires both left and right finger contact plus at least
12 mm object lift. This event is logged for every controller. It disables both
V and W updates only for `frozen_at_payload`; adaptive NAC does not consume the
event. The nominal model-based baseline knows the bundled robot/gripper model
but not the payload. The oracle baseline adds known payload gravity/COM
compensation after acquisition and is reported as an upper reference.

The common 15 s scenario contains unloaded 6D motion, approach, close, an
80 mm lift, one smooth 40 mm-radius loaded XY circle, lower, release, and
retreat. Bounded orientation excitation runs during the circle and returns to
the pickup orientation. All variants share reference, payload, controller
period, four physics substeps, actuator limits, camera, and safety checks.

## Legacy v0.2 robot RBF input and 3D output

The robot controller is instantiated with a twelve-element dynamics context
and a 27D RBF input:

\[
z=[q_6,\dot q_6,x_m,\dot x_m,\ddot x_m,e_m,\dot e_m]
\in\mathbb R^{27}.
\]

It uses 45 fixed Gaussian bases, width 3.5, the scenario seed (default 23),
learning rate 18.0, leakage 0.01, weight limit 120, and adaptive output weights
\(\hat W\in\mathbb R^{45\times3}\). The controller output is only

\[
f_c^B\in\mathbb R^3,
\]

a translational force at the TCP. Neither gripper coordinates nor orientation
errors are RBF inputs in the current implementation. The network does not
estimate a six-dimensional wrench or joint torque.

The default Cartesian command limiter applies per-axis limits of 120, 120,
and 140 N, then a 180 N Euclidean-norm limit. Cartesian limiting does not by
itself restore the pre-update weights; the downstream joint-limit rule is
specified below.

## Legacy v0.2 orientation hold and joint damping

Reset captures a fixed desired TCP orientation \(R_d\). Both \(R\) and \(R_d\)
map TCP-frame vectors into the base frame. The local desired-minus-actual
orientation error is

\[
e_R^B=\frac{1}{2}\operatorname{vee}
\left(R_dR^T-RR_d^T\right).
\]

For the fixed desired orientation, the non-adaptive Cartesian moment is

\[
m_R^B=K_Re_R^B-D_R\omega^B.
\]

The geodesic distance

\[
\theta_R=\cos^{-1}\!\left(\frac{\operatorname{tr}(R_d^TR)-1}{2}\right)
\]

is checked separately because the local `vee` error degenerates near 180
degrees. The configured orientation guard is 35 degrees.

The legacy unbounded arm command is

\[
\boxed{\tau_{raw}=J_v^{B\,T}f_c^B
 +J_\omega^{B\,T}m_R^B-D_q\dot q}.
\]

The first term maps the 3D NAC force. The second is a separate orientation PD
task. The third is dissipative joint damping. The orientation task stabilizes
degrees of freedom that the translation-only NAC does not control; it is not
a learned or validated 6D NAC and can couple into translation through the
actual robot dynamics.

These legacy defaults are:

| Quantity | Values |
|---|---|
| \(K_R\) | 45, 45, 35 |
| \(D_R\) | 6, 6, 5 |
| \(D_q\) | 0.8, 0.8, 0.7, 0.20, 0.20, 0.15 |
| joint torque limits | 140, 140, 140, 27, 27, 27 N m |
| joint torque-rate limits | 8000, 8000, 8000, 3000, 3000, 3000 N m/s |

## Virtual work and nonlinear limits

Before damping and nonlinear limits, the task mapping satisfies

\[
\dot q^T\tau_{task}
=f_c^{B\,T}(J_v^B\dot q)+m_R^{B\,T}(J_\omega^B\dot q),
\]

where
\(\tau_{task}=J_v^{B\,T}f_c^B+J_\omega^{B\,T}m_R^B\). Joint damping obeys

\[
\dot q^T(-D_q\dot q)\le0.
\]

The virtual-work equality is not expected after torque-rate or absolute torque
limiting. The mapper first applies the rate limit

\[
\begin{aligned}
\Delta\tau_k&=\operatorname{clip}
(\tau_{raw,k}-\tau_{k-1},-\dot\tau_{max}\Delta t,
\dot\tau_{max}\Delta t),\\
\tau_k^{rate}&=\tau_{k-1}+\Delta\tau_k,
\end{aligned}
\]

then applies the per-joint absolute limit

\[
\tau_k=\operatorname{clip}
(\tau_k^{rate},-\tau_{max},\tau_{max}).
\]

MuJoCo arm actuators have unit gear, so the six accepted commands are written
directly to the six arm actuator controls. Model actuator force ranges provide
an additional plant-side bound.

## Adaptation and actuator acceptance

The core evaluates \(\hat G\) and the force command with \(\hat W_k\), then
performs its explicit output-weight update. The experiment runner saves
\(\hat W_k\) before that step. If the joint mapper reports either rate
saturation or absolute torque saturation, the saved matrix is restored before
the next sample. The limited torque is still applied.

This is an actuator-acceptance guard, not a proof-preserving projection. It
does not roll back the impedance integration, Cartesian safety history, or
other controller state. Cartesian force saturation inside the core does not
currently trigger this rollback.

## External-force modes

The MuJoCo controller uses \(K_h=I_3\). Its external input is always the force
component expressed in base/world coordinates at the TCP; contact torque is
published and diagnosed but is not passed to the 3D NAC.

### `none`

\[
f_{ext}=0,
\]

and no external force is injected.

### `injected`

At stamp \(k\), one deterministic force sample \(f_{ext,k}^B\) is used in all
three intended places:

```text
impedance model:  +f_ext,k
NAC command:      -f_ext,k
MuJoCo plant:     +f_ext,k
```

For each of the four physics substeps, the plant clears `qfrc_applied`, calls
`mj_applyFT` once with that force, zero moment, the TCP world point, and the
gripper mount body, then advances MuJoCo. Clearing before every call prevents
force accumulation; the sample is simply held for the controller interval.

### `virtual_ft`

The current contact-only force sample is used as \(f_{ext,k}^B\) in the
impedance model and NAC command. The `injected_force_world` argument is exactly
zero. MuJoCo has already generated contact through its constraint solver, so
reapplying the measured force would double count the interaction.

Natural contact can change during the following four substeps. Consequently,
the ideal continuous `+ / + / -` cancellation motivates the signs but is not
an exact sample-by-sample cancellation proof for this delayed, sampled contact
loop.

## Contact-only wrench

`contact_summary()` excludes robot/robot and environment/environment pairs
from the reported wrench. For every remaining contact, MuJoCo returns a local
contact-frame wrench. Let

\[
R^W_C=(\texttt{contact.frame})^T.
\]

The implementation first computes the world-frame wrench on geom 2:

\[
f_2^W=R^W_C f_2^C,\qquad n_2^W=R^W_C n_2^C.
\]

If the robot is geom 2, these values are retained. If the robot is geom 1,
both are negated. The resulting sign is always **environment on robot**. The
resultant contact-only wrench is

\[
f_{contact}^W=\sum_i f_{robot,i}^W,
\]

\[
n_{contact,TCP}^W=\sum_i\left[
n_{robot,i}^W+(p_i^W-p_{TCP}^W)\times f_{robot,i}^W\right].
\]

`contact_force_norm_n` is the sum of individual contact-force norms, not the
norm of the resultant. Object contacts on left/right finger subtrees are
expected grasp contacts. Object contact elsewhere on the robot, or any robot
contact with other environment bodies, is flagged as unexpected. The
penetration diagnostic is the maximum penetration over all current MuJoCo
contacts, while the wrench itself uses only the filtered robot/environment
pairs.

## Raw wrist wrench is a different signal

The MJCF force and torque sensors at `wrist_ft_site` form the raw wrist
cut-wrench. `wrist_wrench_raw()` rotates each local sensor vector into the
world frame. Its moment remains referenced to the sensor site; it is not the
contact-only moment shifted to `gripper_pinch`.

The raw signal can include force transmitted by distal gripper/object inertia,
gravity, and contact. It is published for diagnostics and is not the default
NAC external input. Treating it as contact-only force would require a separate,
documented tare/payload/inertial compensation contract.

## Gripper actuation boundary

The metric adapter maps 0.085 m opening to actuator control 0 and zero opening
to control 255. The MuJoCo tendon actuator couples the two driver joints; joint
equalities and linkage constraints drive all eight articulated gripper joints.
Each accepted plant step also writes the current per-goal maximum effort into
the actuator force range. Thus a request below the global 5 N ceiling is
enforced by MuJoCo rather than used only as action feedback metadata.

## Fixed-step order

The synchronous tracking loop uses \(\Delta t_c=0.002\) s and the MuJoCo model
uses \(\Delta t_p=0.0005\) s. One cycle is:

1. read coherent \(q_k,\dot q_k\), FK, Jacobians, TCP twist, and contact at
   sequence/stamp \(k\);
2. select `none`, `injected`, or `virtual_ft` external force;
3. advance the impedance model once, compute the 27D RBF/NAC force, and update
   weights;
4. map force to bounded arm torque using the state and Jacobians from stamp
   \(k\), restoring the old weights if torque limiting occurred;
5. hold arm torque, gripper target/effort, and any injected force for exactly
   four MuJoCo substeps;
6. evaluate the display/metric reference at \(k+1\), then publish/store the
   reference, state, and contact with that common stamp.

The plant rejects a command unless `sequence_id == step_count`, and it rejects
any substep count other than four. Algorithmic time advances from the fixed
stamp. `missed_wall_deadlines`/`callback_overrun_count` count callbacks whose
own execution exceeded 2 ms; timer inter-arrival median/p95/p99/max are
reported separately so scheduling jitter is not hidden. These wall-time
measurements never change \(\Delta t_c\) and are not a hard-real-time claim.

## Fault, stop, and deterministic reset

The experiment owner has

```text
resetting -> start -> running -> stopping -> stopped
                         \-> fault
```

The pure NAC retains its own five-state safety supervisor without
`resetting`. The robot loop faults on non-finite state, FK, Jacobian, or
torque; control stamp or sequence mismatch; MuJoCo warnings; excessive joint
speed; arm or gripper joint-limit violation beyond the 0.005 rad soft-limit
tolerance; gripper driver-coupling inconsistency; hard raw-torque limits;
workspace or orientation violation; excessive contact force; or unexpected
tracking-scene collision. The tracking defaults use a 3.5 rad/s joint-speed
guard, hard raw-torque limits of 280, 280, 280, 54, 54, 54 N m, a 35-degree
orientation guard, and a 250 N contact-force guard.

State/contact guards are evaluated on both the pre-command sample and the
post-integration sample before metrics, publication, or clean completion. A
fault is latched in both the experiment owner and the pure safety supervisor;
the stop service preserves that fault and reports failure until reset.

Fault and normal stop both reset torque-rate history and replace the last NAC
command by the bounded damping-only torque
\(\tau_{safe}=-D_q\dot q\). The gripper target is replaced by a finite hold at
the measured opening, and externally applied generalized forces are cleared.
If velocity or gripper feedback is non-finite, the affected actuator fallback
is zero. No further `mj_step` occurs, so the dynamic scene is paused in this
safe-hold state and cannot continue on a stale command.

Reset calls `mj_resetData`, restores deterministic arm, gripper, object,
control, applied-force, sequence, step-count, and RNG state, restores the
global gripper effort range, captures the reset TCP orientation, constructs a
new zero-weight controller and mapper, and returns the owner to `start`.

## Validity boundary

The conceptual Cartesian equation is a lumped task-space description. The
implemented plant is joint-space rigid-body/contact dynamics with orientation
feedback, gripper constraints, actuator limits, and changing contacts. The
source continuous-time result does not directly establish UUB or stability
for this implementation because it changes the neural architecture and adds
sampling, semi-implicit impedance integration, explicit projected adaptation,
force and torque saturation, torque-rate limits, conditional adaptation
rollback, MuJoCo integration, orientation coupling, and nonsmooth contact.

Tests can establish signs, dimensions, virtual work before limits,
deterministic replay, finite/bounded numerical behavior, and scenario metrics.
They do not establish formal sampled-data stability, hard real-time behavior,
safe real-robot operation, or validated 6D control.

## Why a larger command range can improve low-speed tracking (joystick “deadzone”)

It’s tempting to think: “If I only train on low speeds, the policy should be best at low speeds.”
In practice for joystick locomotion RL, a **narrow / low command distribution** often creates a
**stand-still local optimum** (“deadzone”) that is *harder* to escape than training with a wider
range of commands.

This note explains why you can observe:

- Training with a *larger* command range (e.g. \(v_x \in [-1, 1]\)) produces a policy that also
  **starts walking and tracks low speeds** (e.g. 0.25 m/s),
- while training with a *smaller* range (e.g. \(v_x \in [-0.5, 0.5]\)) can produce a policy that
  **tilts / shifts COM but won’t initiate stepping** at low speeds.

---

### 1) Narrow/low command ranges create a “do nothing” local optimum

With low speeds, the return can be dominated by:

- **alive / stability** rewards (stay upright),
- **regularization penalties** (action_rate, torque/energy penalties, pose penalties),
- weak pressure from tracking because the commanded velocity is small.

So a policy that **stands still** (or leans slightly) can get decent total reward, and PPO can
converge there.

When you widen the command range, standing still produces large tracking error for many samples,
so the policy must discover a stepping gait to get good return.

---

### 2) Wider ranges improve the learning signal (gradients) early

At low commands:
- “almost moving” vs “actually moving” can be a small reward difference compared to noise and
  penalties.

At higher commands:
- tracking error while stationary is large, so the advantage signal is larger and learning more
  decisively pushes toward motion.

---

### 3) Learning locomotion is learning a gait primitive, not memorizing speeds

Once a stable gait emerges (swing timing, foot clearance, COM shift), producing different speeds
often becomes a “gain scheduling” problem:

- smaller step length / cadence → low speed
- larger step length / cadence → high speed

So training that forces discovery of a gait at higher speeds can indirectly improve low-speed
performance because the primitive is already learned.

---

### 4) Symmetry breaking is easier at larger commands

From symmetric initial conditions, low commands can lead to symmetric actions and “stuck” contact
states where neither foot commits to swing.
Larger commands force asymmetry sooner (unload one foot), which helps the policy discover the
first step. After that, it can apply the same mechanism at low commands.

---

### 5) Practical checks for “deadzone” vs “true low-speed inability”

If a policy won’t start at 0.25 m/s, check:

- **Tracking vs penalties**
  - If it stands still and still gets ok reward, penalties may dominate and encourage “don’t move”.
- **Torques saturating**
  - If the torque envelope clips heavily at startup, the controller may be torque-limited and need
    a different initial pose or reduced penalties.
- **Push disturbances**
  - If walking starts only after a push, the policy may be stuck in a symmetric basin.

---

### 6) Practical training strategies

- **Command range widening** (what you did): often the simplest fix.
- **Command curriculum**: start wide (force gait discovery), then narrow later if you care about
  precision at low speeds.
- **Oversample low speeds** after gait exists: bias command sampling toward low speeds in a later
  fine-tune run.
- **Reduce penalties** that can create “don’t move” optima at low commands (torque/energy/action-rate),
  then reintroduce them gradually once locomotion is stable.


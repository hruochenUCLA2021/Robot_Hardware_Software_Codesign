## NOTE: Chopstickbot “morphology-uniform” policy (MJX/Brax)

Goal: train **one** policy that works across a range of Chopstickbot hardware morphologies (e.g., leg/bar length), by conditioning the policy on a small “morphology vector”, rather than training a separate policy per length.

This note explains the practical options in MJX/Brax and recommends **Option A** as the default approach.

---

## Key constraint (MJX reality check)

In MJX, the compiled program is effectively tied to the MuJoCo model that comes from the XML:

- Changing **link lengths / geom sizes / joint locations / meshes** changes the underlying model structure.
- That generally means a **different XML → different MJX model → separate JIT/compile** (at least once per variant).

So continuous “random length every episode from an infinite set” by generating new XMLs on the fly is not viable: it would trigger compilation repeatedly and kill training throughput.

---

## What we actually want: conditioning + morphology distribution

To get a “uniform policy”, do both:

1. **Vary morphology in the simulator** (across a finite pool of XML variants).
2. **Expose morphology parameters to the policy** (add them to the observation).

This makes the control mapping well-posed:
- without conditioning, the same observation could correspond to different dynamics (different lengths), which makes learning harder and increases instability/forgetting.

---

## Options overview

### Option A (recommended): frequent switching across a finite pool of XMLs

**Idea:** pre-generate \(K\) morphology variants (XML files), and during training **switch which model is used frequently** (per update / per short block), while keeping a single set of policy parameters.

Why it’s best:
- Minimal changes to the standard PPO pipeline.
- No need to mix multiple models inside a single `vmap` batch.
- Compile cost is paid **once per model** (first time it’s used), then reuse is fast.
- Frequent alternation greatly reduces catastrophic forgetting.

### Option B (more complex): “true mixed batch” across multiple models per PPO update

Collect rollouts on multiple models in one outer loop, concatenate experience, then do one PPO update.

Pros:
- Closest to the literal “mixed morphology within one update” idea.

Cons:
- Typically requires refactoring around Brax PPO internals (rollout + update split).
- More engineering + more places to break.

### Option C (not recommended): “mega-XML” containing many robots of different sizes

Put 50 robots into one XML, disable collisions, treat “robot index” as batch dimension.

Cons:
- Hard to do correct per-robot termination/reset (“partial resets”).
- Not aligned with Brax PPO’s expected env API / vectorization assumptions.
- Usually not faster than proper batching, and significantly more complex to maintain.

---

## Option A in detail (how to do it)

### 1) Choose a finite pool of morphologies

Pick a length range and discretize it:
- Example: length \([0.10, 0.30]\) meters
- Choose \(K = 10\)–\(20\) values (uniform grid or stratified)

Generate one XML per variant, e.g.:
- `chopstick_len_0.10.xml`
- `chopstick_len_0.12.xml`
- …

Practical advice:
- Start with **K=10** (often enough to generalize/interpolate).
- Only go to 50–100 if you’ve proven you need it and can afford compilation/memory.

### 2) Add morphology to the observation (policy conditioning)

Add a small vector to obs such as:
- `leg_length` (normalized to \([-1, 1]\) or \([0, 1]\))
- optionally also: `mass_scale`, `com_shift`, etc. if you randomize them

You want the policy to receive a consistent “context” that disambiguates the dynamics.

Rule of thumb:
- Keep the morphology vector small (1–10 scalars).
- Normalize to stable numeric ranges.

### 3) Switch models frequently during training

The key to not forgetting is **frequency**. Avoid long uninterrupted blocks.

Common schedules:
- **Per PPO update**: pick one morphology at random (or round-robin) each update.
- **Small block**: switch every 5–20 updates.

Avoid:
- “train 1 million steps on model A, then 1 million on B” (high risk of forgetting).

### 4) Pay compilation cost once per morphology (warm-up)

The first time a new model is used, JAX will compile the jitted computations for that model.

To control this:
- Warm up all \(K\) models early (one reset/step/short rollout) so compilation happens up front.
- After that, switching among these \(K\) models reuses compiled executables.

### 5) Evaluation to verify no forgetting

Periodically evaluate the current policy on **all** morphologies in your pool.

If performance regresses on some morphologies:
- increase their sampling probability
- decrease the switch block size
- confirm the morphology conditioning input is present and correct

---

## Why Option A is “better” (for MJX/Brax)

- **Matches Brax PPO assumptions**: one model per vectorized rollout batch.
- **Performance friendly**: reuse compiled code; no per-episode recompiles.
- **Engineering effort is low**: no need for multi-model batching logic inside PPO.
- **Forgetting is manageable**: switching per update (or per small block) keeps the policy trained on the full distribution.

---

## Practical expectations (what “uniform” means)

With a finite pool, you’re really training a policy that is:
- correct on the sampled morphologies
- and tends to **interpolate** to in-between lengths (often well, but not guaranteed)

If you need better interpolation:
- increase \(K\) modestly
- sample morphologies more densely in difficult regions
- consider adding more informative morphology inputs (e.g., multiple link lengths, not just one scalar)


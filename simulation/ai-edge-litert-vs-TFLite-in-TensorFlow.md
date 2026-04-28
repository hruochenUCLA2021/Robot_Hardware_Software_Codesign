### ai-edge-litert vs TFLite-in-TensorFlow (why we switched)

This note records a real issue we hit when running **real-time MuJoCo CPU + viewer** inside a **ROS2 node** for Phonebot, while trying to run a policy exported to **`.tflite`**.

### The symptom

- Running the realtime simulator node with `policy_format:=tflite` could **freeze/hang** at:
  - `import tensorflow`
- The same `.tflite` model worked fine in a normal Python script (e.g. batch testing / conversion checks).

### Root cause (practical explanation)

In the realtime ROS2 node we have multiple native subsystems active in one process:

- **MuJoCo passive viewer** (`mujoco.viewer.launch_passive`)
  - spawns a GUI thread (GLFW/OpenGL) + uses native locks + event loop
- **ROS2 executor + timers**
  - its own threading model + signal handling (Ctrl+C)
- **TensorFlow import**
  - loads large native libraries and may initialize CPU thread pools and/or GPU/CUDA/XLA paths during import

On this machine, `import tensorflow` inside that combined environment could hang indefinitely.
This is best thought of as a **native initialization hang / deadlock risk** caused by **init-order** and
**native library interactions** (OpenGL/GLFW + drivers + TF native init), not a Python exception.

Why it looked “weird”:

- `test_policy_convert.py` can run `tf.lite.Interpreter` successfully because it is a **simple script**
  without ROS2 executor + MuJoCo viewer thread running at the same time.

### Why `tflite-runtime` didn’t solve it here

The classic lightweight option is `tflite-runtime`, but on **Python 3.12** it often fails because PyPI
does not provide matching wheels (so `pip install tflite-runtime` can return “No matching distribution”).

### What is `ai-edge-litert` (LiteRT) and why it worked

LiteRT is the newer standalone runtime for running `.tflite` models in Python **without importing the full TensorFlow package**.

- Same model file: **the `.tflite` flatbuffer is unchanged**
- Different runtime: LiteRT provides an `Interpreter` API compatible with the classic TFLite interpreter workflow,
  but avoids pulling in the entire TensorFlow stack (and its heavy native initialization).

In our ROS2 + MuJoCo viewer process, using LiteRT avoided the `import tensorflow` hang and allowed stable realtime inference.

### What we changed in this repo (summary)

- Realtime node (`simulation/ros2/.../realtime_keyboard_node.py`) supports:
  - **Brax checkpoints** (prefer `ppo_checkpoint.load_policy` if `ppo_network_config.json` exists; fallback to `ppo.train(...)`)
  - **TFLite `.tflite` policies** with selectable backend:
    - `tflite_backend:=litert` (recommended on Py3.12)
    - `tflite_backend:=tensorflow` (can hang in the realtime viewer process)

### Recommended usage

- Install LiteRT:

```bash
pip install ai-edge-litert
```

- Run realtime node with LiteRT backend:

```bash
ros2 run simulator_mujoco_phonebot phonebot_realtime_keyboard \
  --ros-args \
  -p checkpoint_dir:=/ABS/PATH/TO/model.tflite \
  -p policy_format:=tflite \
  -p tflite_backend:=litert
```

### Takeaway

If you need `.tflite` inference **inside a realtime ROS2 + MuJoCo viewer process**, prefer **LiteRT (`ai-edge-litert`)**
over `tf.lite.Interpreter` from full TensorFlow to avoid import-time hangs and reduce native initialization complexity.


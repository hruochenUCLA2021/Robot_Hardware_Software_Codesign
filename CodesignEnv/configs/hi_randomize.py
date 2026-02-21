# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Domain randomization for CodesignEnv.

For initial bring-up of new robot models, we keep this as a no-op to avoid
assuming specific geom/body index layouts. You can add task-specific
randomization later once the training pipeline is stable.
"""

# import jax
# from mujoco import mjx

# from mujoco_playground._src.locomotion.t1 import randomize as t1_randomize


# def domain_randomize(model: mjx.Model, rng: jax.Array):
#   """Delegate to the official T1 domain randomizer (JAX-based).

#   Args:
#     model: mjx.Model for the Hi robot.
#     rng: JAX PRNG key or batch of keys, supplied by Brax.

#   Returns:
#     A tuple (model_v, in_axes) matching the expected contract of
#     `wrapper.wrap_for_brax_training` and Brax's PPO.
#   """
#   return t1_randomize.domain_randomize(model, rng)

import jax
from mujoco import mjx


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """No-op randomizer (returns model unchanged)."""
  in_axes = jax.tree_util.tree_map(lambda _: None, model)
  return model, in_axes

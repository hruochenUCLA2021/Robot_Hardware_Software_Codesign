# MJCF `file="..."` asset path gotcha (MuJoCo): `meshdir` / `texturedir`

## What happened

Some of our scene XMLs reference terrain assets like:

- `file="../assets/hfield.png"`
- `file="../assets/rocky_texture.png"`

At first glance, it’s natural to think the paths are resolved “relative to the scene XML directory”. However, MuJoCo’s compiler has two directory prefixes that affect how `file="..."` is resolved:

- **`compiler meshdir`**: used for **meshes and heightfields** (hfield PNGs count here)
- **`compiler texturedir`**: used for **texture files**

In our phonebot FV2 model, the included robot XML sets:

- `meshdir="meshes/"`

So MuJoCo will resolve the heightfield file like:

\[
  \text{full\_path} = \text{scene\_dir} / \text{meshdir} / \text{file}
\]

That means if the scene says:

- `file="assets/hfield.png"`

MuJoCo tries:

- `.../model_phonebot_fred_v2/meshes/assets/hfield.png`

…which is usually *not* where you put terrain assets.

## The robust convention used in this repo

To avoid confusion and make scene loading deterministic (same pattern as the working HERMES rough-terrain scene):

- Keep an `assets/` folder **next to** the `meshes/` folder inside the model folder:
  - `models/model_phonebot/assets/`
  - `models/model_phonebot_fred_v2/assets/`
- In scene XMLs, **use `../assets/...`** so that the `..` cancels the `meshdir="meshes/"` prefix:
  - `file="../assets/hfield.png"`
  - `file="../assets/rocky_texture.png"`

Why this works with `meshdir="meshes/"`:

\[
  \text{scene\_dir}/\text{meshes}/\text{../assets}/x \;=\; \text{scene\_dir}/\text{assets}/x
\]

Additionally, we keep a shared copy under:

- `models/assets/`

and copy/sync into the per-model `assets/` folders when needed.

## Why `.gitignore` matters

`Robot_Hardware_Software_Codesign/.gitignore` ignored `**/*.png`, which prevents these required assets from being committed.

We explicitly *unignore* PNGs under:

- `models/assets/`
- `models/model_phonebot/assets/`
- `models/model_phonebot_fred_v2/assets/`

so these required files can be tracked.


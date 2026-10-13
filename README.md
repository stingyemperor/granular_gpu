# Granular GPU

A CUDA simulation of granular materials (sand, gravel) with adaptive sampling.
Coarse simulation particles are merged into heavier ones in the interior of the
material and split again near the surface. A dense set of fine "upsampled"
particles follows the coarse ones for rendering and export.

## Requirements

- NVIDIA GPU (compute capability 7.5 or newer) and the CUDA Toolkit (12.x)
- CMake ≥ 3.20 and a C++17 compiler
- OpenGL, GLEW, freeglut, nlohmann-json

On Ubuntu/Debian:

```sh
sudo apt install cmake ninja-build libglew-dev freeglut3-dev nlohmann-json3-dev
```

## Building

```sh
git clone --recursive <repo-url> granular_gpu
cd granular_gpu
# if you cloned without --recursive:
git submodule update --init --recursive

cmake -S . -B build -G Ninja          # Release by default
cmake --build build
```

By default the build targets GPU architectures 75, 80, 86 and 89. To build only
for your own GPU, which is faster, add `-DCMAKE_CUDA_ARCHITECTURES=native` to
the first cmake command.

## Running

```sh
./build/granular_gpu              # box scene
./build/granular_gpu funnel       # a scene from scenes/ by name
./build/granular_gpu my/scene.json
```

You can start the program from any directory. Shaders, scenes and obj files are
found relative to the source checkout.

### Controls

Everything is in the control panel on the left. You can also use these keys:

| Key            | Action                              |
| -------------- | ----------------------------------- |
| Space          | play / pause                        |
| N              | advance one step                    |
| E              | explosion (push particles outwards) |
| P              | show / hide coarse particles        |
| Left drag      | rotate the view                     |
| Wheel, `,` `.` | zoom                                |
| R              | reset the camera                    |
| H              | hide / show the panel               |
| F11            | fullscreen                          |
| Q, Esc         | quit                                |

Panel sections:

- **Simulation**: pick a scene, play/pause/step/reset. Under *Parameters* you
  can edit dt, gravity, adaptive sampling, the particle grid and the upsample
  count. Edits take effect when you press **Reset**.
- **Statistics**: particle counts, merges and splits in the last step, and a
  plot of GPU time per step.
- **Display**: show or hide each particle set, choose the coloring (surface,
  distance to surface, or mass), point size and background.
- **Interaction**: the explosion force. In the excavator scene, a toggle for
  the bucket animation.
- **Output**: per-frame exports and the stats files (see below).

## Scenes

| File                  | Description                                          |
| --------------------- | ---------------------------------------------------- |
| `box.json`            | block of sand collapsing in a closed box             |
| `corner.json`         | collapse against three walls                         |
| `piling.json`         | pile on a floor                                      |
| `funnel.json`         | sand flowing through a funnel (`obj/funnel.txt`)     |
| `excavator.json`      | bucket digging through sand (`obj/bucket.txt`)       |
| `box_opening_*.json`  | small test setups                                    |

### Scene file reference

All keys except `scene` are optional. Missing keys fall back to the defaults in
`src/SceneConfig.hpp`.

```jsonc
{
  "scene": "box",              // piling | box | excavator | funnel | corner | box_opening
  "space_size": {"x": 2.0, "y": 1.8, "z": 1.8},   // simulation domain
  "dt": 0.002,                 // time step (s)
  "gravity": {"x": 0, "y": -9.8, "z": 0},
  "density": 238732.4146,
  "cell_length": 0.0404,       // neighbor-search cell size

  "particle_grid": {"x": 50, "y": 50, "z": 40},   // initial particles per axis
  "init_spacing": 0.032,       // distance between initial particles
  "particle_origin": {"x": 0.27, "y": 0.13, "z": 0.17},
  "particle_translation": {"x": 0, "y": 0, "z": 0},
  "boundary_translation": {"x": 0, "y": 0, "z": 0},

  "particle_radius": 0.01,     // coarse particle size
  "upsampled_radius": 0.005,   // fine particle size
  "upsample_count": 25,        // fine particles per coarse particle

  "adaptive": true,            // merge/split coarse particles
  "obj_file": "obj/bucket.txt",// extra boundary points, one "x y z" per line
  "animate_obj": false,        // excavator: start with the bucket animation on

  "output": {
    "dir": "output/box",
    "stats_csv": true,
    "frame_times": true,
    "export": {
      "upsampled": false,
      "upsampled_centered": false,
      "upsampled_csv": false,
      "by_mass": false,
      "animated_boundary": false,
      "animated_boundary_centered": false,
      "start_frame": 0
    }
  }
}
```

Relative paths (`obj_file`, `output.dir`) are resolved against the project root.

## Output

Files are written under `output.dir`, which is `output/<scene>/` by default.

| Export option                | Files                                               |
| ---------------------------- | --------------------------------------------------- |
| `upsampled`                  | `upsampled_{adaptive,normal}/<frame>.vtk`           |
| `upsampled_centered`         | `upsampled_centered_{adaptive,normal}/<frame>.vtk`  |
| `upsampled_csv`              | `upsampled_csv_{adaptive,normal}/<frame>.csv`       |
| `by_mass`                    | `mass_{light,medium,heavy}/<frame>.vtk`             |
| `animated_boundary`          | `obj_boundary/<frame>.vtk`                          |
| `animated_boundary_centered` | `obj_boundary_centered/<frame>.vtk`                 |

- "Centered" exports move the fine particles' centroid to the origin and lift
  everything so the lowest particle sits at height 0.
- VTK files swap y and z, so "up" is +z in ParaView. Open them as a file series
  in ParaView and show them with the *Points* or *Point Gaussian*
  representation.
- `stats.csv` (merges and splits per adaptive step) and `frame_times.txt` (GPU
  ms per step) are written when you pause, change scene, reset or quit.

## Code layout

```
src/
  main.cu             entry point: command line, GLUT/GLEW/ImGui setup
  App.*               application state, input handling, simulation loop
  Ui.cu               ImGui control panel
  Renderer.*          CUDA–OpenGL interop and point-sprite rendering
  SceneConfig.*       scene file loading
  SceneBuilder.*      builds particles, walls and obj boundaries for a scene
  Animation.*         scripted bucket motion (excavator)
  Export.*            VTK/CSV export and stats files
  GranularSystem.*    per-step pipeline: neighbor search, solver, surface
  Solver.*            position-based granular solver, adaptive sampling,
                      upsampled particle update
  GranularParticles.hpp, Particles.hpp, DArray.hpp   GPU particle storage
  shaders/            point-sprite shaders
```

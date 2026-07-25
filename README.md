# Portal Electric Potential Engine

A 2D interactive simulation of electric potential with portals. The engine solves the Laplace equation for the electric potential using either the **Method of Moments (MOM)** with point matching or **Successive Over-Relaxation (SOR)**. See `/explanation/report.pdf` for the full derivation.

The engine ships with a `pygame` UI for building and interacting with a scene: potential anchors, boundaries (Dirichlet or Neumann), and portals. It also simulates the effect of the resulting field on material objects and test charges, so the hypothetical effects of a portal on electric potential can be observed directly. A set of premade scenarios is included to demonstrate this that are described on [Results](#Results).

This project was built for the final assignment of the Electrodynamics course at the National University of Colombia, taught by Ph.D. Juan Domingo Baena. The goal is to analyze how a hypothetical portal would affect electric potential, under a set of axioms detailed in `/explanation/report.pdf`.

<img width="1240" height="990" alt="Image" src="https://github.com/user-attachments/assets/d882b51a-6deb-4f6d-aeba-995c29da66cd" />

## Table of Contents

- [Quick Start](#quick-start)
- [Methodology](#methodology)
- [Physics](#physics)
- [Results](#Results)
- [Project Architecture](#project-architecture)
- [Field Objects](#field-objects)
- [Masks](#masks)
- [Simulation Parameters](#simulation-parameters)
- [Controls](#controls)
- [Color Schemes](#color-schemes)
- [Writing Scenes](#writing-scenes)

---

## Quick Start

**Requirements:** Python 3.9+, numpy 2.0.2+, pygame 2.6.1+

**To run normally, clone the repository (or just the `./src` folder), then launch `main.py` from the `./src` directory:**

```bash
pip install numpy pygame
python main.py
```

To run a different scene, open `main.py` and uncomment the desired line:

```python
def main() -> None:
    sim = example_couple_portals()   # <- active scene
    # sim = example_advanced()
    # sim = example_couple_circles()
    sim.run()
```

or switch the scene in the inspector.

---

# Methodology 

It builds on and extends the work of @ZinCin, adapting it to an electrostatics context by adding the MOM solver and new object classes.

The physics engine follows an object-oriented design, with the `pygame` UI layered on top for interaction; the simulation can also run headless, without the UI. Aside from portals, objects behave as they would in any potential-field simulation: boundaries, fixed potentials, and conductors affect the field by imposing a condition on the pixels they occupy, while material objects and test charges simply move along the gradient without affecting the potential themselves, since modeling that feedback would require electrodynamics effects outside the scope of this simulation.

For interaction with material objects and test charges, each portal has a region on one side that acts as its teleport trigger: any part of an object entering that region reappears at the paired portal. Portals are therefore one-directional for teleportation purposes, which is sufficient for the scope of this project.

---


## Physics

The simulation solves the Laplace equation:

```
∇²φ = 0
```

Method: **Red-Black Gauss-Seidel SOR** — alternates between "red" and "black" checkerboard cells, which doubles convergence speed compared to plain Jacobi.

### Solvers

Two different ways of solving for the same field are available, and only one is active per simulation.

**SOR** is an iterative relaxation that re-evaluates the whole grid every step, so the field, and anything coupled to it, keeps updating as the simulation runs. Portal potential is handled by averaging each pair's pixels on every step.

**MOM** (Method of Moments) discretizes the boundaries into segments and solves a single dense linear system for the potential once, at scene setup; the resulting field is never recomputed afterward. Portal pairs are instead assigned a shared, unknown potential that the linear system solves for directly. This makes MOM a poor fit for anything meant to update the field on the fly, such as a `ConductorObject`, whose potential is supposed to re-equalize to its neighbors every step: doing that correctly under MOM would mean re-solving the whole boundary system on every frame, which is too expensive to run in real time, so this update currently doesn't happen. Because of this, when MOM is active SOR does not run at all.

#### MOM edge fringing

MOM's boundary segments use the free-space log kernel and have no notion of the simulation box's side walls, unlike SOR, which imposes a Neumann (zero-flux) condition on the left/right edges. A Dirichlet segment that spans the full grid width, such as a top or bottom anchor, therefore curves near its two ends instead of staying flat, the fringing effect of a finite capacitor plate. Two independent ways to remove this are available, and can be combined:

- **Method of images** — `Simulation(..., solver_mode="mom", mom_images=True)`. Mirrors every boundary segment's contribution across both vertical walls with same-sign (Neumann) images, reproducing SOR's side-wall condition directly in the solver. The reflection series is truncated to `n_images` reflections per wall (`MOMSolver2D.__init__`, default `1`); no unknowns are added to the linear system, only extra kernel evaluations per matrix entry. This is the only option for boundaries whose extent matters and can't be resized, such as portals or conductors.
- **Oversized anchors** — a scene-authoring choice rather than a solver flag: build an anchor's mask wider than the visible grid, e.g. `RectangleMask(-2*W, 3*W, 0, 1)` instead of `RectangleMask(0, W, 0, 1)`, so its ends fringe far outside the rendered area. Cheaper to reason about and needs no code change, but only makes sense for anchors, since it adds more boundary segments to the MOM system (slower solve) and doesn't apply to objects whose visible extent is part of the scene.

### Boundary Conditions

| Boundary | Type | Value |
|---|---|---|
| Top of grid | Dirichlet | φ = 1.0 |
| Bottom of grid | Dirichlet | φ = 0.0 |
| Left / right | Neumann | ∂φ/∂n = 0 |

The top and bottom edges are constant source and sink. The side boundaries are transparent: the field flows freely left and right. (This implements gravity similar to Earth's.)


### Parameter ω (SOR)

Controls convergence speed:

- `ω = 1.0` — slow but accurate convergence
- `ω = 1.5–1.9` — fast but less accurate
- `ω → 2.0` — near-instant but unstable

---
## Results

---
## Project Architecture

```
main.py          — entry point
scenes.py        — predefined scenes (examples demonstrating how portals work)
simulation.py    — main class, game loop, rendering, UI, teleportation logic
physics.py       — SOR solver
mom_solver.py    — MOM linear system assembly and solve
mom_mesh.py      — discretizes a mask's boundary into segments for MOM
portals.py       — field object classes
masks.py         — geometric masks
test_charge.py   — point probe charges (Velocity Verlet integrator)
colors.py        — color schemes
ui.py            — sidebar widgets
test_charge.py   — TestCharge class (probe particle dynamics)
```

> `physics.py` has no Pygame dependency, for use in standalone tests.

---

## Field Objects

All objects are passed into `Simulation(*field, ...)` in any order and quantity.

---

### `PotentialAnchor(mask, potential_value)`

An anchor: rigidly fixes the potential in its area and holds it constant. Creates a field source or sink at an arbitrary point on the grid.

```python
# Source at center-top
PotentialAnchor(CircleMask(60, 20, 5), potential_value=1.0)

# Sink at center-bottom
PotentialAnchor(CircleMask(60, 80, 5), potential_value=0.0)
```

An anchor with φ above the background creates a potential gradient hill; with φ below — a potential gradient valley. The field decays from the anchor in all directions according to the Laplace equation.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `mask` | `Mask` | Shape of the anchor area |
| `potential_value` | `float` | Fixed φ value ∈ [0, 1] |
| `color` | `tuple` | RGB color |
| `active` | `bool` | Enabled / disabled |

---

### `FixedPotentialPortal(mask, potential_value)`

Functionally identical to `PotentialAnchor` — fixes the potential in its area. Used for potential shielding.

```python
FixedPotentialPortal(RectangleMask(20, 100, 30, 33), potential_value=0.8)
```

---

### `CouplePortal(p1, p2)`

A pair of linked portals with **equal potential**. At each step, the potential across the entire linked area is averaged.

```python
p1 = Portal(RectangleMask(25, 75, 25, 26), color=(255, 153, 0))
p2 = Portal(RectangleMask(25, 75, 94, 95), color=(0, 204, 255))
couple = CouplePortal(p1, p2)
```

### `MultiPortal((p1, p2, p3, ...))`

A few of linked portals with **equal potential**. At each step, the potential across the entire linked area is averaged.

```python
p1 = Portal(RectangleMask(40, 60, 30, 30), (255, 0, 0))
p2 = Portal(RectangleMask(40, 60, 60, 60), (0, 255, 0))
p3 = Portal(RectangleMask(40, 60, 90, 90), (0, 0, 255))
multi = MultiPortal((p1, p2, p3))
```

> ⚠️ `MultiPortal` only couples potential. Teleportation of `MaterialObject`s and `TestCharge`s is wired to `CouplePortal` pairs, so objects will pass through a `MultiPortal` without being teleported.

**`Portal` Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `mask` | `Mask` | Shape of the portal |
| `color` | `tuple` | RGB color |
| `active` | `bool` | Enabled / disabled |

---

### `MaterialObject(mask, ...)`

A solid obstacle, currently used for teleportation and rendering only. `PhysicsEngine` has an `ignore_material_objects` flag (default `True`, not yet exposed through `Simulation`) that, if set to `False`, excludes the object's cells from SOR so the field wraps around it from outside, with the potential inside fixed at its initial value.

Once an object can move, its mask is baked from an analytic shape (circle, rectangle, ...) into a plain boolean pixel grid (`ArrayMask`). Teleportation works directly on that grid: the part of the mask sitting in a portal's back region is cut out, shifted to the paired portal's position, and merged back with the rest, so the object ends up as two separated pixel groups that still count as one `MaterialObject`. A geometric mask cannot represent that split shape, which is why the pixel grid representation is required for teleportation.

```python
# Pinned block
MaterialObject(
    RectangleMask(40, 80, 40, 60),
    color=(180, 180, 180),
    pinned=True,
    label="Block"
)

# Free-floating ball
MaterialObject(
    CircleMask(cx=60, cy=50, radius=8),
    pinned=False,
    mass=1.5
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mask` | `Mask` | — | Shape of the object |
| `color` | `tuple` | `(180, 180, 180)` | RGB color |
| `pinned` | `bool` | `False` | If `True`, cannot be dragged |
| `label` | `str` | `"Object"` | Name in UI |
| `mass` | `float` | `1.0` | Mass |
| `active` | `bool` | `True` | Enabled / disabled |

---

### `ConductorObject(mask, ...)`

A floating conductor. At each step, the potential inside the object is equalized to the average potential of its outer neighboring cells. Creates the field-line distortion characteristic of a conductor.

```python
ConductorObject(
    CircleMask(cx=60, cy=50, radius=10),
    color=(220, 180, 60),
    pinned=True,
    label="Sphere"
)
```

Parameters are identical to `MaterialObject`.
`to_mom_boundary(grid_array, dx=1.0)` converts the conductor's mask into a `BoundaryMesh` for the MOM solver.

### `TestCharge(x, y, ...)`

A point probe charge that responds to the electric field but never contributes to it. Its position and velocity evolve under Newton's second law with a **Velocity Verlet** integrator, which is symplectic and keeps the energy error bounded over long runs. The field is sampled by bilinear interpolation of the precomputed gradient arrays, so the trajectory is smooth even at sub-cell scales.

`TestCharge` is intentionally kept outside `Simulation.field`: the SOR engine never sees it, so it cannot perturb the potential. It is managed separately via `Simulation.test_charges` and updated after each SOR pass.

```python
TestCharge(x=30, 
           y=60, 
           charge=1.0, 
           mass=1.0, 
           color=(255, 255, 0)
)
```

The test charge can be added interactively from the INSPECTOR tab with **+ Test Charge**.



---


**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x`, `y` | `float` | — | Initial position (grid coordinates) |
| `vx`, `vy` | `float` | `0.0` | Initial velocity |
| `charge` | `float` | `1.0` | Probe charge `q` (sign sets force direction) |
| `mass` | `float` | `1.0` | Inertial mass `m` — sets `a = (q/m)E`, unrelated to gravity |
| `color` | `tuple` | `(255, 255, 0)` | RGB trail and dot color |
| `active` | `bool` | `True` | Enabled / disabled |
| `trail_len` | `int` | `200` | Number of past positions drawn as trail; `0` disables it |

**Boundary handling:** position is clamped to `[0, width−1] × [0, height−1]`; the velocity component pointing outward is zeroed on contact so the charge settles at the wall instead of accumulating outward momentum.

**Teleportation:** when the charge enters a portal's trigger region it is shifted to the paired portal by the same pixel offset used for `MaterialObject` teleportation, then nudged one step along its velocity vector to avoid immediate re-entry.

---

## Masks

Masks define the geometric shape of an object on the simulation grid. Any object accepts any mask.

### `RectangleMask(x_min, x_max, y_min, y_max)`

```python
RectangleMask(10, 90, 20, 21)   # thin horizontal strip
RectangleMask(0, 120, 0, 1)     # anchor along the entire top edge
```

### `CircleMask(cx, cy, radius)`

```python
CircleMask(cx=60, cy=50, radius=15)
```

### `PointMask(x, y)`

A single grid cell.

```python
PointMask(60, 50)
```

### `LineMask(x1, y1, x2, y2, thickness)`

A line segment between two points with a given thickness.

```python
LineMask(10, 10, 110, 90, thickness=2.0)
```

### `PolygonMask([(x0,y0), (x1,y1), ...])`

An arbitrary polygon.

```python
PolygonMask([(30, 20), (90, 20), (60, 80)])  # triangle
```

### `FunctionMask("expression")`

A mask defined by an arbitrary expression (function). Variables: `x`, `y`, `np`. The expression must return a boolean scalar or array.

```python
FunctionMask("(x - 60)**2 + (y - 50)**2 < 15**2")  # circle
FunctionMask("np.abs(x - 60) < 5")                  # vertical strip
FunctionMask("(x > 30) & (x < 90) & (y > 40) & (y < 60)")  # rectangle
```

> ⚠️ `PolygonMask` and `FunctionMask` are slower than the others due to complex calculations — avoid using them unless necessary.

---

## Simulation Parameters

```python
Simulation(
    *field,                          # field objects (any number)
    sim_width: int,                  # grid width in cells
    sim_height: int,                 # grid height in cells
    px_scale: float,                 # pixels per cell
    iterations_per_frame: int = 50,  # SOR iterations per frame
    diff_threshold: float = 1e-6,    # early stopping threshold
    view_mode: str = "potential",    # "E_magnitude" | "potential"
    show_vectors: bool = True,       # gradient vectors
    show_isolines: bool = True,      # isolines
    isoline_count: int = 10,         # number of isolines
    fps: int = 60,                   # target FPS
    sor_omega: float = 1.7,          # SOR parameter [1.0, 2.0)
    color_mapper = None,             # custom color scheme
)
```

**Recommended grid sizes:**

| Resolution | `px_scale` | Load |
|---|---|---|
| 80 × 80 | 8 | light |
| 120 × 120 | 6 | medium (default) |
| 200 × 150 | 4 | heavy |

---

## Controls

### Keyboard

| Key | Action |
|---|---|
| `M` | Toggle display mode: electric field magnitude \|E\| / potential |   
| `V` | Show / hide gradient vectors |
| `I` | Show / hide isolines |

### Mouse

| Action | Result |
|---|---|
| **LMB + drag** on a portal or object | Move the object |
| **RMB** on an object | Open the inspector in the SCENE panel |

### Sidebar

**SIMULATION tab** — rendering and physics parameters in real time:

- Display mode (force / potential)
- Vectors, isolines, isoline count
- Color scheme
- Iterations per frame, SOR ω

**INSPECTOR tab** — object management:

- Add portal, anchor, object, conductor, test charge
- Scene presets
- Selected object inspector (mask parameters, color, φ value)

---

## Color Schemes

Available in `COLOR_SCHEMES`, switchable in the SIMULATION panel:

| Name | Description |
|---|---|
| `Default` | Blue → green → yellow → red (for electric field magnitude) |
| `Potential` | Blue → green → red (for potential) |
| `Plasma` | Purple → pink → yellow |
| `Electric` | Black → blue → white |
| `Fire` | Black → red → gold → white |
| `Extra` | Blue → cyan → yellow → red |

A custom scheme is set via `color_mapper`:

```python
from colors import GradientColorMapper

my_mapper = GradientColorMapper([
    (0.0, (0,   0,  50)),   # dark blue at φ = 0
    (0.5, (255, 255, 0)),   # yellow at φ = 0.5
    (1.0, (255,  50, 50)),  # red at φ = 1
])

sim = Simulation(..., color_mapper=my_mapper)
```

---

## Writing Scenes

All scenes are defined in `scenes.py`. A minimal scene:

```python
from simulation import Simulation
from portals import PotentialAnchor, CouplePortal, Portal
from masks import RectangleMask, CircleMask

def my_scene() -> Simulation:
    W, H = 120, 120

    # Anchors define the background field
    top    = PotentialAnchor(RectangleMask(0, W, 0, 1),      1.0)
    bottom = PotentialAnchor(RectangleMask(0, W, H-1, H),    0.0)

    # Two linked portals
    p1 = Portal(CircleMask(30, 60, 10), color=(255, 153, 0))
    p2 = Portal(CircleMask(90, 60, 10), color=(0, 204, 255))

    return Simulation(
        top, bottom, CouplePortal(p1, p2),
        sim_width=W, sim_height=H,
        px_scale=6,
        iterations_per_frame=40,
        sor_omega=1.7,
    )
```

Then in `main.py`:

```python
from scenes import my_scene

def main():
    my_scene().run()
```

### Recipes

**Two anchors instead of grid boundary conditions:**

```python
# Source at top, sink at bottom — field directed downward
PotentialAnchor(RectangleMask(0, W, 0, 1),   1.0)
PotentialAnchor(RectangleMask(0, W, H-1, H), 0.0)
```

**Two parallel screens:**

```python
FixedPotentialPortal(RectangleMask(20, 100, 30, 33), 0.8)
FixedPotentialPortal(RectangleMask(20, 100, 67, 70), 0.2)
```

**Conductor in a screen field:**

```python
ConductorObject(CircleMask(60, 50, 10), pinned=True, label="Sphere")
```

**Non-conducting obstacle:**

```python
MaterialObject(RectangleMask(50, 70, 30, 70), pinned=True, label="Wall")
```

---

**For a detailed explanation of the simulation, read explanation.pdf**

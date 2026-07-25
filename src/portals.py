"""
portals.py - field object classes for the potential simulation.

Functions:
  Portal                basic portal with a mask and color
  FixedPotentialPortal  hard Dirichlet boundary condition (φ = const)
  CouplePortal          pair of portals with equal potential
  PotentialAnchor       background field anchor (φ = const)
  MaterialObject        solid object
  ConductorObject       solid conductor object (conducts potential)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

from masks import RectangleMask

if TYPE_CHECKING:
    from masks import Mask


class Portal:
    """
    Args:
        facing_positive: which side of the portal is the "working" (front)
            face - the side objects normally approach from. True = the
            positive-axis side (down, for a horizontal band; right, for a
            vertical band). False = the negative-axis side.
        back_depth: how far behind the portal (opposite the front face) the
            "already crossed through" trigger region extends. Large by
            default so it effectively reaches to the simulation edge.
        normal_axis: 'x' or 'y' - which axis the front/back split runs
            along. None = infer from the mask's bounding box (whichever
            side is thinner is treated as the normal axis); needed only for
            portals whose bounding box is square (e.g. CircleMask), where
            that inference is ambiguous.
    """

    def __init__(self, mask: "Mask",
                 color: Tuple[int, int, int],
                 active: bool = True,
                 facing_positive: bool = True,
                 back_depth: float = 1e6,
                 normal_axis: Optional[str] = None) -> None:
        self.mask   = mask
        self.color  = color
        self.active = active
        self.facing_positive = facing_positive
        self.back_depth = back_depth
        self.normal_axis = normal_axis

    def __repr__(self) -> str:
        return f"Portal(mask={self.mask!r}, color={self.color})"

    def __contains__(self, point: Tuple[float, float]) -> bool:
        return self.active and (point in self.mask)

    def get_mask(self, X: "np.ndarray", Y: "np.ndarray") -> "np.ndarray":
        if self.active:
            return self.mask(X, Y)
        return np.zeros(X.shape, dtype=bool)

    def back_region_mask(self, X: "np.ndarray", Y: "np.ndarray") -> "np.ndarray":
        """
        Boolean grid of the region behind this portal's working face.
        Anything found here has already fully crossed through the portal
        (not just touched its mouth) and should be teleported to the
        paired portal.
        """
        if not self.active:
            return np.zeros(X.shape, dtype=bool)

        size = self.mask.size()
        if size is None:
            return np.zeros(X.shape, dtype=bool)

        cx, cy = self.mask.center
        w, h = size
        x_min, x_max = cx - w / 2, cx + w / 2
        y_min, y_max = cy - h / 2, cy + h / 2

        axis = self.normal_axis or ("y" if w >= h else "x")
        if axis == "y":
            if self.facing_positive:
                region = RectangleMask(x_min, x_max, y_min - self.back_depth, y_min)
            else:
                region = RectangleMask(x_min, x_max, y_max, y_max + self.back_depth)
        else:
            if self.facing_positive:
                region = RectangleMask(x_min - self.back_depth, x_min, y_min, y_max)
            else:
                region = RectangleMask(x_max, x_max + self.back_depth, y_min, y_max)

        # Include the portal's own footprint: an object overlapping the
        # portal's visible surface has already crossed it, so it must not
        # be left resting on top of the portal graphic either.
        return region(X, Y) & ~self.get_mask(X, Y)


class FixedPotentialPortal(Portal):
    """φ = const (Dirichlet boundary condition)"""

    def __init__(self, mask: "Mask",
                 potential_value: float,
                 color: Optional[Tuple[int, int, int]] = None,
                 active: bool = True) -> None:
        if color is None:
            color = (220, 50, 50) if potential_value >= 0.5 else (50, 100, 220)
        super().__init__(mask, color, active)
        self.potential_value = potential_value

    def __repr__(self) -> str:
        return f"FixedPotentialPortal(φ={self.potential_value}, mask={self.mask!r})"


@dataclass
class CouplePortal:
    """Pair of portals with the same potential"""
    p1: Portal
    p2: Portal

    def __post_init__(self) -> None:
        s1, s2 = self.p1.mask.size(), self.p2.mask.size()
        if s1 is None or s2 is None:
            return  # not analytically comparable (e.g. FunctionMask side)
        if abs(s1[0] - s2[0]) > 1e-6 or abs(s1[1] - s2[1]) > 1e-6:
            raise ValueError(
                f"CouplePortal requires p1 and p2 masks to have the same "
                f"size; got {s1} vs {s2}")

    def get_combined_mask(self, X: "np.ndarray",
                          Y: "np.ndarray") -> "np.ndarray":
        return self.p1.get_mask(X, Y) | self.p2.get_mask(X, Y)

    def __repr__(self) -> str:
        return f"CouplePortal(p1={self.p1!r}, p2={self.p2!r})"


@dataclass
class MultiPortal:
    """Several portals with the same potential"""
    args: tuple[Portal, ...]

    def get_combined_mask(self, X: "np.ndarray",
                          Y: "np.ndarray") -> "np.ndarray":
        if not len(self.args):
            return np.zeros(X.shape, dtype=bool)

        local_mask = self.args[0].get_mask(X, Y)
        for p in self.args[1:]:
            local_mask |= p.get_mask(X, Y)
        return local_mask

    def __repr__(self) -> str:
        return f"MultiPortal(Portal count: {len(self.args)})"


class PotentialAnchor:
    """
    Potential anchor: a region with fixed potential

    Used to define the background field
    """

    def __init__(self, mask: "Mask",
                 potential_value: float,
                 color: Optional[Tuple[int, int, int]] = None,
                 active: bool = True) -> None:
        self.mask            = mask
        self.potential_value = potential_value
        self.active          = active
        if color is None:
            color = (255, 210, 0) if potential_value >= 0.5 else (0, 190, 230)
        self.color = color

    def __repr__(self) -> str:
        return f"PotentialAnchor(φ={self.potential_value:.2f}, mask={self.mask!r})"

    def __contains__(self, point: Tuple[float, float]) -> bool:
        return self.active and (point in self.mask)

    def get_mask(self, X: "np.ndarray",
                 Y: "np.ndarray") -> "np.ndarray":
        if self.active:
            return self.mask(X, Y)
        return np.zeros(X.shape, dtype=bool)


class MaterialObject:
    """
    Solid object

    Args:
        mask:   shape of the object
        color:  display color
        pinned: if True - cannot be moved
        label:  name shown in the UI
        active: if False - ignored
        show_flux: bool - if true it show in every frame the flux
    """

    def __init__(self, mask: "Mask",
                 color: Tuple[int, int, int] = (180, 180, 180),
                 pinned: bool = False,
                 label: str = "Object",
                 mass: float = 1.0,
                 charge: float = 1.0,
                 active: bool = True,
                 show_flux: bool = False) -> None:
        self.mask   = mask
        self.color  = color
        self.pinned = pinned
        self.label  = label
        self.mass   = float(mass)
        self.charge = float(charge)
        self.active = active
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.show_flux = show_flux

    def __repr__(self) -> str:
        return (f"MaterialObject({self.label!r}, pinned={self.pinned}, "
                f"charge={self.charge}, mask={self.mask!r})")

    def __contains__(self, point: Tuple[float, float]) -> bool:
        return self.active and not self.pinned and (point in self.mask)

    def get_mask(self, X: "np.ndarray", Y: "np.ndarray") -> "np.ndarray":
        if self.active:
            return self.mask(X, Y)
        return np.zeros(X.shape, dtype=bool)
    
    def compute_flux(self, X: "np.ndarray", Y: "np.ndarray",
                      grad_x: "np.ndarray", grad_y: "np.ndarray",
                      portals_mask: "np.ndarray") -> float:
        """
        Flujo del campo E = -∇φ a través de la frontera del objeto:

            Φ = ∮ E · n̂ dl

        Procedimiento (todo vectorizado sobre la grilla):
          1. Se toma la máscara binaria del objeto (1 dentro, 0 fuera).
          2. Su gradiente (np.gradient) apunta de afuera hacia adentro,
             así que la normal saliente n̂ es ese gradiente invertido y
             normalizado.
          3. Se restringe a los píxeles de la frontera interior del
             objeto (norma no nula y dentro de la máscara).
          4. Se evalúa E = -∇φ (grad_x, grad_y del PhysicsEngine) en esos
             píxeles y se proyecta sobre n̂.
          5. Se integra sumando sobre el borde, con dl = 1 píxel de la
             grilla como elemento de línea discreto.

        Args:
            X, Y:            grillas de coordenadas (mismas que usa get_mask)
            grad_x, grad_y:  gradiente de potencial (PhysicsEngine.grad_x/grad_y)

        Returns:
            Flujo neto (float) saliendo del objeto. Positivo = flujo neto
            hacia afuera; negativo = flujo neto entrando.
        """
        if not self.active:
            return 0.0

        m = self.get_mask(X, Y).astype(np.float64)
        if not np.any(m):
            return 0.0

        dmy, dmx = np.gradient(m)
        nx, ny = -dmx, -dmy
        norm = np.hypot(nx, ny)

        boundary = (norm > 1e-9) & (m > 0.5)
        boundary &= ~portals_mask  # ignore any pixels that are part of a portal
        if not np.any(boundary):
            return 0.0

        nx_b = nx[boundary] / norm[boundary]
        ny_b = ny[boundary] / norm[boundary]

        Ex = -grad_x[boundary]
        Ey = -grad_y[boundary]

        dl = 1.0
        return float(np.sum((Ex * nx_b + Ey * ny_b) * dl))


class ConductorObject:
    """
    Floating conductor

    Args:
        mask:   shape of the object
        color:  display color
        pinned: if True - cannot be dragged
        label:  name shown in the UI
        active: if False - ignored
    """

    def __init__(self, mask: "Mask",
                 color: Tuple[int, int, int] = (220, 180, 60),
                 pinned: bool = False,
                 label: str = "Conductor",
                 mass: float = 1.0,
                 active: bool = True) -> None:
        self.mask   = mask
        self.color  = color
        self.pinned = pinned
        self.label  = label
        self.mass   = float(mass)
        self.active = active
        self.vx: float = 0.0
        self.vy: float = 0.0

    def __repr__(self) -> str:
        return f"ConductorObject({self.label!r}, pinned={self.pinned}, mask={self.mask!r})"

    def __contains__(self, point: Tuple[float, float]) -> bool:
        return self.active and not self.pinned and (point in self.mask)

    def get_mask(self, X: "np.ndarray", Y: "np.ndarray") -> "np.ndarray":
        if self.active:
            return self.mask(X, Y)
        return np.zeros(X.shape, dtype=bool)
    def to_mom_boundary(self, grid_array, dx=1.0):
        from mom_mesh import BoundaryMesh
        # grid_array es el bool 2D que ya usa tu mask internamente
        return BoundaryMesh(grid_array, potential_value=0.5, dx=dx)

"""
simulation.py - main simulation file that ties all the logic together
"""


import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame
from typing import Dict, List, Optional, Tuple
from test_charge import TestCharge

from portals import *
from masks import *
from physics import PhysicsEngine
from colors import GradientColorMapper, default_color_mapper, COLOR_SCHEMES
from ui import (
    TabbedPanel, Label, Button, Toggle, Stepper, Slider,
    Divider, SectionHeader,
    CLR_BG, CLR_TEXT, CLR_ACCENT, CLR_TEXT_DIM, CLR_BORDER,
    _init_fonts,
)


UI_WIDTH = 250


class Simulation:
    def __init__(self, *field,
                 sim_width: int,
                 sim_height: int,
                 px_scale: float,
                 solver_mode: str = "sor",
                 mom_images: bool = False,
                 iterations_per_frame: int = 50,
                 diff_threshold: float = 1e-4,
                 view_mode: str = "potential",
                 show_vectors: bool = False,
                 show_isolines: bool = False,
                 isoline_count: int = 10,
                 fps: int = 60,
                 sor_omega: float = 1.7,
                 color_mapper: Optional[GradientColorMapper] = None) -> None:

        pygame.init()
        pygame.display.set_caption("Portals Engine")

        self.sim_width = sim_width
        self.sim_height = sim_height
        self.px_scale = px_scale
        self.field: List = list(field)
        self.damping = False
        self.test_charges: List = []

        self.sor_omega = sor_omega
        self.iterations_per_frame = iterations_per_frame
        self.diff_threshold = diff_threshold
        self.diff: float = 0.0
        self.total_iterations: int = 0

        self.view_mode = view_mode
        self.show_vectors = show_vectors
        self.show_isolines = show_isolines
        self.isoline_count = isoline_count

        self._color_scheme_names = list(COLOR_SCHEMES.keys())
        self._color_scheme_idx = 0
        self.color_mapper = color_mapper or default_color_mapper()

        win_w = int(sim_width * px_scale) + UI_WIDTH
        win_h = int(sim_height * px_scale)
        self.screen = pygame.display.set_mode((win_w, win_h))
        self.sim_surface = pygame.Surface((int(sim_width * px_scale), win_h))
        self.clock = pygame.time.Clock()
        self.fps = fps

        self.Y, self.X = np.ogrid[:sim_height, :sim_width]
        self._bake_movable_masks()

        self.solver_mode = solver_mode
        self.mom_images = mom_images

        self._engine = PhysicsEngine(
            sim_width, sim_height, self.field, sor_omega=self.sor_omega)
        self.potential = self._engine.potential
        self.grad_x = self._engine.grad_x
        self.grad_y = self._engine.grad_y
        self.E_magnitude = self._engine.E_magnitude

        # MOM: se resuelve una sola vez en setup
        self._mom_status = "sin resolver"
        if self.solver_mode == "mom":
            self._mom_setup()

        # Drag
        self._dragging_mask: Optional[Mask] = None
        self._dragging_obj = None

        # Isoline cache
        self._isolines_cache: Optional[np.ndarray] = None
        self._isolines_dirty = True

        # Rendering buffers
        sim_w_px = int(sim_width  * px_scale)
        sim_h_px = int(sim_height * px_scale)
        self._sim_w_px = sim_w_px
        self._sim_h_px = sim_h_px
        self._px_int = max(1, int(px_scale))

        # Field: RGB Surface without alpha (faster than make_surface + transform.scale)
        self._field_surf = pygame.Surface((sim_w_px, sim_h_px))

        self._overlay_surf = pygame.Surface((sim_w_px, sim_h_px), pygame.SRCALPHA)

        self._portal_render_cache: Optional[list] = None
        self._portal_render_dirty = True

        self._fonts = _init_fonts()
        self._panel = self._build_panel()

        # Data export (snapshot / recording)
        self._recording: bool = False
        self._output_dir_cache: Optional[Path] = None
        self._last_snapshot_path: Optional[Path] = None
        self._reset_recording_state()

    def _mom_setup(self) -> None:
        from mom_mesh import BoundaryMesh
        from mom_solver import MOMSolver2D

        W, H = self.sim_width, self.sim_height
        meshes = []
        coupled_pairs = []

        for obj in self.field:
            if isinstance(obj, CouplePortal):
                # Los dos portales comparten phi desconocido
                m1 = BoundaryMesh(obj.p1.mask, W, H, potential_value=0.0)
                m2 = BoundaryMesh(obj.p2.mask, W, H, potential_value=0.0)
                coupled_pairs.append((m1, m2))
            elif isinstance(obj, MultiPortal):
                # N portales comparten phi desconocido (grupo neutro)
                group = [BoundaryMesh(p.mask, W, H, potential_value=0.0)
                         for p in obj.args]
                coupled_pairs.append(group)
            elif hasattr(obj, 'potential_value') and hasattr(obj, 'mask'):
                meshes.append(BoundaryMesh(obj.mask, W, H, obj.potential_value))

        if not meshes and not coupled_pairs:
            self._mom_status = "sin objetos con phi fijo"
            return

        solver = MOMSolver2D(
            meshes, coupled_pairs,
            image_walls=(0, W) if self.mom_images else None,
        )
        solver.build_and_solve()

        xs = np.arange(W, dtype=float)
        ys = np.arange(H, dtype=float)
        grid_x, grid_y = np.meshgrid(xs, ys)
        phi = solver.compute_phi_grid(grid_x, grid_y)

        phi_min, phi_max = phi.min(), phi.max()
        if phi_max - phi_min > 1e-10:
            phi = (phi - phi_min) / (phi_max - phi_min)

        self.potential[:] = phi
        self._engine.compute_gradients()
        self.grad_x = self._engine.grad_x
        self.grad_y = self._engine.grad_y
        h = getattr(self._engine, "height", 1.0)
        self.g_force = np.sqrt(self.grad_x ** 2 + self.grad_y ** 2) * h
        self.diff = 0.0

        self._mom_status = f"OK ({solver.N} seg.)"
        # El campo cambió: forzar redibujado de isolíneas y overlays
        self._isolines_dirty = True
        self._portal_render_dirty = True

    def _recompute_mom(self) -> None:
        """Callback del botón 'Recalcular MOM': vuelve a resolver el
        sistema MOM tomando las posiciones actuales de cargas/portales."""
        self._mom_setup()

    def update_physics(self) -> None:
        if self.solver_mode == "mom":
            # MOM never calls engine.step()/run_steps(), so the engine's
            # portal/material mask cache (built lazily inside step()) is
            # never populated - rebuild it by hand so teleport can see it
            if self._engine._cache_dirty:
                self._engine._rebuild_cache()
            self._teleport_material_objects()
            self._update_material_dynamics()
            self._record_frame()
            self._update_test_charges()
            return  # MOM ya resolvió en setup, no hay iteraciones
        self._engine.sor_omega = self.sor_omega
        self.diff = self._engine.run_steps(
            self.iterations_per_frame, self.diff_threshold)
        self.total_iterations += self.iterations_per_frame
        self._engine.compute_gradients()
        self.potential = self._engine.potential
        self.grad_x = self._engine.grad_x
        self.grad_y = self._engine.grad_y
        self.E_magnitude = self._engine.E_magnitude
        self._update_material_dynamics()
        self._teleport_material_objects()
        self._update_test_charges()
        self._isolines_dirty = True
        self._record_frame()

    def _update_material_dynamics(self) -> None:
        """
        Updates the velocity and position of dynamic MaterialObjects
        """

        needs_invalidate = False
        for obj in self.field:
            if not isinstance(obj, MaterialObject):
                continue
            if obj.pinned or not obj.active:
                continue
            mask = obj.get_mask(self.X, self.Y)
            if not np.any(mask):
                continue

            # Average gradient value inside the object
            qm = obj.charge / obj.mass
            fx = float(-qm * np.mean(self.grad_x[mask]))
            fy = float(-qm * np.mean(self.grad_y[mask]))

            if self.damping:
                # Damping factor to prevent oscillations
                obj.vx = (obj.vx + fx) * 0.92  # damping
                obj.vy = (obj.vy + fy) * 0.92
            else:
                obj.vx = (obj.vx + fx)   # damping
                obj.vy = (obj.vy + fy) 

            # Clamp the maximum speed
            speed = (obj.vx**2 + obj.vy**2) ** 0.5
            max_speed = 100000.0
            if speed > max_speed:
                obj.vx = obj.vx / speed * max_speed
                obj.vy = obj.vy / speed * max_speed

            if abs(obj.vx) > 0.01 or abs(obj.vy) > 0.01:
                obj.mask.translate(obj.vx, obj.vy)
                needs_invalidate = True

        if needs_invalidate:
            self._invalidate_caches()
            self._isolines_dirty = True

    def _teleport_material_objects(self) -> None:
        """
        Splits MaterialObjects crossing a portal: the part of the mask that
        has fully crossed into a portal's back region (behind its working
        face, per Portal.facing_positive/back_depth) is removed and
        reappears at the paired portal, while the rest of the object stays
        put - both pieces remain one object
        """
        # Back-region masks only depend on the portals themselves, not on
        # any particular object - computed once per couple per frame rather
        # than once per (object, couple) pair
        back_cache = [
            (couple_obj.p1, couple_obj.p2,
             couple_obj.p1.back_region_mask(self.X, self.Y),
             couple_obj.p2.back_region_mask(self.X, self.Y))
            for couple_obj, _, _ in self._engine._active_couples_cache or []
        ]

        for obj in self.field:
            if not isinstance(obj, MaterialObject):
                continue
            if obj.pinned or not obj.active:
                continue

            obj_bool = obj.get_mask(self.X, self.Y)
            if not np.any(obj_bool):
                continue

            for p1, p2, back1, back2 in back_cache:
                intersect1, intersect2 = back1 & obj_bool, back2 & obj_bool
                has1, has2 = np.any(intersect1), np.any(intersect2)
                if not has1 and not has2:
                    continue

                # If the object overlaps both mouths at once (mid-transit
                # continuation, or a large object spanning both), the
                # larger-overlap side is treated as the active source
                if has1 and has2:
                    if np.sum(intersect1) >= np.sum(intersect2):
                        src, dst, intersection = p1, p2, intersect1
                    else:
                        src, dst, intersection = p2, p1, intersect2
                elif has1:
                    src, dst, intersection = p1, p2, intersect1
                else:
                    src, dst, intersection = p2, p1, intersect2

                remainder = obj_bool & ~intersection
                shift_x, shift_y = self._compute_teleport_shift(src, dst)

                shifted_piece = ArrayMask._shift_grid(intersection, shift_x, shift_y)
                new_grid = remainder | shifted_piece

                if not np.any(new_grid):
                    break  # defensive: don't assign an empty mask

                obj.mask.set(new_grid)

                self._invalidate_caches()

                
                break
        # ---- TestCharge teleportation ----
        for q in self.test_charges:
            if not q.active:
                continue

            for p1, p2, back1, back2 in back_cache:
                xi = max(0, min(back1.shape[1] - 1, int(round(q.x))))
                yi = max(0, min(back1.shape[0] - 1, int(round(q.y))))

                in1 = bool(back1[yi, xi])
                in2 = bool(back2[yi, xi])

                if not in1 and not in2:
                    continue

                shift_x, shift_y = self._compute_teleport_shift(
                    p1, p2) if in1 else self._compute_teleport_shift(p2, p1)

                q.x = max(0.0, min(self.sim_width  - 1.0, q.x + shift_x))
                q.y = max(0.0, min(self.sim_height - 1.0, q.y + shift_y))

                # empujar un paso en dirección de la velocidad para
                # evitar re-teletransportación en el frame siguiente
                speed = (q.vx**2 + q.vy**2) ** 0.5
                if speed > 1e-6:
                    q.x = max(0.0, min(self.sim_width  - 1.0,
                                       q.x + q.vx / speed * 2.0))
                    q.y = max(0.0, min(self.sim_height - 1.0,
                                       q.y + q.vy / speed * 2.0))

                q._initialized_accel = False
                q.reset_trail()
                break
            

    def _compute_teleport_shift(self, src, dst) -> Tuple[int, int]:
        """Pixel shift from src's footprint center to dst's"""
        sx, sy = src.mask.center
        dxc, dyc = dst.mask.center
        return int(round(dxc - sx)), int(round(dyc - sy))

    def _bake_movable_masks(self) -> None:
        """
        Replaces MaterialObject/ConductorObject masks with an ArrayMask
        baked from their current analytic shape, once a coordinate grid
        exists. No-op for objects already baked.
        """
        for obj in self.field:
            if isinstance(obj, (MaterialObject, ConductorObject)) \
                    and not isinstance(obj.mask, ArrayMask):
                obj.mask = ArrayMask(obj.mask(self.X, self.Y))

    def _update_test_charges(self) -> None:
        """
        Advances all test charges using the already-solved field.
        Charges never modify self.field or invalidate the engine cache.
        """
        for q in self.test_charges:
            q.update(self._engine, dt=5.0)

    def _invalidate_caches(self) -> None:
        """Invalidates the physics cache and the portal render cache in one call"""
        self._engine.invalidate_mask_cache()
        self._portal_render_dirty = True
        self._isolines_dirty      = True

    def _reset_engine(self) -> None:
        self._engine = PhysicsEngine(
            self.sim_width, self.sim_height, self.field,
            sor_omega=self.sor_omega)
        self.potential = self._engine.potential
        self.grad_x = self._engine.grad_x
        self.grad_y = self._engine.grad_y
        self.E_magnitude = self._engine.E_magnitude
        self.total_iterations = 0
        self._isolines_dirty = True
        self._portal_render_dirty = True

    def _render_field(self) -> None:
        """Renders the field into a preallocated Surface with no allocations"""
        data = self.potential if self.view_mode == "potential" else self.E_magnitude
        rgb = self.color_mapper(data)  # (H, W, 3) uint8
        ps = self._px_int
        rgbT = np.ascontiguousarray(rgb.transpose(1, 0, 2))  # (W, H, 3)
        up = np.repeat(np.repeat(rgbT, ps, axis=1), ps, axis=0)  # (W_px, H_px, 3)
        up = up[:self._sim_w_px, :self._sim_h_px]
        pxa = pygame.surfarray.pixels3d(self._field_surf)
        pxa[:] = up
        del pxa
        self.sim_surface.blit(self._field_surf, (0, 0))

    def _render_isolines(self) -> None:
        if not self.show_isolines:
            return
        data = self.potential if self.view_mode == "potential" else self.E_magnitude    
        if self._isolines_dirty or self._isolines_cache is None:
            self._isolines_cache = self._compute_isolines(data)
            self._isolines_dirty = False

        ps = max(1, int(self.px_scale))
        sim_w_px = int(self.sim_width  * self.px_scale)
        sim_h_px = int(self.sim_height * self.px_scale)
        up = np.repeat(np.repeat(self._isolines_cache, ps, 0), ps, 1)
        up = up[:sim_h_px, :sim_w_px]

        ov = self._overlay_surf
        ov.fill((0, 0, 0, 0))
        p3 = pygame.surfarray.pixels3d(ov)
        pa = pygame.surfarray.pixels_alpha(ov)
        ut = up.T
        p3[ut] = (255, 255, 255)
        pa[ut] = 80
        del p3, pa
        self.sim_surface.blit(ov, (0, 0))

    def _compute_isolines(self, data: np.ndarray) -> np.ndarray:
        d_min, d_max = float(np.min(data)), float(np.max(data))
        if d_max - d_min < 1e-9:
            return np.zeros(data.shape, dtype=bool)
        levels = np.linspace(d_min, d_max, self.isoline_count + 2)[1:-1]
        mask   = np.zeros(data.shape, dtype=bool)
        for level in levels:
            above = data > level
            ch = above[:-1, :] ^ above[1:, :]
            mask[:-1, :] |= ch;  mask[1:, :] |= ch
            cv = above[:, :-1] ^ above[:, 1:]
            mask[:, :-1] |= cv;  mask[:, 1:] |= cv
        return mask

    def _render_portals(self) -> None:
        """Cached pixel-level render"""

        if self._portal_render_dirty or self._portal_render_cache is None:
            self._portal_render_cache = self._build_portal_render_cache()
            self._portal_render_dirty = False

        if not self._portal_render_cache:
            return

        ov = self._overlay_surf
        ov.fill((0, 0, 0, 0))
        p3 = pygame.surfarray.pixels3d(ov)
        pa = pygame.surfarray.pixels_alpha(ov)
        for ut, color in self._portal_render_cache:
            p3[ut] = color
            pa[ut] = 210
        del p3, pa
        self.sim_surface.blit(ov, (0, 0))

    def _build_portal_render_cache(self) -> list:
        """Computes upscaled masks for all objects. Called rarely."""
        ps = self._px_int
        sim_w_px = self._sim_w_px
        sim_h_px = self._sim_h_px
        result = []

        to_draw: list = []
        for obj in self.field:
            if isinstance(obj, CouplePortal):
                to_draw += [(obj.p1.mask, obj.p1.color),
                             (obj.p2.mask, obj.p2.color)]
            elif isinstance(obj, (FixedPotentialPortal, PotentialAnchor,
                                   MaterialObject, ConductorObject)):
                to_draw.append((obj.mask, obj.color))
            elif isinstance(obj, MultiPortal):
                to_draw += list((p.mask, p.color) for p in obj.args)

        for mask, color in to_draw:
            m = mask(self.X, self.Y)  # (H_sim, W_sim) bool
            if not np.any(m):
                continue
            up = np.repeat(np.repeat(m, ps, axis=0), ps, axis=1)
            up = up[:sim_h_px, :sim_w_px]
            result.append((up.T.copy(), color))  # T: column-major for surfarray

        return result

    def _render_portal_arrows(self) -> None:
        """Draws an arrow at each teleport portal's mouth pointing toward
        its working (front) side"""
        ps = self.px_scale
        arrow_len = 6.0   # grid units
        head_len = arrow_len * ps * 0.35
        lw = max(1, int(ps * 0.4))

        portals: list = []
        for obj in self.field:
            if isinstance(obj, CouplePortal):
                portals += [obj.p1, obj.p2]
            elif isinstance(obj, MultiPortal):
                portals += list(obj.args)

        for p in portals:
            if not p.active:
                continue
            size = p.mask.size()
            if size is None:
                continue
            cx, cy = p.mask.center
            w, h = size
            axis = p.normal_axis or ("y" if w >= h else "x")
            if axis == "y":
                dx, dy = 0.0, (1.0 if p.facing_positive else -1.0)
            else:
                dx, dy = (1.0 if p.facing_positive else -1.0), 0.0

            sx, sy = cx * ps, cy * ps
            ex, ey = sx + dx * arrow_len * ps, sy + dy * arrow_len * ps

            pygame.draw.line(self.sim_surface, p.color, (sx, sy), (ex, ey), lw)
            angle = np.arctan2(dy, dx)
            for wa in (angle + 2.618, angle - 2.618):
                pygame.draw.line(self.sim_surface, p.color, (ex, ey),
                                 (ex + np.cos(wa) * head_len,
                                  ey + np.sin(wa) * head_len), lw)

    def _render_vectors(self) -> None:
        """Vector field"""
        if not self.show_vectors:
            return

        step = 10
        max_len = step * self.px_scale * 2.0
        head_len = max_len * 0.12
        lw  = max(1, int(head_len * 0.2))
        color = (60, 60, 70)
        ps = self.px_scale

        grad_mag = np.sqrt(self.grad_x ** 2 + self.grad_y ** 2)
        max_mag  = float(np.max(grad_mag))
        if max_mag < 1e-12:
            return
        scale = max_len / max_mag

        rows = np.arange(0, self.sim_height, step)
        cols = np.arange(0, self.sim_width,  step)
        R, C = np.meshgrid(rows, cols, indexing='ij')  # (Nr, Nc)

        gx = -self.grad_x[R, C]  # (Nr, Nc)
        gy = -self.grad_y[R, C]
        mag = grad_mag[R, C]

        valid = mag > 1e-12
        if not np.any(valid):
            return

        sx = (C * ps + ps / 2).astype(float)
        sy = (R * ps + ps / 2).astype(float)

        ex = sx + gx * scale
        ey = sy + gy * scale

        angles = np.arctan2(gy, gx)  # (Nr, Nc)

        vi = np.argwhere(valid)  # (N_valid, 2)
        for r, c in vi:
            pygame.draw.line(self.sim_surface, color,
                             (sx[r, c], sy[r, c]), (ex[r, c], ey[r, c]), lw)
            a = angles[r, c]
            for wa in (a + 2.618, a - 2.618):
                pygame.draw.line(self.sim_surface, color,
                                 (ex[r, c], ey[r, c]),
                                 (ex[r, c] + np.cos(wa) * head_len,
                                  ey[r, c] + np.sin(wa) * head_len), lw)
                
    def _render_test_charges(self) -> None:
        ps = self.px_scale
        for q in self.test_charges:
            if not q.active:
                continue
            sx, sy = int(q.x * ps), int(q.y * ps)
            if len(q.trail) > 1:
                pts = [(int(px * ps), int(py * ps)) for px, py in q.trail]
                pygame.draw.lines(self.sim_surface, q.color, False, pts, 1)
            radius = max(4, int(ps * 1.5))
            pygame.draw.circle(self.sim_surface, q.color, (sx, sy), radius)


    def _portals_mask(self) -> np.ndarray:
        """Boolean grid of every teleport portal's footprint (CouplePortal /
        MultiPortal), dilated by 1px, used to exclude portal pixels from
        flux boundaries.

        A MaterialObject mid-teleport is split into a remainder (touching
        the source portal) and a shifted piece (touching the destination
        portal) - see Simulation._teleport_material_objects. The cut where
        the object was severed is an artifact, not real surface, and must
        be excluded from compute_flux's boundary. The source-side cut
        lands exactly on the portal's own mask cell, but the
        destination-side cut lands 1px short of it (back_region_mask
        excludes the portal's own footprint asymmetrically - see
        Portal.back_region_mask), so the raw footprint alone only cancels
        one of the two cuts. Dilating by 1px covers both.
        """
        mask = np.zeros((self.sim_height, self.sim_width), dtype=bool)
        for obj in self.field:
            if isinstance(obj, CouplePortal):
                mask |= obj.p1.get_mask(self.X, self.Y)
                mask |= obj.p2.get_mask(self.X, self.Y)
            elif isinstance(obj, MultiPortal):
                for p in obj.args:
                    mask |= p.get_mask(self.X, self.Y)

        dilated = mask.copy()
        dilated[1:, :]  |= mask[:-1, :]
        dilated[:-1, :] |= mask[1:, :]
        dilated[:, 1:]  |= mask[:, :-1]
        dilated[:, :-1] |= mask[:, 1:]
        return dilated

    def _render_material_flux(self) -> None:
        """Draws the E-field flux through each MaterialObject as text,
        centered above the object."""
        ps = self.px_scale
        font = self._fonts["small"]
        portals_mask = self._portals_mask()

        for obj in self.field:
            if not isinstance(obj, MaterialObject) or not obj.active:
                continue
            if not obj.show_flux:
                continue
            mask = obj.get_mask(self.X, self.Y)
            if not np.any(mask):
                continue

            flux = obj.compute_flux(self.X, self.Y, self.grad_x, self.grad_y,
                                     portals_mask)
            cx, cy = obj.mask.center
            sx, sy = cx * ps, cy * ps

            txt = font.render(f"\u03a6 = {flux:+.3f}", True, (255, 255, 255))
            bg = pygame.Surface((txt.get_width() + 6, txt.get_height() + 4))
            bg.set_alpha(150)
            bg.fill((0, 0, 0))

            tx = sx - txt.get_width() / 2
            ty = sy - (obj.mask.size()[1] if obj.mask.size() else 0) * ps / 2 - 18

            self.sim_surface.blit(bg, (tx - 3, ty - 2))
            self.sim_surface.blit(txt, (tx, ty))

    # ------------------------------------------------------------------
    # Data export (snapshot / recording)
    # ------------------------------------------------------------------

    def _output_dir(self) -> Path:
        if self._output_dir_cache is None:
            out = Path(__file__).resolve().parent.parent / "output"
            out.mkdir(parents=True, exist_ok=True)
            self._output_dir_cache = out
        return self._output_dir_cache

    def _is_pinned_like(self, obj) -> bool:
        if isinstance(obj, (MaterialObject, ConductorObject)):
            return obj.pinned
        return isinstance(obj, (Portal, PotentialAnchor))

    def _iter_pinned_candidates(self):
        """Flattens CouplePortal/MultiPortal into their child portals."""
        for obj in self.field:
            if isinstance(obj, CouplePortal):
                yield obj.p1
                yield obj.p2
            elif isinstance(obj, MultiPortal):
                yield from obj.args
            else:
                yield obj

    def _collect_pinned_objects_data(self) -> Tuple[List[dict], Dict[str, np.ndarray]]:
        meta: List[dict] = []
        arrays: Dict[str, np.ndarray] = {}
        for obj in self._iter_pinned_candidates():
            if not self._is_pinned_like(obj):
                continue
            i = len(meta)
            key = f"pinned_mask_{i}"
            meta.append({
                "index": i,
                "array_key": key,
                "type": type(obj).__name__,
                "label": getattr(obj, "label", None),
                "color": list(getattr(obj, "color", (0, 0, 0))),
                "potential_value": getattr(obj, "potential_value", None),
                "active": getattr(obj, "active", True),
                "facing_positive": getattr(obj, "facing_positive", None),
                "normal_axis": getattr(obj, "normal_axis", None),
            })
            arrays[key] = obj.get_mask(self.X, self.Y)
        return meta, arrays

    def _collect_field_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "potential": self.potential.copy(),
            "grad_x": self.grad_x.copy(),
            "grad_y": self.grad_y.copy(),
            "E_magnitude": self.E_magnitude.copy(),
        }

    def _sim_params_dict(self) -> dict:
        return {
            "sim_width": self.sim_width,
            "sim_height": self.sim_height,
            "px_scale": self.px_scale,
            "solver_mode": self.solver_mode,
            "mom_images": self.mom_images,
            "sor_omega": self.sor_omega,
            "iterations_per_frame": self.iterations_per_frame,
            "diff_threshold": self.diff_threshold,
            "view_mode": self.view_mode,
            "fps": self.fps,
            "total_iterations": self.total_iterations,
            "diff": self.diff,
        }

    def _save_snapshot(self) -> None:
        try:
            meta, pinned_arrays = self._collect_pinned_objects_data()
            field_arrays = self._collect_field_arrays()

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = self._output_dir()
            npz_path = out / f"snapshot_{ts}.npz"
            json_path = out / f"snapshot_{ts}.json"

            np.savez_compressed(npz_path, **pinned_arrays, **field_arrays)
            with open(json_path, "w") as f:
                json.dump({
                    "kind": "snapshot",
                    "timestamp": ts,
                    "sim_params": self._sim_params_dict(),
                    "pinned_objects": meta,
                    "field_arrays": ["potential", "grad_x", "grad_y", "E_magnitude"],
                }, f, indent=2)

            self._last_snapshot_path = npz_path
            print(f"[export] snapshot saved to {npz_path}")
        except Exception as e:
            print(f"[export] snapshot failed: {e}")

    def _toggle_recording(self) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        self._recording = True
        self._recording_start_ts = datetime.now()

        self._rec_pinned_meta, self._rec_pinned_arrays = self._collect_pinned_objects_data()
        self._rec_field_arrays = self._collect_field_arrays()

        self._rec_material_objs = [
            obj for obj in self.field
            if isinstance(obj, MaterialObject) and obj.active
        ]
        self._rec_test_charges = list(self.test_charges)

        self._rec_material_frames: Dict[int, list] = {
            i: [] for i in range(len(self._rec_material_objs))
        }
        self._rec_charge_frames: Dict[int, list] = {
            i: [] for i in range(len(self._rec_test_charges))
        }
        self._rec_frame_count = 0

    def _record_frame(self) -> None:
        if not self._recording:
            return
        for i, obj in enumerate(self._rec_material_objs):
            self._rec_material_frames[i].append(obj.get_mask(self.X, self.Y).copy())
        for i, q in enumerate(self._rec_test_charges):
            self._rec_charge_frames[i].append((q.x, q.y))
        self._rec_frame_count += 1

    def _stop_recording(self) -> None:
        self._recording = False

        if self._rec_frame_count == 0:
            print("[export] recording stopped with 0 frames, nothing saved")
            self._reset_recording_state()
            return

        try:
            mat_arrays = {
                f"material_mask_{i}": np.stack(frames, axis=0)
                for i, frames in self._rec_material_frames.items() if frames
            }
            charge_arrays = {
                f"charge_pos_{i}": np.asarray(frames, dtype=np.float64)
                for i, frames in self._rec_charge_frames.items() if frames
            }

            ts = self._recording_start_ts.strftime("%Y%m%d_%H%M%S")
            out = self._output_dir()
            npz_path = out / f"recording_{ts}.npz"
            json_path = out / f"recording_{ts}.json"

            np.savez_compressed(
                npz_path,
                **self._rec_pinned_arrays, **self._rec_field_arrays,
                **mat_arrays, **charge_arrays,
            )
            with open(json_path, "w") as f:
                json.dump({
                    "kind": "recording",
                    "timestamp": ts,
                    "frame_count": self._rec_frame_count,
                    "sim_params": self._sim_params_dict(),
                    "pinned_objects": self._rec_pinned_meta,
                    "material_objects": [
                        {
                            "index": i, "array_key": f"material_mask_{i}",
                            "label": obj.label, "color": list(obj.color),
                            "charge": obj.charge, "mass": obj.mass,
                        }
                        for i, obj in enumerate(self._rec_material_objs)
                    ],
                    "test_charges": [
                        {
                            "index": i, "array_key": f"charge_pos_{i}",
                            "charge": q.charge, "mass": q.mass,
                            "color": list(q.color),
                        }
                        for i, q in enumerate(self._rec_test_charges)
                    ],
                }, f, indent=2)

            print(f"[export] recording saved to {npz_path} "
                  f"({self._rec_frame_count} frames)")
        except Exception as e:
            print(f"[export] recording failed: {e}")
        finally:
            self._reset_recording_state()

    def _reset_recording_state(self) -> None:
        self._recording_start_ts = None
        self._rec_pinned_meta = []
        self._rec_pinned_arrays = {}
        self._rec_field_arrays = {}
        self._rec_material_objs = []
        self._rec_test_charges = []
        self._rec_material_frames = {}
        self._rec_charge_frames = {}
        self._rec_frame_count = 0

    def draw(self) -> None:
        self._render_field()
        self._render_isolines()
        self._render_portals()
        self._render_portal_arrows()
        self._render_vectors()
        self._render_material_flux()
        self._render_test_charges()
        self.screen.blit(self.sim_surface, (0, 0))
        self._panel.draw(self.screen, self._fonts)
        pygame.display.flip()

    def handle_input(self) -> bool:
        ps = self.px_scale
        sim_w_px = int(self.sim_width * ps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.view_mode = ("E_magnitude" if self.view_mode == "potential"
                                      else "potential")
                elif event.key == pygame.K_v:
                    self.show_vectors = not self.show_vectors
                elif event.key == pygame.K_i:
                    self.show_isolines = not self.show_isolines

            mouse_in_sim = (hasattr(event, "pos") and
                            event.pos[0] < sim_w_px)

            if not mouse_in_sim:
                self._panel.handle_event(event)
                continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos[0] / ps, event.pos[1] / ps
                self._dragging_mask, self._dragging_obj = \
                    self._find_draggable(mx, my)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                mx, my = event.pos[0] / ps, event.pos[1] / ps
                obj = self._find_obj_at(mx, my)
                if obj is not None:
                    self._panel._active = 1
                    self._open_inspector(obj)

            elif event.type == pygame.MOUSEBUTTONUP:
                self._dragging_mask = None
                self._dragging_obj  = None

            elif (event.type == pygame.MOUSEMOTION
              and self._dragging_mask is not None):
                dx = event.rel[0] / ps
                dy = event.rel[1] / ps
                if isinstance(self._dragging_obj, TestCharge):
                    self._dragging_obj.x = max(0.0, min(self.sim_width  - 1.0,
                                                        self._dragging_obj.x + dx))
                    self._dragging_obj.y = max(0.0, min(self.sim_height - 1.0,
                                                        self._dragging_obj.y + dy))
                    self._dragging_obj.vx = 0.0
                    self._dragging_obj.vy = 0.0
                    self._dragging_obj._initialized_accel = False
                    self._dragging_obj.reset_trail()
                else:
                    self._dragging_mask.translate(dx, dy)
                    self._invalidate_caches()
                    self._isolines_dirty = True
                    self._panel.invalidate_tab("INSPECTOR")

        return True

    def _find_draggable(self, mx, my) -> Tuple[Optional[Mask], object]:
        pt = (mx, my)
        for obj in self.field:
            if isinstance(obj, CouplePortal):
                if pt in obj.p1:
                    return obj.p1.mask, obj.p1
                if pt in obj.p2:
                    return obj.p2.mask, obj.p2
            elif isinstance(obj, MaterialObject):
                if pt in obj:
                    return obj.mask, obj
            elif isinstance(obj, (FixedPotentialPortal, PotentialAnchor)):
                if pt in obj:
                    return obj.mask, obj
            elif isinstance(obj, MultiPortal):
                for p in obj.args:
                    if pt in p:
                        return p.mask, p
        for q in self.test_charges:
            if q.active and abs(mx - q.x) < 3 and abs(my - q.y) < 3:
                return q, q
        return None, None

    def _find_obj_at(self, mx: float, my: float):
        """Returns the object under the cursor"""
        pt = (mx, my)
        for obj in self.field:
            if isinstance(obj, CouplePortal):
                if pt in obj.p1.mask:
                    return obj.p1
                if pt in obj.p2.mask:
                    return obj.p2
            elif isinstance(obj, (MaterialObject, FixedPotentialPortal,
                                   PotentialAnchor)):
                if obj.active and pt in obj.mask:
                    return obj
            elif isinstance(obj, MultiPortal):
                for p in obj.args:
                    if pt in p:
                        return p
        for q in self.test_charges:
            if q.active and abs(mx - q.x) < 3 and abs(my - q.y) < 3:
                return q
        return None

    def _build_panel(self) -> TabbedPanel:
        sim_w_px = int(self.sim_width * self.px_scale)
        sim_h_px = int(self.sim_height * self.px_scale)
        panel = TabbedPanel(sim_w_px, 0, UI_WIDTH, sim_h_px, ["SIMULATION", "INSPECTOR"])

        T = "SIMULATION"

        # region Interface objects
        panel.add(T, SectionHeader(0, 0, 0, "Portals Engine"))

        panel.add(T, SectionHeader(0, 0, 0, "Info"))
        panel.add(T, Label(0, 0, 0, "FPS",
                           value_fn=lambda: f"{self.clock.get_fps():.0f}"))
        panel.add(T, Label(0, 0, 0, "Diff",
                           value_fn=lambda: f"{self.diff:.2e}"))
        panel.add(T, Label(0, 0, 0, "Iterations",
                           value_fn=lambda: str(self.total_iterations)))
        panel.add(T, Label(0, 0, 0, "Grid",
                           value_fn=lambda: f"{self.sim_width}×{self.sim_height}"))
        panel.add(T, Divider(0, 0, 0))

        panel.add(T, SectionHeader(0, 0, 0, "View Mode"))
        panel.add(T, Button(0, 0, 0, 26, "Electric field |E|",
                            callback=lambda: setattr(self, "view_mode", "E_magnitude"),
                            active_fn=lambda: self.view_mode == "E_magnitude"))
        panel.add(T, Button(0, 0, 0, 26, "Potential",
                            callback=lambda: setattr(self, "view_mode", "potential"),
                            active_fn=lambda: self.view_mode == "potential"))
        panel.add(T, Divider(0, 0, 0))

        panel.add(T, SectionHeader(0, 0, 0, "Display"))
        panel.add(T, Toggle(0, 0, 0, "Vectors",
                            getter=lambda: self.show_vectors,
                            setter=lambda v: setattr(self, "show_vectors", v)))
        panel.add(T, Toggle(0, 0, 0, "Isolines",
                            getter=lambda: self.show_isolines,
                            setter=lambda v: setattr(self, "show_isolines", v)))
        panel.add(T, Stepper(0, 0, 0, "Isolines count",
                             getter=lambda: self.isoline_count,
                             setter=lambda v: self._set_isoline_count(int(v)),
                             step=1, fmt="{:.0f}", min_val=2, max_val=30))
        panel.add(T, Divider(0, 0, 0))

        panel.add(T, SectionHeader(0, 0, 0, "Color Scheme"))
        panel.add(T, Label(0, 0, 0, "Scheme",
                           value_fn=lambda: self._color_scheme_names[
                               self._color_scheme_idx]))
        panel.add(T, Button(0, 0, 0, 26, "Previous",
                            callback=lambda: self._cycle_color(-1)))
        panel.add(T, Button(0, 0, 0, 26, "Next",
                            callback=lambda: self._cycle_color(+1)))
        panel.add(T, Divider(0, 0, 0))

        panel.add(T, SectionHeader(0, 0, 0, "Physics"))
        panel.add(T, Stepper(0, 0, 0, "Iter/frame",
                             getter=lambda: self.iterations_per_frame,
                             setter=lambda v: setattr(self, "iterations_per_frame",
                                                       int(v)),
                             step=5, fmt="{:.0f}", min_val=1, max_val=500))
        panel.add(T, Stepper(0, 0, 0, "SOR ω",
                             getter=lambda: self.sor_omega,
                             setter=lambda v: setattr(self, "sor_omega",
                                                       round(v, 2)),
                             step=0.05, fmt="{:.2f}", min_val=1.0, max_val=1.99))
        panel.add(T, Divider(0, 0, 0))

        if self.solver_mode == "mom":
            panel.add(T, SectionHeader(0, 0, 0, "MOM"))
            panel.add(T, Label(0, 0, 0, "Estado",
                               value_fn=lambda: self._mom_status))
            panel.add(T, Button(0, 0, 0, 28, "Recalcular MOM",
                                callback=self._recompute_mom))
            panel.add(T, Divider(0, 0, 0))

        panel.add(T, SectionHeader(0, 0, 0, "Export"))
        panel.add(T, Button(0, 0, 0, 26, "Save Snapshot",
                             callback=self._save_snapshot))
        panel.add(T, Button(0, 0, 0, 26, "Start Recording",
                             callback=self._toggle_recording,
                             active_fn=lambda: self._recording))
        panel.add(T, Label(0, 0, 0, "Frames",
                            value_fn=lambda: str(self._rec_frame_count)
                            if self._recording else "-"))
        panel.add(T, Divider(0, 0, 0))

        panel.add(T, SectionHeader(0, 0, 0, "Hotkeys"))

        for line in ["M - change view mode", "V - show vectors",
                     "I - show isolines", "", ":3"]:
            lbl = Label(0, 0, 0, line, font_key="small")
            lbl.rect.h = 16
            panel.add(T, lbl)

        panel.set_tab_factory("INSPECTOR", self._build_scene_widgets)
        # endregion

        return panel

    def _build_scene_widgets(self) -> List:
        w: List = []

        w.append(SectionHeader(0, 0, 0, "Objects"))

        if not self.field:
            lbl = Label(0, 0, 0, "(empty scene)", font_key="small")
            lbl.rect.h = 16
            w.append(lbl)

        for i, obj in enumerate(self.field):
            if isinstance(obj, CouplePortal):
                w.append(SectionHeader(0, 0, 0, "Portal Pair"))
                w.append(Button(0, 0, 0, 24,
                                f"  Portal 1 ({_mask_type(obj.p1.mask)})",
                                callback=lambda o=obj.p1: self._open_inspector(o)))
                w.append(Button(0, 0, 0, 24,
                                f"  Portal 2 ({_mask_type(obj.p2.mask)})",
                                callback=lambda o=obj.p2: self._open_inspector(o)))
            elif isinstance(obj, FixedPotentialPortal):
                w.append(Button(0, 0, 0, 24,
                                f"Fixed  φ={obj.potential_value:.2f}"
                                f"  ({_mask_type(obj.mask)})",
                                callback=lambda o=obj: self._open_inspector(o)))
            elif isinstance(obj, PotentialAnchor):
                w.append(Button(0, 0, 0, 24,
                                f"Anchor φ={obj.potential_value:.2f}"
                                f"  ({_mask_type(obj.mask)})",
                                callback=lambda o=obj: self._open_inspector(o)))
            elif isinstance(obj, MaterialObject):
                pin = "pinned" if obj.pinned else "dynamic"
                w.append(Button(0, 0, 0, 24,
                                f"{pin} {obj.label}  ({_mask_type(obj.mask)})",
                                callback=lambda o=obj: self._open_inspector(o)))
        for i, q in enumerate(self.test_charges):
            w.append(Button(0, 0, 0, 24,
                            f"TestCharge {i+1}  q={q.charge:.3f}",
                            callback=lambda o=q: self._open_test_charge_inspector(o)))

        w.append(Divider(0, 0, 0))
        w.append(SectionHeader(0, 0, 0, "Add"))
        w.append(Button(0, 0, 0, 24, "+ Portal Pair",
                        callback=self._add_couple_portal))
        w.append(Button(0, 0, 0, 24, "+ Fixed Potential",
                        callback=self._add_fixed_potential))
        w.append(Button(0, 0, 0, 24, "+ Anchor",
                        callback=self._add_anchor))
        w.append(Button(0, 0, 0, 24, "+ Cube",
                        callback=lambda: self._add_material("Cube",
                            RectangleMask, (180, 180, 180))))
        w.append(Button(0, 0, 0, 24, "+ Planet",
                        callback=lambda: self._add_material("Planet",
                            CircleMask, (140, 90, 60))))
        w.append(Button(0, 0, 0, 24, "+ Test Charge",
                        callback=self._add_test_charge))
        w.append(Divider(0, 0, 0))

        w.append(SectionHeader(0, 0, 0, "Presets"))
        for name, fn in self._presets().items():
            w.append(Button(0, 0, 0, 24, name, callback=fn))

        return w

    def _cx(self): return self.sim_width // 2
    def _cy(self): return self.sim_height // 2

    def _add_couple_portal(self) -> None:
        cx, cy = self._cx(), self._cy()
        p1 = Portal(RectangleMask(cx-20, cx+20, cy-15, cy-13), (255, 153, 0))
        p2 = Portal(RectangleMask(cx-20, cx+20, cy+13, cy+15), (0, 204, 255))
        self.field.append(CouplePortal(p1, p2))
        self._refresh_field()

    def _add_fixed_potential(self) -> None:
        cx, cy = self._cx(), self._cy()
        self.field.append(
            FixedPotentialPortal(RectangleMask(cx-10, cx+10, cy-3, cy+3), 0.7))
        self._refresh_field()

    def _add_anchor(self) -> None:
        cx, cy = self._cx(), self._cy()
        self.field.append(PotentialAnchor(CircleMask(cx, cy, 4), 0.5))
        self._refresh_field()

    def _add_material(self, label: str, mask_cls, color: tuple) -> None:
        cx, cy = self._cx(), self._cy()
        if mask_cls is RectangleMask:
            mask = RectangleMask(cx-10, cx+10, cy-10, cy+10)
        else:
            mask = CircleMask(cx, cy, 10)
        self.field.append(MaterialObject(mask, color=color, label=label))
        self._refresh_field()
    
    def _add_test_charge(self) -> None:
        cx, cy = self._cx(), self._cy()
        q = TestCharge(x=float(cx), y=float(cy),
                    vx=0.0, vy=0.0,
                    charge=0.1, mass=1.0,
                    color=(255, 220, 0), trail_len=300)
        self.test_charges.append(q)
        self._panel.invalidate_tab("INSPECTOR")   # refresca la lista, NO toca field ni engine

    def _open_test_charge_inspector(self, q: TestCharge) -> None:
        self._panel.show_inspector(self._build_test_charge_inspector(q))

    def _build_test_charge_inspector(self, q: TestCharge) -> List:
        w: List = []
        w.append(Button(0, 0, 0, 26, "<- Back",
                        callback=self._close_inspector))
        w.append(SectionHeader(0, 0, 0, "Test Charge"))
        w.append(Button(0, 0, 0, 24, "Remove",
                        callback=lambda: (self.test_charges.remove(q),
                                        self._close_inspector())))
        w.append(Divider(0, 0, 0))
        w.append(SectionHeader(0, 0, 0, "Parameters"))
        w.append(Stepper(0, 0, 0, "Charge q",
                        getter=lambda: q.charge,
                        setter=lambda v: setattr(q, "charge",
                                                round(v, 4)),
                        step=0.05, fmt="{:.2f}",
                        min_val=-5.0, max_val=5.0))
        w.append(Stepper(0, 0, 0, "Mass m",
                        getter=lambda: q.mass,
                        setter=lambda v: setattr(q, "mass", max(0.01, v)),
                        step=0.1, fmt="{:.2f}",
                        min_val=0.01, max_val=100.0))
        w.append(Stepper(0, 0, 0, "Trail length",
                        getter=lambda: q.trail_len,
                        setter=lambda v: (setattr(q, "trail_len", int(v)),
                                        q.reset_trail()),
                        step=50, fmt="{:.0f}",
                        min_val=0, max_val=2000))
        w.append(Divider(0, 0, 0))
        w.append(SectionHeader(0, 0, 0, "State"))
        w.append(Label(0, 0, 0, "x",
                    value_fn=lambda: f"{q.x:.1f}"))
        w.append(Label(0, 0, 0, "y",
                    value_fn=lambda: f"{q.y:.1f}"))
        w.append(Label(0, 0, 0, "|v|",
                    value_fn=lambda: f"{(q.vx**2+q.vy**2)**0.5:.3f}"))
        w.append(Label(0, 0, 0, "KE",
                    value_fn=lambda: f"{q.kinetic_energy():.4e}"))
        w.append(Button(0, 0, 0, 24, "Stop (reset velocity)",
                        callback=lambda: (setattr(q, "vx", 0.0),
                                        setattr(q, "vy", 0.0),
                                        setattr(q, "_initialized_accel", False))))
        w.append(Button(0, 0, 0, 24, "Clear trail",
                        callback=q.reset_trail))
        return w

    def _refresh_field(self) -> None:
        self._bake_movable_masks()
        self._invalidate_caches()
        self._panel.invalidate_tab("INSPECTOR")

    # region Presets
    def _presets(self) -> dict:
        return {
            "Couple Portals": self._preset_couple_portals,
            "Couple Circles":  self._preset_couple_circles,
            "Advanced":        self._preset_advanced,
            "Dynamic Ball":    self._preset_dynamic,
            "Clear Scene":     self._preset_clear,
        }

    def _load_preset(self, field_objs: list) -> None:
        self.field = field_objs
        self._bake_movable_masks()
        self._reset_engine()
        self._panel.invalidate_tab("INSPECTOR")
        self._panel.close_inspector()

    def _preset_couple_portals(self) -> None:
        # p1 sits near the top wall - front faces down (into the domain);
        # p2 sits near the bottom wall - front faces up
        p1 = Portal(RectangleMask(25, 75, 25, 26), (255, 153, 0),
                    facing_positive=True)
        p2 = Portal(RectangleMask(25, 75, 74, 75), (0, 204, 255),
                    facing_positive=False)
        a_hi = PotentialAnchor(RectangleMask(0, self.sim_width, 0, 1),   1.0)
        a_lo = PotentialAnchor(RectangleMask(0, self.sim_width,
                                             self.sim_height-1,
                                             self.sim_height), 0.0)
        self._load_preset([a_hi, a_lo, CouplePortal(p1, p2)])

    def _preset_couple_circles(self) -> None:
        # Circles have a square bounding box, so the front/back axis can't
        # be inferred from shape - set it explicitly. p1 sits near the left
        # wall (front faces right); p2 near the right wall (front faces left)
        p1 = Portal(CircleMask(35, self.sim_height//2, 10), (255, 100, 50),
                    facing_positive=True, normal_axis="x")
        p2 = Portal(CircleMask(self.sim_width-35, self.sim_height//2, 10),
                    (50, 200, 255), facing_positive=False, normal_axis="x")
        a_hi = PotentialAnchor(RectangleMask(0, self.sim_width, 0, 1),   1.0)
        a_lo = PotentialAnchor(RectangleMask(0, self.sim_width,
                                             self.sim_height-1,
                                             self.sim_height), 0.0)
        self._load_preset([a_hi, a_lo, CouplePortal(p1, p2)])

    def _preset_advanced(self) -> None:
        W, H = self.sim_width, self.sim_height
        top = FixedPotentialPortal(
            RectangleMask(20, W-20, H//3, H//3 + 3), 0.8, (255, 80, 80))
        bot = FixedPotentialPortal(
            RectangleMask(20, W-20, 2*H//3 - 3, 2*H//3), 0.2, (80, 120, 255))
        obs = MaterialObject(CircleMask(W//2, H//2, 8),
                             color=(80, 220, 120), pinned=True, label="Conductor")
        self._load_preset([top, bot, obs])

    def _preset_dynamic(self) -> None:
        W, H = self.sim_width, self.sim_height
        a_hi = PotentialAnchor(RectangleMask(0, W, 0, 1),   1.0)
        a_lo = PotentialAnchor(RectangleMask(0, W, H-1, H), 0.0)
        ball = MaterialObject(CircleMask(W//2, H//4, 7),
                              color=(220, 160, 60), pinned=False,
                              label="Ball", mass=1.0)
        self._load_preset([a_hi, a_lo, ball])

    def _preset_clear(self) -> None:
        self._load_preset([])
    # endregion

    def _open_inspector(self, obj) -> None:
        self._panel.show_inspector(self._build_inspector(obj))

    def _close_inspector(self) -> None:
        self._panel.close_inspector()
        self._panel.invalidate_tab("INSPECTOR")

    def _build_inspector(self, obj) -> List:
        w: List = []
        w.append(Button(0, 0, 0, 26, "<- Back",
                        callback=self._close_inspector))
        w.append(SectionHeader(0, 0, 0, type(obj).__name__))

        # Remove the object
        w.append(Button(0, 0, 0, 24, "Remove from scene",
                        callback=lambda o=obj: self._remove_obj(o)))
        
        #TestCharge
        if isinstance(obj, TestCharge):
            return self._build_test_charge_inspector(obj)

        # Potential
        if isinstance(obj, (FixedPotentialPortal, PotentialAnchor)):
            w.append(SectionHeader(0, 0, 0, "Potential"))
            w.append(Slider(0, 0, 0, "φ value",
                            getter=lambda o=obj: o.potential_value,
                            setter=lambda v, o=obj: (
                                setattr(o, "potential_value", round(v, 3)),
                                self._invalidate_caches()),
                            min_val=0.0, max_val=1.0, fmt="{:.3f}"))

        # MaterialObject
        if isinstance(obj, MaterialObject):
            w.append(SectionHeader(0, 0, 0, "Object"))
            w.append(Toggle(0, 0, 0, "Pinned",
                            getter=lambda o=obj: o.pinned,
                            setter=lambda v, o=obj: setattr(o, "pinned", v)))
            w.append(Stepper(0, 0, 0, "Charge q",
                             getter=lambda o=obj: o.charge,
                             setter=lambda v, o=obj: setattr(o, "charge", round(v, 2)),
                             step=0.01, fmt="{:.2f}", min_val=-10.0, max_val=10.0))
            w.append(Stepper(0, 0, 0, "Mass",
                             getter=lambda o=obj: o.mass,
                             setter=lambda v, o=obj: setattr(o, "mass", max(0.1, v)),
                             step=0.5, fmt="{:.1f}", min_val=0.1, max_val=100.0))
            w.append(Label(0, 0, 0, "Speed",
                           value_fn=lambda o=obj: f"{(o.vx**2+o.vy**2)**0.5:.2f}"))
            w.append(Button(0, 0, 0, 24, "Stop (reset velocity)",
                            callback=lambda o=obj: (
                                setattr(o, "vx", 0.0),
                                setattr(o, "vy", 0.0))))
            w.append(Toggle(0, 0, 0, "Show flux (\u03a6)",
                            getter=lambda o=obj: o.show_flux,
                            setter=lambda v, o=obj: setattr(o, "show_flux", v)))

        # Mask (hidden for baked ArrayMask objects - there's no formula left
        # to expose numerically, only a pixel grid; reposition by dragging)
        if hasattr(obj, "mask") and not isinstance(obj.mask, ArrayMask):
            w.append(SectionHeader(0, 0, 0, "Mask type"))
            w.append(Button(0, 0, 0, 24, "Rectangle",
                            callback=lambda o=obj: self._change_mask(
                                o, "rect"),
                            active_fn=lambda o=obj: isinstance(
                                o.mask, RectangleMask)))
            w.append(Button(0, 0, 0, 24, "Circle",
                            callback=lambda o=obj: self._change_mask(
                                o, "circle"),
                            active_fn=lambda o=obj: isinstance(
                                o.mask, CircleMask)))
            w.append(Button(0, 0, 0, 24, "Line",
                            callback=lambda o=obj: self._change_mask(
                                o, "line"),
                            active_fn=lambda o=obj: isinstance(
                                o.mask, LineMask)))
            w.append(Button(0, 0, 0, 24, "Point",
                            callback=lambda o=obj: self._change_mask(
                                o, "point"),
                            active_fn=lambda o=obj: isinstance(
                                o.mask, PointMask)))
            w += self._mask_widgets(obj)

        # Color
        if hasattr(obj, "color"):
            w.append(SectionHeader(0, 0, 0, "Color"))
            w += self._color_widgets(obj)

        return w

    # Change mask type

    def _change_mask(self, obj, mask_type: str) -> None:
        """Replaces the object's mask with a new one"""
        old = obj.mask
        cx = cy = None
        if isinstance(old, RectangleMask):
            cx = (old.x_min + old.x_max) / 2
            cy = (old.y_min + old.y_max) / 2
        elif isinstance(old, CircleMask):
            cx, cy = old.cx, old.cy
        elif isinstance(old, PointMask):
            cx, cy = old.x, old.y
        elif isinstance(old, LineMask):
            cx = (old.x1 + old.x2) / 2
            cy = (old.y1 + old.y2) / 2
        if cx is None:
            cx, cy = self._cx(), self._cy()

        if mask_type == "rect":
            obj.mask = RectangleMask(cx-10, cx+10, cy-5, cy+5)
        elif mask_type == "circle":
            obj.mask = CircleMask(cx, cy, 8)
        elif mask_type == "point":
            obj.mask = PointMask(cx, cy)
        elif mask_type == "line":
            obj.mask = LineMask(cx-10, cy, cx+10, cy, 1.0)

        self._invalidate_caches()
        self._panel.show_inspector(self._build_inspector(obj))

    # Mask parameters
    def _mask_widgets(self, obj) -> List:
        mask = obj.mask
        w: List = []
        w.append(SectionHeader(0, 0, 0, "Mask params"))

        def mk(label, getter, setter, step=1.0, mn=-1e6, mx=1e6):
            def _set(v, s=setter):
                s(v)
                self._on_mask_changed()
            return Stepper(0, 0, 0, label, getter=getter, setter=_set,
                           step=step, fmt="{:.1f}", min_val=mn, max_val=mx)

        if isinstance(mask, RectangleMask):
            w.append(mk("X min", lambda m=mask: m.x_min,
                         lambda v, m=mask: setattr(m, "x_min", v),
                         mn=0, mx=self.sim_width))
            w.append(mk("X max", lambda m=mask: m.x_max,
                         lambda v, m=mask: setattr(m, "x_max", v),
                         mn=0, mx=self.sim_width))
            w.append(mk("Y min", lambda m=mask: m.y_min,
                         lambda v, m=mask: setattr(m, "y_min", v),
                         mn=0, mx=self.sim_height))
            w.append(mk("Y max", lambda m=mask: m.y_max,
                         lambda v, m=mask: setattr(m, "y_max", v),
                         mn=0, mx=self.sim_height))

        elif isinstance(mask, CircleMask):
            w.append(mk("CX", lambda m=mask: m.cx,
                         lambda v, m=mask: setattr(m, "cx", v),
                         mn=0, mx=self.sim_width))
            w.append(mk("CY", lambda m=mask: m.cy,
                         lambda v, m=mask: setattr(m, "cy", v),
                         mn=0, mx=self.sim_height))
            w.append(mk("Radius", lambda m=mask: m.radius,
                         lambda v, m=mask: setattr(m, "radius", max(1, v)),
                         mn=1, mx=min(self.sim_width, self.sim_height)//2))

        elif isinstance(mask, PointMask):
            w.append(mk("X", lambda m=mask: m.x,
                         lambda v, m=mask: setattr(m, "x", v),
                         mn=0, mx=self.sim_width))
            w.append(mk("Y", lambda m=mask: m.y,
                         lambda v, m=mask: setattr(m, "y", v),
                         mn=0, mx=self.sim_height))

        elif isinstance(mask, LineMask):
            w.append(mk("X1", lambda m=mask: m.x1,
                         lambda v, m=mask: (setattr(m, "x1", v), m._update_cache())))
            w.append(mk("Y1", lambda m=mask: m.y1,
                         lambda v, m=mask: (setattr(m, "y1", v), m._update_cache())))
            w.append(mk("X2", lambda m=mask: m.x2,
                         lambda v, m=mask: (setattr(m, "x2", v), m._update_cache())))
            w.append(mk("Y2", lambda m=mask: m.y2,
                         lambda v, m=mask: (setattr(m, "y2", v), m._update_cache())))
            w.append(mk("Thickness", lambda m=mask: m.thickness,
                         lambda v, m=mask: setattr(m, "thickness", max(0.5, v)),
                         mn=0.5, mx=20))
        return w

    def _on_mask_changed(self) -> None:
        self._invalidate_caches()
        self._isolines_dirty = True

    # Color
    def _color_widgets(self, obj) -> List:
        def set_ch(o, ch, v):
            c = list(o.color)
            c[ch] = int(max(0, min(255, v)))
            o.color = tuple(c)

        return [
            Stepper(0, 0, 0, "R", getter=lambda o=obj: o.color[0],
                    setter=lambda v, o=obj: set_ch(o, 0, v),
                    step=5, fmt="{:.0f}", min_val=0, max_val=255),
            Stepper(0, 0, 0, "G", getter=lambda o=obj: o.color[1],
                    setter=lambda v, o=obj: set_ch(o, 1, v),
                    step=5, fmt="{:.0f}", min_val=0, max_val=255),
            Stepper(0, 0, 0, "B", getter=lambda o=obj: o.color[2],
                    setter=lambda v, o=obj: set_ch(o, 2, v),
                    step=5, fmt="{:.0f}", min_val=0, max_val=255),
        ]

    def _remove_obj(self, obj) -> None:
        """Removes the object or the portal containing obj from self.field"""
        to_remove = None
        for f in self.field:
            if f is obj:
                to_remove = f
                break
            if isinstance(f, CouplePortal) and (f.p1 is obj or f.p2 is obj):
                to_remove = f
                break
        if to_remove is not None:
            self.field.remove(to_remove)
        self._invalidate_caches()
        self._panel.invalidate_tab("INSPECTOR")
        self._close_inspector()

    def _set_isoline_count(self, n: int) -> None:
        self.isoline_count   = n
        self._isolines_dirty = True

    def _cycle_color(self, d: int) -> None:
        self._color_scheme_idx = (
            (self._color_scheme_idx + d) % len(self._color_scheme_names))
        self.color_mapper = COLOR_SCHEMES[
            self._color_scheme_names[self._color_scheme_idx]]()

    def run(self) -> None:
        running = True
        try:
            while running:
                running = self.handle_input()
                self.update_physics()
                self.draw()
                self.clock.tick(self.fps)
        finally:
            pygame.quit()


def _mask_type(mask) -> str:
    return type(mask).__name__.replace("Mask", "")

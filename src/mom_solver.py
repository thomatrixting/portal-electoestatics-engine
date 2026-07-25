import numpy as np

class MOMSolver2D:
    def __init__(self, boundary_meshes, coupled_pairs=None,
                 image_walls=None, image_walls_y=None, n_images: int = 1):
        """
        boundary_meshes: lista de BoundaryMesh con phi fijo
        coupled_pairs: lista de tuplas (BoundaryMesh, BoundaryMesh)
                       donde ambas superficies comparten phi desconocido
        image_walls: (x_min, x_max) - if given, every segment's influence is
                     mirrored across both vertical walls using the method of
                     images (same-sign/Neumann images, since these walls are
                     meant to reproduce the SOR engine's zero-horizontal-flux
                     side boundary, see physics.py's `p[:, 0] = p[:, 1]`).
                     Without this, finite Dirichlet segments (e.g. anchors
                     spanning the full sim width) fringe/curve near the left
                     and right edges, since the free-space log kernel has no
                     notion of the simulation box's side walls.
        image_walls_y: (y_min, y_max) - same idea as image_walls but for the
                     top/bottom walls. Unlike the side walls, these mirror
                     with the *opposite* sign (Dirichlet images), since the
                     top/bottom are meant to reproduce a fixed-potential wall
                     rather than a zero-flux one. When both image_walls and
                     image_walls_y are set, corner images (one x-reflection
                     composed with one y-reflection) are also added, forming
                     a full rectangular box; their sign is the product of
                     the two individual signs (+1 * -1 = -1).
        n_images: reflection order per wall (truncates the infinite image
                  series - each step adds 4 more mirrored copies per segment).
                  Shared by both image_walls and image_walls_y.
        """
        self.image_walls = image_walls
        self.image_walls_y = image_walls_y
        self.n_images = n_images
        self.segments = []
        self.segment_group = []  # índice de grupo para acoplados

        # Segmentos con phi fijo
        for mesh in boundary_meshes:
            for seg in mesh.segments:
                self.segments.append(seg)
                self.segment_group.append(None)  # phi conocido

        # Segmentos acoplados: phi desconocido compartido entre N mallas
        self.coupled_pairs = coupled_pairs or []
        self.coupled_offsets = []  # dónde empieza cada grupo en self.segments
        for pair_idx, group in enumerate(self.coupled_pairs):
            offset = len(self.segments)
            for mesh in group:
                for seg in mesh.segments:
                    self.segments.append((seg[0], seg[1], seg[2], None))
                    self.segment_group.append(pair_idx)
            self.coupled_offsets.append(offset)

        self.N = len(self.segments)
        self.n_pairs = len(self.coupled_pairs)
        self.sigma = None
        self.phi_coupled = None  # potenciales resueltos de los pares

    def _image_xs(self, x0: float) -> list:
        """x-coordinates of x0's mirror images across both walls in
        self.image_walls, truncated to self.n_images reflections per side."""
        if self.image_walls is None:
            return []
        x_min, x_max = self.image_walls
        W = x_max - x_min
        if W <= 0:
            return []
        xs = []
        for k in range(-self.n_images, self.n_images + 1):
            if k != 0:
                xs.append(x0 + 2 * k * W)
            xs.append(2 * x_min - x0 + 2 * k * W)
        return xs

    def _image_ys(self, y0: float) -> list:
        """Same as _image_xs but for the top/bottom walls in
        self.image_walls_y (used with opposite/Dirichlet sign by callers)."""
        if self.image_walls_y is None:
            return []
        y_min, y_max = self.image_walls_y
        H = y_max - y_min
        if H <= 0:
            return []
        ys = []
        for k in range(-self.n_images, self.n_images + 1):
            if k != 0:
                ys.append(y0 + 2 * k * H)
            ys.append(2 * y_min - y0 + 2 * k * H)
        return ys

    def _image_terms(self, xj: float, yj: float) -> list:
        """(x_img, y_img, sign) triples for all image charges of segment
        (xj, yj): x-only reflections (sign +1, Neumann), y-only reflections
        (sign -1, Dirichlet), and - when both wall pairs are configured -
        corner reflections (sign -1) combining one of each."""
        xs = self._image_xs(xj)
        ys = self._image_ys(yj)
        terms = [(x_img, yj, 1.0) for x_img in xs]
        terms += [(xj, y_img, -1.0) for y_img in ys]
        if xs and ys:
            terms += [(x_img, y_img, -1.0) for x_img in xs for y_img in ys]
        return terms

    def build_and_solve(self):
        N = self.N
        P = self.n_pairs
        size = N + P
        A = np.zeros((size, size))
        b = np.zeros(size)

        for i, (xi, yi, li, phi_i) in enumerate(self.segments):
            for j, (xj, yj, lj, _) in enumerate(self.segments):
                if i == j:
                    A[i, j] = lj * (1.0 - np.log(lj / 2.0))
                else:
                    r = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    A[i, j] = -np.log(r) * lj

                for xj_img, yj_img, sign in self._image_terms(xj, yj):
                    r_img = np.sqrt((xi - xj_img)**2 + (yi - yj_img)**2 + 1e-12)
                    A[i, j] += sign * (-np.log(r_img)) * lj

            # RHS: phi conocido o acoplado
            if phi_i is not None:
                b[i] = phi_i
            else:
                pair_idx = self.segment_group[i]
                A[i, N + pair_idx] = -1.0
                b[i] = 0.0

        for pair_idx, offset in enumerate(self.coupled_offsets):
            row = N + pair_idx
            for j in range(offset, len(self.segments)):
                if self.segment_group[j] == pair_idx:
                    A[row, j] = self.segments[j][2]
            b[row] = 0.0

        solution = np.linalg.solve(A, b)
        self.sigma = solution[:N]
        self.phi_coupled = solution[N:]
        return self.sigma

    def compute_phi_grid(self, grid_x, grid_y):
        phi = np.zeros_like(grid_x, dtype=float)
        for k, (xj, yj, lj, _) in enumerate(self.segments):
            r = np.sqrt((grid_x - xj)**2 + (grid_y - yj)**2 + 1e-10)
            phi += -np.log(r) * lj * self.sigma[k]

            for xj_img, yj_img, sign in self._image_terms(xj, yj):
                r_img = np.sqrt((grid_x - xj_img)**2 + (grid_y - yj_img)**2 + 1e-10)
                phi += sign * (-np.log(r_img)) * lj * self.sigma[k]
        return phi
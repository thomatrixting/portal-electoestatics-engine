"""
This file contains example scenes that demonstrate some of this program's features
"""


from simulation import Simulation
from portals import *
from masks import *
from test_charge import TestCharge
from colors import default_color_mapper, colormap_plasma, extra_mapper

def _anchors(sim_width, sim_height, extend=False):
    """Special preset with potential """

    if extend:
        sim_min_width = -sim_width * 0.25
        sim_max_width = sim_width * 1.25
    else:
        sim_min_width = 0
        sim_max_width = sim_width
    return [
        PotentialAnchor(RectangleMask(sim_min_width, sim_max_width, 0, 1), 1.0),
        PotentialAnchor(RectangleMask(sim_min_width, sim_max_width, sim_height-1, sim_height), 0.0),
    ]

def _null_anchors(sim_width, sim_height):
    """Special preset with potential """
    return [
        PotentialAnchor(RectangleMask(0, sim_width, 0, 1), 0.0),
        PotentialAnchor(RectangleMask(0, sim_width, sim_height-1, sim_height), 0.0),
    ]

#scenes

def equipotential_scene(solver='sor',pinned=True,distance_portals=120) -> Simulation:
    """A scene with a couple of portals and a material object"""
    W, H = 800, 400
    cx = W // 2
    cy = H // 2
    
    # Valores estáticos multiplicados por 2
    portals_width = 80  
    apart_d = distance_portals       # Distancia vertical de los portales
    shift = 120         # Distancia entre portales
    
    # Se ajusta automáticamente al nuevo W (800 * 0.35 = 280)
    px = int(W * 0.35)
    
    p1 = Portal(RectangleMask(px-portals_width+shift, px+portals_width+shift, cy-apart_d, cy-apart_d+1), (255, 0, 0), facing_positive=True)
    p2 = Portal(RectangleMask(px-portals_width-shift, px+portals_width-shift, cy+apart_d, cy+apart_d+1), (0, 0, 255), facing_positive=False)

    # Radio y posición inicial multiplicados por 2
    r = 12
    y_start = 20
    
    obs_1 = MaterialObject(RectangleMask(px-r-shift, px+r-shift, y_start-r, y_start+r),
                         color=(80, 220, 120), pinned=pinned, label="obj 1", active=True,
                         charge=1.0, mass=2.0)

    # Se ajusta automáticamente al nuevo W (800 * 0.8 = 640)
    x2_start = int(W * 0.8)
    
    obs_2 = MaterialObject(RectangleMask(x2_start-r, x2_start+r, y_start-r, y_start+r),
                        color=(80, 220, 120), pinned=pinned, label="obj 2", active=True,
                        charge=1.0, mass=2.0)
    
    return Simulation(
        *_anchors(W, H), CouplePortal(p1, p2), obs_1, obs_2, 
        sim_width=W, sim_height=H,
        px_scale=2, 
        iterations_per_frame=500,
        sor_omega=1.8,
        isoline_count=15,
        solver_mode=solver,
        show_vectors=True,
        show_isolines=True,
        color_mapper=extra_mapper()
    )

def close_portals_scene(solver='sor',pinned=True,distance_portals=120) -> Simulation:
    """A scene with a couple of portals and a material object"""
    W, H = 500, 400
    cx = W // 2
    cy = H // 2
    
    # Valores estáticos multiplicados por 2
    portals_width = 70  
    apart_d = distance_portals       # Distancia vertical de los portales
    shift = 100         # Distancia entre portales
    
    # Se ajusta automáticamente al nuevo W (800 * 0.35 = 280)
    px = int(W * 0.5)
    
    p1 = Portal(RectangleMask(px-portals_width+shift, px+portals_width+shift, cy-apart_d, cy-apart_d+1), (255, 0, 0), facing_positive=True)
    p2 = Portal(RectangleMask(px-portals_width-shift, px+portals_width-shift, cy+apart_d, cy+apart_d+1), (0, 0, 255), facing_positive=False)
    
    return Simulation(
        *_anchors(W, H, extend=True), CouplePortal(p1, p2), 
        sim_width=W, sim_height=H,
        px_scale=2, 
        iterations_per_frame=500,
        sor_omega=1.8,
        isoline_count=15,
        solver_mode=solver,
        show_vectors=True,
        show_isolines=True,
        color_mapper=extra_mapper(),
        mom_images=True
    )

def falling_object_scene(solver='sor',pinned=True) -> Simulation:
    """A scene with a couple of portals and a material object"""
    W, H = 500,500
    cx = W // 2
    cy = H // 2
    d = 100
    r = 10
    p1 = Portal(RectangleMask(cx-d,cx+d,d,d), (255, 0, 0), facing_positive=True)
    p2 = Portal(RectangleMask(cx-d,cx+d, H-d, H-d), (0, 0, 255), facing_positive=False)

    obs = MaterialObject(RectangleMask(cx-r, cx+r, cy-r, cy+r),
                         color=(80, 220, 120), pinned=pinned, label="Conductor",active=True,
                         charge=5.0, mass=0.5)
    
    
    return Simulation(
        *_anchors(W, H), CouplePortal(p1, p2), obs, 
        sim_width=W, sim_height=H,
        px_scale=2,
        iterations_per_frame=500,
        sor_omega=1.8,
        isoline_count=15,
        solver_mode=solver,
        show_vectors=True,
        show_isolines=True,
        color_mapper=extra_mapper()
    )

#testing

def test_portals_scene_mom(solver='sor') -> Simulation:
    W, H = 80, 80
    top    = PotentialAnchor(RectangleMask(0, W, 0, 1),      1.0)
    bottom = PotentialAnchor(RectangleMask(0, W, H-1, H),    0.0)
    cond   = PotentialAnchor(CircleMask(W//2, H//2, 10),     0.5)  # conductor como anchor
    return Simulation(
        top, bottom, cond,
        sim_width=W, sim_height=H,
        px_scale=5,
        solver_mode="mom",
        show_isolines=True,
    )

def test_portals_scene(solver='sor') -> Simulation:
    """A scene with a couple of portals and a material object"""
    W, H = 500,500
    cx = W // 2
    cy = H // 2
    d = 100
    r = 50
    p1 = Portal(RectangleMask(cx-d,cx+d,d,d), (255, 0, 0), facing_positive=True)
    p2 = Portal(RectangleMask(cx-d,cx+d, H-d, H-d), (0, 0, 255), facing_positive=False)

    obs = MaterialObject(RectangleMask(cx-r, cx+r, cy-r, cy+r),
                         color=(80, 220, 120), pinned=False, label="Conductor",active=True,
                         charge=0.0, mass=1.0)
    
    
    return Simulation(
        *_anchors(W, H), CouplePortal(p1, p2), obs, 
        sim_width=W, sim_height=H,
        px_scale=1,
        iterations_per_frame=500,
        sor_omega=1.8,
        isoline_count=15,
        solver_mode=solver
    )

def test_gradient() -> Simulation:
    """A scene with a couple of portals and a material object"""
    W, H = 500,500
    cx = W // 2
    cy = H // 2
    d = 100
    r = 50
    p1 = Portal(RectangleMask(cx-d,cx+d,d,d), (255, 0, 0), facing_positive=True)
    p2 = Portal(RectangleMask(cx-d,cx+d, H-d, H-d), (0, 0, 255), facing_positive=False)

    obs = MaterialObject(RectangleMask(cx-r, cx+r, cy-r, cy+r),
                         color=(80, 220, 120), pinned=False, label="Conductor",active=True,
                         charge=0.0, mass=1.0)
    
    cond   = PotentialAnchor(CircleMask(cx, cy, 10/2),     0.5) 

    
    return Simulation(
        *_anchors(W, H), CouplePortal(p1, p2), obs, cond,
        sim_width=W, sim_height=H,
        px_scale=1,
        iterations_per_frame=500,
        sor_omega=1.9,
        isoline_count=15,
        solver_mode="mom"
    )

def vertical_portals_many_objects_scene() -> Simulation:
    """
    Two vertical portals with a real vertical potential (gravity pulling
    down) and a grid of many small material objects falling between them
    """
    W, H = 220, 120
    p1 = Portal(RectangleMask(130, 180,30, 30), (255, 0, 0), facing_positive=True)
    p2 = Portal(RectangleMask(30, 80,90, 90), (0, 0, 255), facing_positive=False)

    objs = []
    rows, cols = 10, 10
    spacing = 5
    x0, y0 = 40, 30
    radius = 0
    for i in range(rows):
        for j in range(cols):
            cx = x0 + j * spacing
            cy = y0 + i * spacing
            objs.append(MaterialObject(
                RectangleMask(cx-radius, cx+radius, cy-radius, cy+radius),
                color=(80, 220, 120), pinned=False,
                label=f"Ball_{i}_{j}", mass=1.0, active=True))

    return Simulation(
        *_anchors(W, H), CouplePortal(p1, p2), *objs,
        sim_width=W, sim_height=H,
        px_scale=6,
        iterations_per_frame=50,
        sor_omega=1.8,
        isoline_count=15,
    )


def example_portal_on_capacitor() -> Simulation:
    """one paralel portals between a capacitor that would be the roof, to another portal far away from the other portal"""
    W,H = 400,200
    px_sclae = 3

    cap_length = H/2
    portal_lenght= H/4

    cap_delta_with_top = (H - cap_length)/2
    portal_delta_with_top = (H - portal_lenght)/2

    #define the capacitor
    capacitor = [
        PotentialAnchor(RectangleMask(W/8,W/8,cap_delta_with_top,cap_delta_with_top + cap_length),1.0,),
        PotentialAnchor(RectangleMask(3*W/8,W/8,cap_delta_with_top,cap_delta_with_top + cap_length),0)
    ]

    p1 = Portal(RectangleMask(W/4, W/4, portal_delta_with_top, portal_delta_with_top + portal_lenght), color=(255, 153, 0),facing_positive=False)
    p2 = Portal(RectangleMask(3*W/4, 3*W/4, portal_delta_with_top, portal_delta_with_top + portal_lenght), color=(0, 204, 255),facing_positive=False)

    return Simulation(
        *capacitor, CouplePortal(p1,p2),
        sim_width=W, sim_height=H,
        px_scale=px_sclae,
        iterations_per_frame=2000,
        sor_omega=1.7,
        isoline_count=50,
        solver_mode="mom"
    )


def example_couple_portals() -> Simulation:
    """Two parallel portals"""
    #W, H = 120, 120
    W,H = 200,250
    p1 = Portal(RectangleMask(40, 80, 40, 40), color=(255, 153, 0), facing_positive=True)
    p2 = Portal(RectangleMask(40, 80, 80, 80), color=(0, 204, 255), facing_positive=False)

    return Simulation(
        *_anchors(W, H), CouplePortal(p1, p2),
        sim_width=W, sim_height=H,
        px_scale=4,
        iterations_per_frame=40,
        sor_omega=1.7,
    )



def example_advanced() -> Simulation:
    """FixedPotential + object"""
    W, H = 120, 100
    top = FixedPotentialPortal(RectangleMask(20, 100, 30, 33), 0.8, (255, 80, 80))
    bot = FixedPotentialPortal(RectangleMask(20, 100, 67, 70), 0.2, (80, 120, 255))
    obs = MaterialObject(CircleMask(60, 50, 8),
                         color=(80, 220, 120), pinned=True, label="Conductor")
    return Simulation(
        top, bot, obs,
        sim_width=W, sim_height=H,
        px_scale=6,
        iterations_per_frame=60,
        sor_omega=1.75,
    )


def example_couple_circles() -> Simulation:
    """Two circular portals"""
    W, H = 120, 100
    p1 = Portal(CircleMask(cx=35, cy=50, radius=10), color=(255, 100, 50),
                facing_positive=True, normal_axis="x")
    p2 = Portal(CircleMask(cx=85, cy=50, radius=10), color=(50, 200, 255),
                facing_positive=False, normal_axis="x")

    return Simulation(
        *_anchors(W, H), CouplePortal(p1, p2),
        sim_width=W, sim_height=H,
        px_scale=6,
        iterations_per_frame=50,
        sor_omega=1.8,
        isoline_count=15,
    )


def triple_portals() -> Simulation:
     W, H = 120, 100
     p1 = Portal(RectangleMask(40, 60, 30, 30), (255, 0, 0))
     p2 = Portal(RectangleMask(40, 60, 60, 60), (0, 255, 0))
     p3 = Portal(RectangleMask(40, 60, 90, 90), (0, 0, 255))
     return Simulation(
         *_anchors(W, H), MultiPortal((p1, p2, p3)),
         sim_width=W, sim_height=H,
         px_scale=6,
         iterations_per_frame=50,
         sor_omega=1.8,
         isoline_count=15,
     )

def triple_portals_mom() -> Simulation:
    W, H = 600, 500  # escalado x5 desde 120,100

    p1 = Portal(RectangleMask(200, 300, 150, 150), (255, 0, 0))
    p2 = Portal(RectangleMask(200, 300, 300, 300), (0, 255, 0))
    p3 = Portal(RectangleMask(200, 300, 450, 450), (0, 0, 255))

    return Simulation(
        *_anchors(W, H), MultiPortal((p1, p2, p3)),
        sim_width=W, sim_height=H,
        px_scale=1,
        isoline_count=15,
        solver_mode="mom",
    )

def example_mom_conductor():
    from mom_mesh import BoundaryMesh
    from mom_solver import MOMSolver2D
    import numpy as np

    W, H = 120, 120
    dx = 1.0

    # Grilla de coordenadas para reconstrucción
    xs = np.arange(W) * dx
    ys = np.arange(H) * dx
    grid_x, grid_y = np.meshgrid(xs, ys)

    # Defines conductores como arrays booleanos directamente
    # (o los sacas de tus Masks existentes)
    from masks import CircleMask, RectangleMask

    mask_conductor = CircleMask(cx=60, cy=60, radius=15)
    mask_plate_top = RectangleMask(0, W, 0, 1)
    mask_plate_bot = RectangleMask(0, W, H-1, H)

    # Extraes la representación booleana de cada mask
    bool_conductor = mask_conductor.to_array(W, H)  # ajusta al método real
    bool_top       = mask_plate_top.to_array(W, H)
    bool_bot       = mask_plate_bot.to_array(W, H)

    meshes = [
        BoundaryMesh(bool_top,       potential_value=1.0, dx=dx),
        BoundaryMesh(bool_bot,       potential_value=0.0, dx=dx),
        BoundaryMesh(bool_conductor, potential_value=0.5, dx=dx),
    ]

    solver = MOMSolver2D(meshes)
    solver.build_and_solve()
    phi_grid = solver.compute_phi_grid(grid_x, grid_y)

    # phi_grid es un numpy array 2D listo para visualizar
    # Puedes pasarlo a Simulation como campo inicial o visualizarlo con matplotlib
    return phi_grid, solver.sigma

def example_mom() -> Simulation:
    W, H = 80, 80
    top    = PotentialAnchor(RectangleMask(0, W, 0, 1),      1.0)
    bottom = PotentialAnchor(RectangleMask(0, W, H-1, H),    0.0)
    cond   = PotentialAnchor(CircleMask(W//2, H//2, 10),     0.5)  # conductor como anchor
    return Simulation(
        top, bottom, cond,
        sim_width=W, sim_height=H,
        px_scale=5,
        solver_mode="mom",
        show_isolines=True,
    )

def example_mom_couple() -> Simulation:
    W, H = 80, 80
    top    = PotentialAnchor(RectangleMask(0, W, 0, 1),      1.0)
    bottom = PotentialAnchor(RectangleMask(0, W, H-1, H),    0.0)
    # Portales horizontales: uno arriba y uno abajo
    p1 = Portal(RectangleMask(20, 60, 20, 21), (255, 153, 0))
    p2 = Portal(RectangleMask(20, 60, 59, 60), (0, 204, 255))
    return Simulation(
        top, bottom, CouplePortal(p1, p2),
        sim_width=W, sim_height=H,
        px_scale=5,
        solver_mode="mom",
        show_isolines=True,
    )

def example_mom_carga_entre_portales() -> Simulation:
    """Carga flotante entre dos portales, sin campo externo"""
    W, H = 80, 80
    # Sin placas top/bottom — solo los portales y la carga
    p1 = Portal(RectangleMask(10, 11, 15, 65), (255, 153, 0))
    p2 = Portal(RectangleMask(69, 70, 15, 65), (0, 204, 255))
    carga = PotentialAnchor(CircleMask(W//2, H//2, 6), 1.0)
    return Simulation(
        CouplePortal(p1, p2), carga,
        sim_width=W, sim_height=H,
        px_scale=5,
        solver_mode="mom",
        show_isolines=True,
    )


def example_mom_cargas_afuera() -> Simulation:
    """Dos portales verticales con cargas a los lados externos"""
    W, H = 80, 80
    top    = PotentialAnchor(RectangleMask(0, W, 0, 1),   0.0)
    bottom = PotentialAnchor(RectangleMask(0, W, H-1, H), 0.0)
    # Portales verticales en el centro
    p1 = Portal(RectangleMask(35, 36, 20, 60), (255, 153, 0))
    p2 = Portal(RectangleMask(44, 45, 20, 60), (0, 204, 255))
    # Cargas a los lados externos de los portales
    carga_izq = PotentialAnchor(CircleMask(15, H//2, 5), 0.9)
    carga_der = PotentialAnchor(CircleMask(65, H//2, 5), 0.1)
    return Simulation(
        CouplePortal(p1, p2), carga_izq, carga_der,
        sim_width=W, sim_height=H,
        px_scale=5,
        solver_mode="mom",
        show_isolines=True,
    )

def axiom_continuity_sor() -> Simulation:
    """
    Axiom: potential continuity across portal boundaries (SOR solver).
    """
    W, H = 200, 200
    cx = W // 2
    portal_h = H // 2
    portal_y0 = (H - portal_h) // 2
    portal_y1 = portal_y0 + portal_h
    gap = 4

    # anchors LATERALES: campo apunta en x, hacia el portal
    left_anchor  = PotentialAnchor(RectangleMask(0, 1, 0, H), 0.7)
    right_anchor = PotentialAnchor(RectangleMask(W-1, W, 0, H), 0.3)

    p1 = Portal(
        RectangleMask(cx - gap, cx - gap, portal_y0, portal_y1),
        color=(255, 153, 0),
        facing_positive=False,
        normal_axis="x",
    )
    p2 = Portal(
        RectangleMask(cx + gap, cx + gap, portal_y0, portal_y1),
        color=(0, 204, 255),
        facing_positive=True,
        normal_axis="x",
    )

    sim = Simulation(
        left_anchor, right_anchor, CouplePortal(p1, p2),
        sim_width=W, sim_height=H,
        px_scale=3,
        solver_mode="sor",
        iterations_per_frame=500,
        sor_omega=1.85,
        show_isolines=True,
        isoline_count=20,
    )
    sim.test_charges.append(
        TestCharge(x=20.0, y=float(H // 2),
                   vx=0.0, vy=0.0,
                   charge=0.20, mass=3.0,
                   color=(255, 255, 0),
                   trail_len=800)
    )
    return sim


def axiom_continuity_mom() -> Simulation:
    """
    Axiom: potential continuity across portal boundaries (MOM solver).
    """
    W, H = 200, 200
    cx = W // 2
    portal_h = H // 2
    portal_y0 = (H - portal_h) // 2
    portal_y1 = portal_y0 + portal_h
    gap = 4

    left_anchor  = PotentialAnchor(RectangleMask(0, 1, 0, H), 0.7)
    right_anchor = PotentialAnchor(RectangleMask(W-1, W, 0, H), 0.3)

    p1 = Portal(
        RectangleMask(cx - gap, cx - gap, portal_y0, portal_y1),
        color=(255, 153, 0),
        facing_positive=False,
        normal_axis="x",
    )
    p2 = Portal(
        RectangleMask(cx + gap, cx + gap, portal_y0, portal_y1),
        color=(0, 204, 255),
        facing_positive=True,
        normal_axis="x",
    )

    sim = Simulation(
        left_anchor, right_anchor, CouplePortal(p1, p2),
        sim_width=W, sim_height=H,
        px_scale=3,
        solver_mode="mom",
        show_isolines=True,
        isoline_count=20,
    )
    sim.test_charges.append(
        TestCharge(x=20.0, y=float(H // 2),
                   vx=0.0, vy=0.0,
                   charge=0.20, mass=3.0,
                   color=(255, 255, 0),
                   trail_len=800)
    )
    return sim
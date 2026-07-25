"""
Main simulation script

Objects:
  CouplePortal(p1, p2)           pair of portals with equal potential
  FixedPotentialPortal(mask, v)  region with fixed potential
  PotentialAnchor(mask, v)       potential anchor

Masks:
  RectangleMask(x_min, x_max, y_min, y_max)
  CircleMask(cx, cy, radius)
  PointMask(x, y)
  LineMask(x1, y1, x2, y2, thickness)
  PolygonMask([(x0,y0), ...])
  FunctionMask("expression(x, y)")

Controls:
  M - toggle display mode: gravitational acceleration / potential
  V - vectors on/off
  I - isolines on/off

  Drag - drag any portal with the mouse
  SIM tab - render and physics parameters
  SCENE tab - scene objects, add, presets, inspector

See README.md for details
"""


from scenes import *

def equipotencials_closeportals_falling():
    #sim = equipotential_scene(solver='sor')
    #sim = close_portals_scene(solver='sor', distance_portals=120)
    #sim = close_portals_scene(solver='sor', distance_portals=80)
    #sim = close_portals_scene(solver='sor', distance_portals=40)
    #sim = close_portals_scene(solver='mom', distance_portals=0)
    #sim = falling_object_scene(solver='sor',pinned=True)
    sim = falling_object_scene(solver='mom',pinned=False)
    #sim = falling_object_scene(solver='sor',pinned=True)
    #sim = capacitor_scene()
    #sim = capacitor_scene_corrected()
    sim.run()

def main() -> None:
    equipotencials_closeportals_falling()
    
    # sim = equipotential_scene(solver='mom')
    #sim = faling_object_scene()

    #sim = vertical_portals_many_objects_scene()
    #sim = test_portals_scene()  
    #sim = example_mom_carga_entre_portales()
    #sim.run()
    #sim = example_mom_cargas_afuera()
    #sim.run()
    #sim = example_mom_couple()
    #sim.run()
    #sim = example_mom()
    #sim = test_portals_scene()
    #sim = test_gradient()
    # sim = example_portal_on_capacitor()
    #sim = example_couple_portals()  # Starting scene
    #sim = triple_portals()
    #sim = axiom_continuity_sor()
    # sim = axiom_continuity_mom()

    #sim.run()


if __name__ == "__main__":
    main()

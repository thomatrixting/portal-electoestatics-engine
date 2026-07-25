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
    sim = close_portals_scene(solver='mom', distance_portals=0)
    #sim = falling_object_scene(solver='sor',pinned=True)
    sim.run()
    sim = falling_object_scene(solver='mom',pinned=True)
    sim.run()


def main() -> None:
    sim = equipotencials_closeportals_falling()
    
    # sim = equipotential_scene(solver='mom')
    # sim = faling_object_scene()
    # sim = vertical_portals_many_objects_scene()
    # sim = test_portals_scene()  
    # sim = example_mom_carga_entre_portales()
    # sim = example_mom_cargas_afuera()
    # sim = example_mom_couple()
    # sim = example_mom()
    # sim = test_portals_scene()
    # sim = test_gradient()
    # sim = example_portal_on_capacitor()
    # sim = example_couple_portals()  # Starting scene
    # sim = triple_portals()
    # sim = axiom_continuity_sor()
    # sim = axiom_continuity_mom()
    #sim = triple_portals_mom()
    #sim.run()
    #sim = triple_portals_sor()
    #sim.run()
    #sim = flux_zero_test_scene('mom')
    #sim.run()
    #sim = flux_zero_test_scene('sor')
    #sim.run()
    #sim = triple_portal_flux_probes_scene('mom')
    #sim.run()
    #sim = triple_portal_flux_probes_scene('sor')
    #sim.run()
    #sim = falling_flux_scene('sor')
    #sim.run()
    #sim = falling_flux_scene('mom')
    #sim.run()
    #sim = object_at_portal_scene('sor')
    #sim.run()    
    #sim = object_at_portal_scene('mom')
    #sim.run()

    sim.run()


if __name__ == "__main__":
    main()

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
    #sim = falling_object_scene(solver='mom',pinned=True)
    #sim = falling_object_scene(solver='sor',pinned=True)
    sim = capacitor_scene()
    sim.run()


def main() -> None:
    equipotencials_closeportals_falling()
    #TODO organize scenes when everyone uploads them


if __name__ == "__main__":
    main()

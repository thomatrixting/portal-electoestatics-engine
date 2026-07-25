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

Each function below runs the scene(s) used for one subsection of the
"Resultados" section of explanation/report.tex, so the report and the
code stay traceable to each other.
"""

from scenes import *


# --- 4.1 Equipotenciales ------------------------------------------------

def equipotenciales_primera_configuracion(solver='mom', distance_portals=80) -> None:
    """4.1.1 Primera configuración: par de portales acoplados sobre un
    fondo de gradiente vertical uniforme, variando la separación vertical
    d entre sus centros (figuras campo_por_distancia_{mom,sor}_{0,40,80,120})."""
    #sim = close_portals_scene(solver=solver, distance_portals=0)
    #sim = close_portals_scene(solver=solver, distance_portals=40)
    #sim = close_portals_scene(solver=solver, distance_portals=80)
    sim = close_portals_scene(solver=solver, distance_portals=distance_portals)
    sim.run()


def equipotenciales_segunda_configuracion(solver) -> None:
    """4.1.2 Segunda configuración: portales enfrentados en y=80 y y=120
    con una carga de prueba que se teletransporta (figuras potential_MOM,
    potential_SOR, vel_sor, vel_MoM)."""
    sim = axiom_continuity(solver='mom') if solver == 'mom' else axiom_continuity(solver='sor')
    sim.run()


# --- 4.2 Conservación de la velocidad al atravesar un portal -----------

def conservacion_velocidad(solver='mom', pinned=True) -> None:
    """Dos objetos que caen en la misma escena: uno atraviesa el par de
    portales y el otro cae libremente, para comparar su rapidez final
    (figuras equipotential_field_{mom,sor}, velocidad_{MOM,SOR})."""
    sim = equipotential_scene(solver=solver, pinned=pinned)
    sim.run()


# --- 4.3 Objeto oscilante entre portales --------------------------------

def objeto_oscilante(solver='mom', pinned=False) -> None:
    """Portales enfrentados directamente (d=0) con un conductor cargado
    (q=5, m=0.5) que oscila entre las dos bocas por la zona de equilibrio
    del campo (figuras campo_objeto_oscilante, trayectoria_objeto_oscilante)."""
    sim = falling_object_scene(solver=solver, pinned=pinned)
    sim.run()


# --- 4.4 Discontinuidad del campo a través de los portales -------------

def discontinuidad_capacitor(corrected=False) -> None:
    """Portal entre las placas de un capacitor acoplado a un portal lejano
    en el vacío, sin (raw) y con (corrected) ancla de potencial fija detrás
    de cada boca (figuras capacitor_raw, capacitor_corrected). Solo MOM,
    el ancho necesario hace inviable SOR."""
    sim = capacitor_scene_corrected() if corrected else capacitor_scene()
    sim.run()

def main() -> None:
    # equipotenciales_primera_configuracion(solver='mom', distance_portals=80)
    equipotenciales_segunda_configuracion(solver='mom')
    # conservacion_velocidad(solver='mom')
    # objeto_oscilante(solver='mom')
    # discontinuidad_capacitor(corrected=False)


if __name__ == "__main__":
    main()

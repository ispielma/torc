# Path manipulation to ensure the example can run from the project directory:
import sys
import pathlib
path = sys.path.insert(0, str(pathlib.Path(__file__).absolute().parent.parent))

from torc import inch, X, Z, COPPER, SILVER, RacetrackCoil, show 
racetrack = RacetrackCoil(
    r0=(0, 0, 0),
    n=Z,
    n_perp=X,
    width=3 * inch,
    length=5 * inch,
    height=1 * inch,
    R_inner=1 * inch,
    R_outer=2 * inch,
    n_turns=1,
    arc_segs=12,
    cross_sec_segs=12,
)

racetrack.show(lines=True, surfaces=False, color=SILVER)
racetrack.show(lines=False, surfaces=True, opacity=0.5, color=COPPER)
show()

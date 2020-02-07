# torc

Torc is a Python library for computing magnetic fields resulting from magnetic coils
such as loops, lines, round coils and racetrack coils of rectangular cross section,
positioned and oriented arbitrarily in space.

Documentation and examples yet to come, but there is one example file you can run:

```bash
python -m torc.example
```
to display an arrangement of coils that represents the magnetic transport assembly for
the cold atom experiment in the RbChip lab in the Spielman group at NIST:

![rb_chip_coils.png](rb_chip_coils.png)

## Requirements

Requires `numpy` and `scipy`. '`mayavi` is an optional requirement, it is needed for
displaying 3D renderings of the magnetic coils.


## Installation

From pip: `pip install torc`
From source: Clone this repository and tun `python setup.py install`
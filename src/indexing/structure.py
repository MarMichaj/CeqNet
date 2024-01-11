from typing import Final

idx: Final[str] = "_idx"

# structure
Z: Final[str] = "_atomic_numbers"  #: nuclear charge
position: Final[str] = "_positions"  #: atom positions
R: Final[str] = position  #: atom positions

cell: Final[str] = "_cell"  #: unit cell
pbc: Final[str] = "_pbc"  #: periodic boundary conditions

seg_m: Final[str] = "_seg_m"  #: start indices of systems
idx_m: Final[str] = "_idx_m"  #: indices of systems
idx_i: Final[str] = "_idx_i"  #: indices of center atoms
idx_j: Final[str] = "_idx_j"  #: indices of neighboring atoms

Rij: Final[str] = "_Rij"

Y_ij: Final[str] = "_Y_ij"

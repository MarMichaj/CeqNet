data_path = "/Users/martin/Master/data/gdb9-14b/dsC7O2H10nsd.xyz"

import ase

from ase.io.extxyz import read_xyz as read

#data = read(file=data_path)
data = ase.io.read(filename=data_path, format='extxyz')

print(data)
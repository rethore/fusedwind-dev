import numpy as np
from fusedwind.turbine.structure import read_bladestructure, write_bladestructure, interpolate_bladestructure

st3d = read_bladestructure('data/DTU10MW')
write_bladestructure(st3d, 'test_out')
st3dn = read_bladestructure('test_out')

#s_new = np.linspace(0, 1, 20)
#st3dn = interpolate_bladestructure(r.st3d, s_new)

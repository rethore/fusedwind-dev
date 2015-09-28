
import time
import numpy as np

from openmdao.core.mpi_wrap import MPI

from fusedwind.turbine.structure import *

# stuff for running in parallel under MPI
def mpi_print(prob, *args):
    """ helper function to only print on rank 0"""
    if prob.root.comm.rank == 0:
        print(args)

if MPI:
    # if you called this script with 'mpirun', then use the petsc data passing
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    # if you didn't use `mpirun`, then use the numpy data passing
    from openmdao.core import BasicImpl as impl



# read the blade structure
st3d = read_bladestructure('data/DTU10MW')

# and interpolate onto new distribution
nsec = 8
s_new = np.linspace(0, 1, nsec)
st3dn = interpolate_bladestructure(st3d, s_new)

# dummy data for blade surface
surf = np.zeros((200, nsec, 3))
for i in range(nsec):
    surf[:, i, 2] = [s_new[i]]*200


# define which code is used to compute the cross-sectional props
# in this case just the base class
config = {}
config['cs_props'] = {}
config['cs_props']['model'] = CSPropsBase
config['cs_props']['args'] = {}
config['cs_props']['promotes'] = []

t0 = time.time()
p = Problem(impl=impl, root=CSBeamStructure(config, st3dn, surf))
t1 = time.time()
p.setup()
t2 = time.time()
p.run()
mpi_print(p, 'init time %3.3f' % (t1 - t0))
mpi_print(p, 'setup time %3.3f' % (t2 - t1))
mpi_print(p, 'run time %3.3f' % (time.time() - t2))

mpi_print(p, 's %s' % ' '.join(map(str, p.root.unknowns['beam_structure'][:, 0])))

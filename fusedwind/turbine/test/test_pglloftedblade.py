
import numpy as np
import unittest

from openmdao.api import Problem, Group

from fusedwind.turbine.geometry import read_blade_planform,\
                                       redistribute_planform,\
                                       PGLLoftedBladeSurface,\
                                       PGLRedistributedPlanform


def configure(cfg):

    pf = read_blade_planform('data/DTU_10MW_RWT_blade_axis_prebend.dat')
    nsec = 8
    s_new = np.linspace(0, 1, nsec)
    pf = redistribute_planform(pf, s=s_new)

    cfg['redistribute_flag'] = False
    cfg['blend_var'] = np.array([0.241, 0.301, 0.36, 1.0])
    afs = []
    for f in ['data/ffaw3241.dat',
              'data/ffaw3301.dat',
              'data/ffaw3360.dat',
              'data/cylinder.dat']:

        afs.append(np.loadtxt(f))
    cfg['base_airfoils'] = afs
    d = PGLLoftedBladeSurface(cfg, size_in=nsec, size_out=(200, nsec, 3), suffix='_st')
    p = Problem(root=Group())
    r = p.root.add('blade_surf', d, promotes=['*'])
    p.setup()
    for k, v in pf.iteritems():
        if k+'_st' in p.root.blade_surf.params.keys():
            p.root.blade_surf.params[k+'_st'] = v

    return p

class PGLLoftedBladeSurfaceTestCase(unittest.TestCase):

    def test_surf(self):

        cfg = {}

        p = configure(cfg)
        p.run()
        self.assertAlmostEqual(np.sum(p['blade_surface_st']), 775.21809362184081, places=6)

if __name__ == '__main__':

    unittest.main()
    # p = configure({})
    # p.run()

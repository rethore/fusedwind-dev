
import unittest
import numpy as np

from openmdao.api import Group, Problem

from fusedwind.turbine.structure import read_bladestructure, \
                                        interpolate_bladestructure, \
                                        SplinedBladeStructure


def configure():

    st3d = read_bladestructure('data/DTU10MW')
    st3dn = interpolate_bladestructure(st3d, np.linspace(0, 1, 8))


    p = Problem(root=Group())
    spl = p.root.add('st_splines', SplinedBladeStructure(st3dn), promotes=['*'])
    spl.add_spline('DP04', np.linspace(0, 1, 4), spline_type='bezier')
    spl.add_spline('r04uniaxT', np.linspace(0, 1, 4), spline_type='bezier')
    spl.add_spline('w02biaxT', np.linspace(0, 1, 4), spline_type='bezier')
    spl.configure()
    p.setup()
    return p

class TestSplinedBladeStructure(unittest.TestCase):

    def test_it(self):

        r04uniax= np.array([ 0.008     ,  0.03097776,  0.04077973,  0.04464869,  0.04330389,
                             0.03712204,  0.02271313,  0.0015    ])
        w02biax = np.array([ 0.0026    ,  0.00334698,  0.00596805,  0.00784869,  0.00869437,
                             0.00836037,  0.00643492,  0.0013    ])
        DP04 = np.array([-0.49644126, -0.48696236, -0.34930237, -0.31670771, -0.33028392,
                         -0.35351936, -0.3801164 , -0.3808905 ])

        p = configure()
        p['r04uniaxT_C'][2] = 0.01
        p['w02biaxT_C'][2] = 0.01
        p['DP04_C'][1] = 0.1
        p.run()

        self.assertEqual(np.testing.assert_array_almost_equal(p['r04uniaxT'], r04uniax, decimal=6), None)
        self.assertEqual(np.testing.assert_array_almost_equal(p['w02biaxT'], w02biax, decimal=6), None)
        self.assertEqual(np.testing.assert_array_almost_equal(p['DP04'], DP04, decimal=6), None)


if __name__ == '__main__':

    unittest.main()
    # p = configure()
    # p['r04uniaxT_C'][2] = 0.01
    # p['w02biaxT_C'][2] = 0.01
    # p['DP04_C'][1] = 0.1
    #
    # p.run()

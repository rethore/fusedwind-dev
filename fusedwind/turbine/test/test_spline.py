
import unittest
import numpy as np
from openmdao.api import Problem, Group
from fusedwind.turbine.geometry import FFDSpline

expected = np.array([ 0.        ,  0.16474038,  0.32586582,  0.47988383,  0.62354352,
        0.75394814,  0.86865795,  0.96578062,  1.04404672,  1.10286821,
        1.14237834,  1.16345188,  1.1677051 ,  1.15747556,  1.13578223,
        1.10626695,  1.07311899,  1.04098464,  1.01486427,  1.        ])


def configure():

    p = Problem(root=Group())
    s = np.linspace(0, 1, 20)
    P = np.sin(np.linspace(0, 1, 20)*np.pi)
    a = p.root.add('spla', FFDSpline('a', s, P, np.linspace(0, 1, 4)), promotes=['*'])
    p.setup()
    return p

class TestFFDSpline(unittest.TestCase):


    def test_it(self):
        p = configure()
        p['a_C'] = np.array([0, 0, 0, 1.])
        p.run()

        self.assertEqual(np.testing.assert_array_almost_equal(p['a'], expected, decimal=6), None)


if __name__ == '__main__':

    unittest.main()
    # import matplotlib.pylab as plt
    # p = configure()
    # p['a_C'] = np.array([0, 0, 0, 1.])
    # p.run()
    # plt.plot(p.root.spla.s, p.root.spla.Pinit)
    # plt.plot(p.root.spla.s, p['a'])
    # plt.show()

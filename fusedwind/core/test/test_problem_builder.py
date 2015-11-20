
import unittest
import yaml
import os

from fusedwind.core.problem_builder import FUSEDProblem
from fusedwind.core.test.simple_comps import SimpleComp

def configure():
    """
    configures a test problem with an IndepVarComp,
    ExecComp, and a simple multiplier comp
    """

    c = {'root':
            {'class': 'Group',
             'components':
                [
                {'name': 'x_c',
                'class': 'IndepVarComp',
                'parameter': ['p0', 6.],
                'promotes': ['*']},
                {'name': 'x_div',
                'class': 'ExecComp',
                'expr': ['x = p0 / 2.'],
                'parameters': {'p0': 25.},
                'promotes': ['*']},
                {'name': 'simple',
                 'class': 'fusedwind.core.test.simple_comps.SimpleComp',
                 'parameters':
                        {'multiplier': 4.},
                  'promotes': ['*']}]}}

    yaml.dump(c, open('simple.yml', 'w'))

    p = FUSEDProblem(filename='simple.yml')

    return p


class TestProblemBuilder(unittest.TestCase):

    def tearDown(self):

        os.remove('simple.yml')

    def test_pb(self):

        p = configure()
        p.setup()
        p.run()
        self.assertEqual(p['x'], 3.)
        self.assertEqual(p['y'], 12.)

if __name__ == '__main__':

    unittest.main()
    # p = configure()
    # p.setup()
    # p.run()


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

    def test_simple(self):

        c = {'root':
                {'class': 'Group',
                 'components':
                    [
                    {'name': 'simple',
                     'class': 'fusedwind.core.test.simple_comps.SimpleComp',
                     'parameters':
                            {'multiplier': 4.},
                      'promotes': ['*']}]}}

        yaml.dump(c, open('simple.yml', 'w'))

        p = FUSEDProblem(filename='simple.yml')
        p.setup()
        p.run()
        self.assertEqual(p['x'], 3.)
        self.assertEqual(p['y'], 12.)

    def test_indep(self):

        c = {'root':
                {'class': 'Group',
                 'components':
                    [
                    {'name': 'x_c',
                    'class': 'IndepVarComp',
                    'parameter': ['p0', 6.],
                    'promotes': ['*']}]}}

        yaml.dump(c, open('simple.yml', 'w'))

        p = FUSEDProblem(filename='simple.yml')
        p.setup()
        p.run()
        self.assertEqual(p['p0'], 6.)

    def test_listindeps(self):

        c = {'root':
                {'class': 'Group',
                 'components':
                    [
                    {'name': 'x_c',
                    'class': 'IndepVarComp',
                    'parameter': ['p0', 6.],
                    'promotes': ['*']}]}}
        yaml.dump(c, open('simple.yml', 'w'))
        p = FUSEDProblem(filename='simple.yml')
        p.setup()
        self.assertEqual(p.list_indepvars(), ['p0'])

    def test_exec(self):
        c = {'root':
                {'class': 'Group',
                 'components':
                    [
                    {'name': 'x_div',
                    'class': 'ExecComp',
                    'expr': ['p1 = p0 * 4', 'x = p1 / 2.'],
                    'parameters': {'p0': 25.},
                    'promotes': ['*']}]}}

        yaml.dump(c, open('simple.yml', 'w'))

        p = FUSEDProblem(filename='simple.yml')
        p.setup()
        p.run()
        self.assertEqual(p['p0'], 25.)
        self.assertEqual(p['x'], 50.)

    def test_combined(self):

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
        p.setup()
        p.run()
        self.assertEqual(p['x'], 3.)
        self.assertEqual(p['y'], 12.)


if __name__ == '__main__':

    unittest.main()
    # p = configure()
    # p.setup()
    # p.run()

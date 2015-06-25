from numpy.core.umath import sin, pi
import numpy as np
from fusedwind.core.openmdao_interface import fused_func, Inputs, Outputs
from fusedwind.variables import wind_speed, wind_direction, power
from openmdao.components.execcomp import ExecComp
from openmdao.components.paramcomp import ParamComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem

__author__ = 'pire'

import unittest


class TestFused_func(unittest.TestCase):

    def test_one_output(self):

        @fused_func(
            inputs = [wind_speed, wind_direction],
            outputs = [power])
        def my_silly_func(ws, wd):
            return ws**3.0 * sin(wd * pi/180.0)
        a = my_silly_func(3.0, 0.0)
        self.assertAlmostEqual(a(5.0, 90.0), 5.0**3.0 * sin(90.0 * pi/180.0))

    def test_several_outputs(self):

        @fused_func(
            inputs = [wind_speed, wind_direction],
            outputs = [('p1',power),('p2', power),('p3', power)])
        def my_silly_func(ws, wd):
            return [i * ws**3.0 * sin(wd * pi/180.) for i in range(3)]

        a = my_silly_func(3.0, 0.0)

        for k in ['p1', 'p2', 'p3']:
            self.assertTrue(k in a._unknowns_dict)

        self.assertAlmostEqual(a(5.0, 90.0), [i*5.0**3.0 for i in range(3)])

    def test_call_error(self):
        @fused_func(
            inputs = [wind_speed, wind_direction],
            outputs = [power])
        def my_silly_func(ws, wd):
            return ws**3.0 * sin(wd * pi/180.)

        a = my_silly_func(3.0, 0.0)
        with self.assertRaises(NotImplementedError,
                               msg='Only use args or kwargs independently, not both at the same time.'):
            a(3.0, wd=90.0)



class TestOpenMDAOInterface(unittest.TestCase):

    def setUp(self):
        @fused_func(
            inputs = [wind_speed, wind_direction],
            outputs = [power])
        def my_silly_func(ws, wd):
            return ws**3.0 * sin(wd * pi/180.0)

        self.func = my_silly_func()

    def test_add(self):
        pb = Problem(root=Group())
        pb.root.add('a', self.func)
        pb.setup()
        self.assertTrue(hasattr(pb.root,'a'))
        self.assertTrue('power' in pb.root.a.unknowns)

    def test_promotes(self):
        pb = Problem(root=Group())
        pb.root.add('a', self.func, promotes=['*'])
        pb.setup()
        self.assertTrue('power' in pb.root.unknowns)

    def test_connection(self):
        pb = Problem(root=Group())
        pb.root.add('a', self.func, promotes=['*'])
        pb.root.add('eq', ExecComp('y = power*2.0'), promotes=['*'])
        pb.setup()
        self.assertEqual(pb.root.connections, {'eq.power':'a.power'})

    def test_run(self):
        pb = Problem(root=Group())
        pb.root.add('iws', ParamComp('wind_speed', 5.0), promotes=['*'])
        pb.root.add('iwd', ParamComp('wind_direction', 90.0), promotes=['*'])
        pb.root.add('a', self.func, promotes=['*'])
        pb.root.add('eq', ExecComp('y = power*2.0'), promotes=['*'])
        pb.setup()
        pb.run()
        result = 5.0**3.0 * sin(90.0 * pi / 180.0)  * 2.0
        self.assertAlmostEqual(pb.root.unknowns['y'], result)

class TestSpecialOpenMDAO(unittest.TestCase):
    def test_same_types(self):
        @fused_func(
            inputs = [('ws1', wind_speed),
                      ('ws2', wind_speed),
                      wind_direction],
            outputs = [power])
        def my_silly_func(ws1, ws2, wd):
            return ((ws1 + ws2)/2.0)**3.0 * sin(wd * pi/180.0)

        pb = Problem(root=Group())
        ws1 = 4.0
        ws2 = 25.0
        wd = 45.0
        pb.root.add('iws1', ParamComp('ws1', ws1), promotes=['*'])
        pb.root.add('iws2', ParamComp('ws2', ws2), promotes=['*'])
        pb.root.add('iwd', ParamComp('wind_direction', wd), promotes=['*'])
        pb.root.add('a', my_silly_func(), promotes=['*'])
        pb.root.add('eq', ExecComp('y = power*2.0'), promotes=['*'])
        pb.setup()
        pb.run()
        result = (((ws1 + ws2)/2.0)**3.0 * sin(wd * pi/180.0)) * 2.0
        self.assertAlmostEqual(pb.root.unknowns['y'], result)


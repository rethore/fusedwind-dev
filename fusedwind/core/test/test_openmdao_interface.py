import numpy as np
from fusedwind.core.openmdao_interface import fused_func, Inputs, Outputs, WindSpeed, WindDirection, Power


__author__ = 'pire'

import unittest


class TestFused_func(unittest.TestCase):

    def test_one_output(self):

        @fused_func(
            Inputs(
                ws = WindSpeed(4.0),
                wd = WindDirection(270.0)),
            Outputs(
                p = Power))
        def my_silly_func(u, d):
            return u**3.0 * np.sin(d * np.pi/180.)

        a = my_silly_func(3.0, 0.0)
        self.assertAlmostEqual(a(5.0, 90.0), 5.0**3.0)

    def test_several_outputs(self):

        @fused_func(
            Inputs(
                ws = WindSpeed(4.0),
                wd = WindDirection(270.0)),
            Outputs(
                p1 = Power,
                p2 = Power,
                p3 = Power))
        def my_silly_func(u, d):
            return [i * u**3.0 * np.sin(d * np.pi/180.) for i in range(3)]

        a = my_silly_func(3.0, 0.0)
        for i in range(3):
            self.assertAlmostEqual(a(5.0, 90.0)[i], i*5.0**3.0)

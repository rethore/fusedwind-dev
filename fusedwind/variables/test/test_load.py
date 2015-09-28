
__author__ = 'pire'
import unittest

class TestLoadInit(unittest.TestCase):
    def test_import(self):
        from fusedwind.variables import wind_speed
        print(wind_speed)
        print(wind_speed.desc)
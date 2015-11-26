
import numpy as np
import copy
import unittest

from fusedwind.turbine.layup import BladeLayup, create_bladestructure


def configure():
    
    bl = BladeLayup()
    
    biax = bl.add_material('biax')
    
    biax.set_props(E1 = 13.92e9,
                     E2 = 13.92e9,
                     E3 = 13.92e9,
                     nu12 = 0.533,
                     nu13 = 0.533,
                     nu23 = 0.533,
                     G12 = 11.5e9,
                     G13 = 4.539e9,
                     G23 = 4.539e9,
                     rho = 1845)
    
    biax.set_resists_strains(failcrit = 1,
                             e11_t = 9.52E-03,
                             e22_t = 1.00E+06,
                             e33_t = 1.00E+06,
                             e11_c = 6.80E-03,
                             e22_c = 1.00E+06,
                             e33_c = 1.00E+06,
                             g12 = 1.00E+06,
                             g13 = 1.00E+06,
                             g23 = 1.00E+06)
    
    biax.set_safety_GL2010(gM0 = 1.25,
                           C1a = 1.0,
                           C2a = 1.0,
                           C3a = 1.0,
                           C4a = 1.0)

    uniax = bl.add_material('uniax')
    uniax.E1 = 41.63e9
    uniax.E2 = 14.93e9
    uniax.E3 = 14.93e9
    uniax.nu12 = 0.241
    uniax.nu13 = 0.241
    uniax.nu23 = 0.241
    uniax.G12 = 5.047e9
    uniax.G13 = 5.047e9
    uniax.G23 = 5.047e9
    uniax.rho = 1915.5
    
    uniax.set_resists_strains(failcrit = 1,
                             e11_t = 9.52E-03,
                             e22_t = 1.00E+06,
                             e33_t = 1.00E+06,
                             e11_c = 6.80E-03,
                             e22_c = 1.00E+06,
                             e33_c = 1.00E+06,
                             g12 = 1.00E+06,
                             g13 = 1.00E+06,
                             g23 = 1.00E+06)
    
    uniax.set_safety_GL2010(gM0 = 1.25,
                           C1a = 1.0,
                           C2a = 1.0,
                           C3a = 1.0,
                           C4a = 1.0)

    core = bl.add_material('core')
    core.E1 = 50e6
    core.E2 = 50e6
    core.E3 = 50e6
    core.nu12 = 0.5
    core.nu13 = 0.013
    core.nu23 = 0.013
    core.G12 = 16.67e6
    core.G13 = 150e6
    core.G23 = 150e6
    core.rho = 110

    core.set_resists_strains(failcrit = 1,
                             e11_t = 9.52E-03,
                             e22_t = 1.00E+06,
                             e33_t = 1.00E+06,
                             e11_c = 6.80E-03,
                             e22_c = 1.00E+06,
                             e33_c = 1.00E+06,
                             g12 = 1.00E+06,
                             g13 = 1.00E+06,
                             g23 = 1.00E+06)
    
    core.set_safety_GL2010(gM0 = 1.25,
                           C1a = 1.0,
                           C2a = 1.0,
                           C3a = 1.0,
                           C4a = 1.0)
    
    bl.s = [0, 0.25, 0.6, 1.]
    
    bl.init_regions(5)
    
    bl.DPs['DP00'].arc = np.ones(4) * -1.
    bl.DPs['DP05'].arc = np.ones(4) * 1.
    bl.DPs['DP01'].arc = np.ones(4) * -0.5 
    bl.DPs['DP04'].arc = np.ones(4) * 0.5
    bl.DPs['DP02'].arc = np.ones(4) * -0.35 
    bl.DPs['DP03'].arc = np.ones(4) * 0.3
    
    # add materials to regions
    r = bl.regions['region00']
    l = r.add_layer('triax')
    l.thickness = np.array([0.008, 0.003, 0.002, 0.001])
    l.angle = np.zeros(4)
    l = r.add_layer('uniax')
    l.thickness = np.array([0.008, 0.000, 0.000, 0.000])
    l.angle = np.zeros(4)
    l = r.add_layer('core')
    l.thickness = np.array([0.00, 0.07, 0.06, 0.000])
    l.angle = np.zeros(4)
    l = r.add_layer('uniax')
    l.thickness = np.array([0.008, 0.000, 0.000, 0.000])
    l.angle = np.zeros(4)
    l = r.add_layer('triax')
    l.thickness = np.array([0.008, 0.003, 0.002, 0.001])
    l.angle = np.zeros(4)
    bl.regions['region04'] = copy.copy(r)

    r = bl.regions['region01']
    l = r.add_layer('triax')
    l.thickness = np.array([0.008, 0.000, 0.000, 0.000])
    l.angle = np.zeros(4)
    l = r.add_layer('uniax')
    l.thickness = np.array([0.008, 0.04, 0.04, 0.002])
    l.angle = np.zeros(4)
    l = r.add_layer('uniax')
    l.thickness = np.array([0.008, 0.04, 0.04, 0.002])
    l.angle = np.zeros(4)
    l = r.add_layer('triax')
    l.thickness = np.array([0.008, 0.000, 0.099, 0.000])
    l.angle = np.zeros(4)
    bl.regions['region03'] = copy.copy(r)
    
    r = bl.regions['region02']
    l = r.add_layer('triax')
    l.thickness = np.array([0.008, 0.003, 0.0015, 0.0011])
    l.angle = np.zeros(4)
    l = r.add_layer('uniax')
    l.thickness = np.array([0.008, 0.001, 0.0007, 0.00])
    l.angle = np.zeros(4)
    l = r.add_layer('core')
    l.thickness = np.array([0.00, 0.035, 0.02, 0.000])
    l.angle = np.zeros(4)
    l = r.add_layer('uniax')
    l.thickness = np.array([0.008, 0.001, 0.0007, 0.00])
    l.angle = np.zeros(4)
    l = r.add_layer('triax')
    l.thickness = np.array([0.008, 0.003, 0.0015, 0.0011])
    l.angle = np.zeros(4)
    
    bl.init_webs(2, [[2, 3], [1, 4]])
    w = bl.webs['web00']
    l = w.add_layer('biax')
    l.thickness = np.array([0.0025, 0.0045, 0.004, 0.001])
    l.angle = np.zeros(4)
    l = w.add_layer('core')
    l.thickness = np.array([0.065, 0.05, 0.02, 0.005])
    l.angle = np.zeros(4)
    l = w.add_layer('biax')
    l.thickness = np.array([0.0025, 0.0045, 0.004, 0.001])
    l.angle = np.zeros(4)
    bl.webs['web01'] = copy.copy(w)
    
    return bl, uniax

class LayupTests(unittest.TestCase):
    ''' This class contains the unit tests for
        :mod:`fusedwind.turbine.layup`.
    '''
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.bl, self.uniax = configure()
        self.st3d = create_bladestructure(self.bl)
        
    def test_create_bladestructure_division_points(self):
        self.assertEqual(np.testing.assert_array_equal(
                         self.st3d['DPs'][:,4],
                         self.bl.DPs['DP04'].arc), None)
        
    def test_create_bladestructure_thicknesses(self):
        self.assertEqual(np.testing.assert_array_equal(
                         self.st3d['regions'][2]['thicknesses'][:,4],
                         self.bl.regions['region02'].layers['triax01'].thickness), None)
        
    def test_create_bladestructure_materials(self):
        uniax = self.uniax
        self.assertEqual(np.testing.assert_array_equal(
                         self.st3d['matprops'][1,:],
                         [uniax.E1,
                          uniax.E2,
                          uniax.E3,
                          uniax.nu12,
                          uniax.nu13,
                          uniax.nu23,
                          uniax.G12,
                          uniax.G13,
                          uniax.G23,
                          uniax.rho,
                          ]), None)
        
if __name__ == '__main__':
    #configure()
    unittest.main()
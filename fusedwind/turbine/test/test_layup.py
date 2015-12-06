
import numpy as np
import copy
import unittest

from fusedwind.turbine.layup import BladeLayup, create_bladestructure
from fusedwind.turbine.structure import write_bladestructure,\
    read_bladestructure
import os
import shutil
import collections


def configure():
    
    bl = BladeLayup()
    
    biax = bl.add_material('triax')
    
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
    
    biax.set_resists_strains(failcrit = 'maximum_strain',
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
    uniax.set_props(E1 = 41.63e9,
                    E2 = 14.93e9,
                    E3 = 14.93e9,
                    nu12 = 0.241,
                    nu13 = 0.241,
                    nu23 = 0.241,
                    G12 = 5.047e9,
                    G13 = 5.047e9,
                    G23 = 5.047e9,
                    rho = 1915.5)
    
    
    uniax.set_resists_strains(failcrit = 'maximum_strain',
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
    core.set_props(E1 = 50e6,
                   E2 = 50e6,
                   E3 = 50e6,
                   nu12 = 0.5,
                   nu13 = 0.013,
                   nu23 = 0.013,
                   G12 = 16.67e6,
                   G13 = 150e6,
                   G23 = 150e6,
                   rho = 110)


    core.set_resists_strains(failcrit = 'maximum_strain',
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
    bl.DPs['DP01'].arc[0] = -0.49
    bl.DPs['DP01'].arc[-1] = -0.51
    bl.DPs['DP04'].arc = np.ones(4) * 0.5
    bl.DPs['DP04'].arc[0] = 0.49
    bl.DPs['DP04'].arc[-1] = 0.51
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
    
    bl.init_webs(2, [[2, 3], [1, 4]], ['top', 'mid'])
    w = bl.webs['web00']
    l = w.add_layer('triax')
    l.thickness = np.array([0.0025, 0.0045, 0.004, 0.001])
    l.angle = np.zeros(4)
    l = w.add_layer('core')
    l.thickness = np.array([0.065, 0.05, 0.02, 0.005])
    l.angle = np.zeros(4)
    l = w.add_layer('triax')
    l.thickness = np.array([0.0025, 0.0045, 0.004, 0.001])
    l.angle = np.zeros(4)
    bl.webs['web01'] = copy.copy(w)
    
    bl.check_consistency()
    
    return bl, uniax

def configure_incorrect():

    bl = BladeLayup()
    
    biax = bl.add_material('triax')
    
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
    
    biax.set_resists_strains(failcrit = 'maximum_strain',
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
    
    # incorrect setting of uniax properties
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
    
    uniax.set_resists_strains(failcrit = 'maximum_strain',
                             e11_t = 9.52E-03,
                             e22_t = 1.00E+06,
                             e33_t = 1.00E+06,
                             e11_c = 6.80E-03,
                             e22_c = 1.00E+06,
                             e33_c = 1.00E+06,
                             g12 = 1.00E+06,
                             g13 = 1.00E+06,
                             g23 = 1.00E+06)
    
    # missing safety factors
    #--------------------------------------- uniax.set_safety_GL2010(gM0 = 1.25,
    #----------------------------------------- C1a = 1.0,
    #----------------------------------------- C2a = 1.0,
    #----------------------------------------- C3a = 1.0,
    #----------------------------------------- C4a = 1.0)

    core = bl.add_material('core')
    core.set_props(E1 = 50e6,
                   E2 = 50e6,
                   E3 = 50e6,
                   nu12 = 0.5,
                   nu13 = 0.013,
                   nu23 = 0.013,
                   G12 = 16.67e6,
                   G13 = 150e6,
                   G23 = 150e6,
                   rho = 110)

    core.set_resists_strains(failcrit = 'maximum_strain',
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
    # incorrect DP length
    bl.DPs['DP01'].arc = np.ones(1) * -0.5 
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
    # non-existing material in region00 and region04
    l = r.add_layer('biax')
    l.thickness = np.array([0.008, 0.003, 0.002, 0.001])
    l.angle = np.zeros(4)
    bl.regions['region04'] = copy.copy(r)

    r = bl.regions['region01']
    l = r.add_layer('triax')
    l.thickness = np.array([0.008, 0.000, 0.000, 0.000])
    l.angle = np.zeros(4)
    l = r.add_layer('uniax')
    # incorrect thickness and angle lengths
    l.thickness = np.array([0.008, 0.04]) #, 0.04, 0.002
    l.angle = np.zeros(3)
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
    
    bl.init_webs(2, [[2, 3], [1, 4]], ['top', 'mid'])
    w = bl.webs['web00']
    l = w.add_layer('triax')
    l.thickness = np.array([0.0025, 0.0045, 0.004, 0.001])
    l.angle = np.zeros(4)
    l = w.add_layer('core')
    l.thickness = np.array([0.065, 0.05, 0.02, 0.005])
    l.angle = np.zeros(4)
    l = w.add_layer('triax')
    l.thickness = np.array([0.0025, 0.0045, 0.004, 0.001])
    l.angle = np.zeros(4)
    bl.webs['web01'] = copy.copy(w)
    
    bl.check_consistency()
    
    return bl
    pass
    

class LayupTests(unittest.TestCase):
    ''' This class contains the unit tests for
        :mod:`fusedwind.turbine.layup`.
    '''
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.test_dir = 'test_dir'
        self.bl, self.uniax = configure()
        self.st3d = create_bladestructure(self.bl)
    
    def tearDown(self):
        unittest.TestCase.tearDown(self)
    
    def test_check_consistency(self):
        self.assertEqual(self.bl._warns, 0, None)
    
    def test_check_consistency_incorrect(self):
        self.bl_inc = configure_incorrect()
        self.assertEqual(self.bl_inc._warns, 15, None)
        
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
        
    def test_create_write_read_bladestructure(self):
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        write_bladestructure(self.st3d, os.path.join(self.test_dir,'test'))
        st3dn = read_bladestructure(os.path.join(self.test_dir,'test'))
        self.assertEqual(np.testing.assert_array_equal(
                         self.st3d['DPs'][:,4],
                         st3dn['DPs'][:,4]), None)
        self.assertEqual(np.testing.assert_array_equal(
                         self.st3d['regions'][2]['thicknesses'][:,4],
                         st3dn['regions'][2]['thicknesses'][:,4]), None)
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
        shutil.rmtree(self.test_dir)
        
if __name__ == '__main__':
    #configure()
    unittest.main()
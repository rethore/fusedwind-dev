
import numpy as np
import unittest

from fusedwind.turbine.structure import write_bladestructure,\
    read_bladestructure
import os
import shutil

st3d_desired = {}
st3d_desired['web_def'] = np.array([[-1, 0], [2, -3], [4, -5], [5, -6]])
st3d_desired['bond_def'] = np.array([[0, 1, -2, -1]])

class StructureTests(unittest.TestCase):
    ''' This class contains the unit tests for
        :mod:`fusedwind.turbine.structure`.
    '''
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.data_version_0 = 'data'
        self.data_version_1 = 'data_version_1'
        self.data_version_2 = 'data_version_2'
        self.blade = 'DTU10MW'
        self.test_dir = 'test_dir'
    
    def tearDown(self):
        unittest.TestCase.tearDown(self)
        
    def test_read_bladestructure_version_0(self):
        st3d_actual = read_bladestructure(os.path.join(self.data_version_0, self.blade))
        self.assertEqual(np.testing.assert_array_equal(
                         st3d_actual['web_def'],
                         st3d_desired['web_def']), None)
    
    def test_read_bladestructure_version_1(self):
        st3d_actual = read_bladestructure(os.path.join(self.data_version_1, self.blade))
        self.assertEqual(np.testing.assert_array_equal(
                         st3d_actual['web_def'],
                         st3d_desired['web_def']), None)
        
    def test_read_write_read_bladestructure_version_0(self):
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        st3d = read_bladestructure(os.path.join(self.data_version_0, self.blade))
        # convert to version 1
        st3d['version'] = 1
        write_bladestructure(st3d, os.path.join(self.test_dir, 'test'))
        st3dn = read_bladestructure(os.path.join(self.test_dir, 'test'))
        self.assertEqual(np.testing.assert_array_equal(
                         st3dn['web_def'],
                         st3d_desired['web_def']), None)
        shutil.rmtree(self.test_dir)
        
    def test_read_write_read_bladestructure_version_1(self):
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        st3d = read_bladestructure(os.path.join(self.data_version_1, self.blade))
        write_bladestructure(st3d, os.path.join(self.test_dir, 'test'))
        st3dn = read_bladestructure(os.path.join(self.test_dir, 'test'))
        self.assertEqual(np.testing.assert_array_equal(
                         st3dn['web_def'],
                         st3d_desired['web_def']), None)
        shutil.rmtree(self.test_dir)
        
    def test_read_write_read_bladestructure_version_2(self):
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        st3d = read_bladestructure(os.path.join(self.data_version_2, self.blade))
        write_bladestructure(st3d, os.path.join(self.test_dir, 'test'))
        st3dn = read_bladestructure(os.path.join(self.test_dir, 'test'))
        self.assertEqual(np.testing.assert_array_equal(
                         st3dn['web_def'],
                         st3d_desired['web_def']), None)
        self.assertEqual(np.testing.assert_array_equal(
                         st3dn['bond_def'],
                         st3d_desired['bond_def']), None)
        shutil.rmtree(self.test_dir)
        
if __name__ == '__main__':
    unittest.main()
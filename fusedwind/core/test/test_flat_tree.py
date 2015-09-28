from collections import OrderedDict
import unittest
from fusedwind.core.flat_tree import FlatTree, branch

__author__ = 'pire'


class TestFlatTree(unittest.TestCase):

    _test_dics_ = [
        [
            ('base', 1.),
            ('kat:kat', 'leopold'),
            ('kat:lapin', 'bernard'),
        ]
    ]

    def _run_test(self, dic):
        tree = FlatTree(dic)
        tree.base = 2.
        self.assertEqual(dic['base'], 2.)
        tree.kat.kat = 'albert'
        self.assertEqual(dic['kat:kat'], 'albert')

    def test_dict(self):
        for dic in self._test_dics_:
            self._run_test(dic)


    def test_oredereddict(self):
        for dic in self._test_dics_:
            self._run_test(OrderedDict(dic))


class BranchTestCase(unittest.TestCase):
    def setUp(self):
        self.b = branch(
            {'kat': {'val':2.,
                    'type':'float'
                    },
            'dog': {'type':'vartree',
                    'val': {
                        'giraf': {
                            'type': 'float',
                            'val': 10.
                        },
                        'mouse': {
                            'type': 'float',
                            'val': 50.
                        },}}})
    def test_setattr(self):
        self.b.kat = 3.0
        self.b.dog.giraf = 3.0
        self.assertAlmostEqual(self.b._flat_dic['kat'], 3.0)
        self.assertAlmostEqual(self.b._flat_dic['dog:giraf'], 3.0)

    def test_bad_setattr(self):
        with self.assertRaises(KeyError) as context:
            self.b.dgo = 3.
        self.assertTrue('dgo is not permitted' in str(context.exception))

        with self.assertRaises(KeyError) as context:
            self.b.dog.lapin = 3.
        self.assertTrue('lapin is not permitted' in str(context.exception))


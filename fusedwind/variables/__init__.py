__author__ = 'pire'
import os
import yaml
from path import Path
from fusedwind.core.fused_variable import FUSEDVar

fall = {}
# Loop on all the yaml files and add the base keys as local dictionaries
for f in Path(os.path.dirname(os.path.realpath(__file__))).files('*yaml'):
    dic = yaml.load(f.text())
    fil = os.path.basename(f.split('.')[0])
    fall[fil] = {}
    for k, v in dic.items():
        v['name'] = k
        exec('%s=FUSEDVar(**%s)'%(k, repr(v)))
        fall[fil][k] = FUSEDVar(**v)

    #globals().update(dic)


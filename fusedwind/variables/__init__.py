__author__ = 'pire'
import os
import yaml
from path import Path
from fusedwind.core.fused_variable import FUSEDVar

# Loop on all the yaml files and add the base keys as local dictionaries
for f in Path(os.path.dirname(os.path.realpath(__file__))).files('*yaml'):
    dic = yaml.load(f.text())
    for k, v in dic.items():
        exec('%s=FUSEDVar(**%s)'%(k, repr(v)))
    #globals().update(dic)


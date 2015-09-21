
from openmdao.core import Group

import fusedwind.models as models


class FUSEDModelBuilder(Group):

    def __init__(self, config):
        super(FUSEDSimpleTurbineDesign, self).__init__()

        for name, dic in config.iteritems():
            klass = getattr(models, dic['model'])
            try:
                args = dic['args']
            except:
                args = {}
            print 'adding model %s %s' % (name, dic['model'])
            self.add(name, klass(**args), promotes=['*'])
            if 'deps' in dic:
                for dname, ddic in dic['deps'].iteritems():
                    if not hasattr(self, dname):
                        kklass = getattr(models, ddic['model'])
                        try:
                            args = ddic['args']
                        except:
                            args = {}
                        self.add(dname, kklass(**args), promotes=['*'])

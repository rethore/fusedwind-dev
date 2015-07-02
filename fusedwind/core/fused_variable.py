from numpy import zeros, array

__author__ = 'pire'

class FUSEDVar(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, default=None, **kwargs):
        super(FUSEDVar, self).__init__(kwargs)
        if default is None:
            if 'default' in kwargs:
                default = kwargs['default']
            elif 'low' in kwargs:
                default = kwargs['low']
            elif kwargs['type']=='float':
                default = 0.0
            elif kwargs['type']=='array':
                if 'size' in kwargs:
                    if isinstance(kwargs['size'], str):
                        default = array([])
                    else:
                        default = zeros(kwargs['size'])
                else:
                    default = array([])
            else:
                raise Exception('Default value should be given as input')

        self.__setitem__('val', default)
        self.__setitem__('default', default)

    def __call__(self, val=None, **kwargs):
        self.__setitem__('val', val)
        for k, v in kwargs.items():
            self.__setitem__(k, v)
        return self


Float = lambda *args, **kwargs: FUSEDVar(*args, type='float', **kwargs)
Array = lambda *args, **kwargs: FUSEDVar(*args, type='array', **kwargs)
Str = lambda *args, **kwargs: FUSEDVar(*args, type='str', **kwargs)

class VarTree(object):
    def __init__(self, vartree, **kwargs):
        self.vartree = vartree()
        self.kwargs = kwargs

    def items(self):
        out = dict(self.vartree.items())
        out.update(self.kwargs)
        return out.items()

class VariableTree(object):
    def items(self):
        return [('type', 'vartree'),
                ('val', dict(self._items()))]

    def _items(self):
        for k in dir(self):
            if not k.startswith('_'):
                obj = getattr(self, k)
                if isinstance(obj, FUSEDVar):
                    yield (k, obj)
                elif isinstance(obj, VarTree):
                    yield (k, dict(obj.items()))




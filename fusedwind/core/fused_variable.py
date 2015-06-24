from numpy import zeros, array

__author__ = 'pire'

class FUSEDVar(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, default=None, **kwargs):
        super().__init__(kwargs)
        if default is None:
            if 'default' in kwargs:
                default = kwargs['default']
            elif 'low' in kwargs:
                default = kwargs['low']
            elif kwargs['type']=='float':
                default = 0.0
            elif kwargs['type']=='array':
                if isinstance(kwargs['size'], str):
                    default = array([])
                else:
                    default = zeros(kwargs['size'])
            else:
                raise Exception('Default value should be given as input')

        self.__setitem__('val', default)
        self.__setitem__('default', default)

    def __call__(self, val=None, **kwargs):
        self.__setitem__('val', val)
        for k, v in kwargs.items():
            self.__setitem__(k, v)
        return self

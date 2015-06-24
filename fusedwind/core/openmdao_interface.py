__author__ = 'pire'

from collections import OrderedDict
import numpy as np
from openmdao.core.component import Component

Inputs = OrderedDict
Outputs = OrderedDict

class FUSEDVar(dict):
    def __init__(self, default, **kwargs):
        super().__init__(kwargs)
        self.__setitem__('value', default)
        self.__setitem__('default', default)

    def __call__(self, value, **kwargs):
        self.__setitem__('value', value)
        for k, v in kwargs.items():
            self.__setitem__(k, v)
        return self

WindSpeed = FUSEDVar(10.0, units='m/s')
WindDirection = FUSEDVar(0.0, units='deg')
Power = FUSEDVar(0.0, units='W')

def fused_func(inputs=Inputs(), outputs=Outputs()):
    def _outer_wrapper(wrapped_function):
        ## Create the component that wrapp the function
        class _C(Component):
            def __init__(self):
                super().__init__()
                for k, v in inputs.items():
                    self.add_param(k, v['value'])

                for k, v in outputs.items():
                    self.add_output(k, v['value'])

            def solve_nonlinear(self, params, unknowns, resids):
                results = wrapped_function(*[params[k] for k in inputs.keys()])
                if not isinstance(results, list):
                    unknowns[list(outputs.keys())[0]] = results
                else:
                    for k, v in zip(outputs.keys(), results):
                        unknowns[k] = v

            def __call__(self, *args, **kwargs):
                params = self._params_dict
                unknowns = self._unknowns_dict
                resids = OrderedDict()
                input_keys = list(params.keys())
                for i, a in enumerate(args):
                    k = input_keys[i]
                    params[k] = a
                for k,v in kwargs.items():
                    params[k] = v
                self.solve_nonlinear(params, unknowns, resids)

                values = list(unknowns.values())
                if len(values) > 1:
                    return [v for v in unknowns.values()]
                else:
                    return values[0]


        _C.__name__ = 'F'+ wrapped_function.__name__

        ## Now returns a function that initialize the component
        def _initialize(*args, **kwargs):
            c = _C()
            input_keys = list(c._params_dict.keys())
            for i, a in enumerate(args):
                k = input_keys[i]
                c._params_dict[k] = a
            for k,v in kwargs.items():
                c._params_dict[k] = v
            return c

        return _initialize
    return _outer_wrapper


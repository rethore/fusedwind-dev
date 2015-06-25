__author__ = 'pire'

from collections import OrderedDict
import numpy as np
from openmdao.core.component import Component

Inputs = OrderedDict
Outputs = OrderedDict


class FUSEDComponent(Component):
    def __init__(self, inputs, outputs, wrapped_function):
        super(FUSEDComponent, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.func = wrapped_function
        self.ref_inputs = []
        self.ref_outputs = []

        # Create the inputs
        for i in self.inputs:
            if isinstance(i, tuple):
                name, v = i
            else:
                v = i
                name = v['name']

            self.ref_inputs.append(name)
            self.add_param(name, v['val'])

        # Create the outputs
        for o in self.outputs:
            if isinstance(o, tuple):
                name, v = o
            else:
                v = o
                name = v['name']
            self.ref_outputs.append(name)
            self.add_output(name, v['val'])


    def params2inputs(self, params):
        """Convert the openmdao  `params` pseudo dict into a fusedwind compatible dict"""
        return [params[k] for k in self.ref_inputs]

    def outputs2unknowns(self, fused_outputs):
        """Convert the a fusedwind compatible output list into an `unknowns` pseudo dict"""
        if len(self.ref_outputs) == 1:
            # If there is only one output, then we wrap it inside a function
            return {self.ref_outputs[0]: fused_outputs}
        else:
            return dict(zip(self.ref_outputs, fused_outputs))

    def solve_nonlinear(self, params, unknowns, resids):
        new_unknowns = self.outputs2unknowns(self.func(*self.params2inputs(params)))
        for k,v in new_unknowns.items():
            unknowns[k] = v


    def __call__(self, *args, **kwargs):
        """Call the component in a pythonic way:
        [output_1, ..., output_n] = func(input_1, ..., input_n)
        If only part of the inputs are specified, the component will use the default values or the previously
        defined values.

        :param args:    list
                        inputs defined by the wrapper. The order corresponds to the order indicated in the
                        wrapper decorator
        :param kwargs:  dict
                        inputs defined by the wrapper.
        :return:
        """
        params = self._params_dict
        unknowns = self._unknowns_dict
        resids = OrderedDict()
        input_keys = list(params.keys())

        # TODO: verify that the args are not replaced by the kwargs
        if len(args)>0 and len(kwargs)>0:
            raise NotImplementedError('Only use args or kwargs independently, not both at the same time.')

        for i, a in enumerate(args):
            k = input_keys[i]
            params[k] = a
        for k,v in kwargs.items():
            params[k] = v
        self.solve_nonlinear(params, unknowns, resids)

        values = list(unknowns.values())
        if len(values) > 1:
            return [unknowns[k] for k in self.ref_outputs]
        else:
            return values[0]


def fused_func(inputs=Inputs(), outputs=Outputs()):
    def _outer_wrapper(wrapped_function):
        ## Now returns a function that initialize the component
        def _initialize(*args, **kwargs):
            c = FUSEDComponent(inputs, outputs, wrapped_function)
            input_keys = list(c._params_dict.keys())
            for i, a in enumerate(args):
                k = input_keys[i]
                c._params_dict[k] = a
            for k,v in kwargs.items():
                c._params_dict[k] = v
            return c

        return _initialize
    return _outer_wrapper


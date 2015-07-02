from copy import copy
from fusedwind.core.flat_tree import branch, flatten
from fusedwind.core.fused_variable import FUSEDVar
from fusedwind.core.fused_variable import VarTree
from fusedwind.variables import fall
import yaml

__author__ = 'pire'

from collections import OrderedDict
import numpy as np
from openmdao.core.component import Component

Inputs = OrderedDict
Outputs = OrderedDict


class MetaFUSEDWind(type):
    """
    Allows backward compatibility for old openMDAO style components. The class will basically translate the class variables
    into a nested dictionary stored in two places: _inputs and _outputs.
    """
    def __new__(metaname, classname, baseclasses, attrs):
        attrs['_inputs'] = {}
        attrs['_outputs'] = {}
        keys = list(attrs.keys())
        for k in keys:
            if not k[0] == '_':
                obj = attrs[k]
                print(k, obj.__class__.__name__)
                if isinstance(obj, FUSEDVar):
                    if obj['iotype'] == 'in':
                        attrs['_inputs'][k] = obj
                    if obj['iotype'] == 'out':
                        attrs['_outputs'][k] = obj
                    attrs.pop(k)
                if isinstance(obj, VarTree):
                    if obj.kwargs['iotype'] == 'in':
                        attrs['_inputs'][k] = dict(obj.items())
                    if obj.kwargs['iotype'] == 'out':
                        attrs['_outputs'][k] = dict(obj.items())
                    attrs.pop(k)

        return type.__new__(metaname, classname, baseclasses, attrs)

class FUSEDComponent(Component, metaclass=MetaFUSEDWind):

    def __init__(self, *args, **kwargs):
        super(FUSEDComponent, self).__init__()

        if not hasattr(self, '_inputs'):
            self._inputs = {}
        if not hasattr(self, '_outputs'):
            self._outputs = {}

        # Where the flat_dictionaries of OpenMDAO are going to be stored
        self._fused_params = {}
        self._fused_unknowns = {}

        # Create the inputs
        for k, v in self._inputs.items():
            if v['type'] == 'vartree':
                setattr(self, k, branch(v['val'], self._fused_params, k))
            for k2, v2 in flatten(k,v):
                self.add_param(k2, v2)
                self._fused_params[k2] = v2

        # Create the outputs
        for k, v in self._outputs.items():
            if v['type'] == 'vartree':
                setattr(self, k, branch(v['val'], self._fused_unknowns, k))
            for k2, v2 in flatten(k,v):
                self.add_output(k2, v2)
                self._fused_unknowns[k2] = v2

    def solve_nonlinear(self, params, unknowns, resids):
        self._fused_params = params
        self._fused_unknowns = unknowns
        self.execute()


    def execute(self):
        pass

    def __getattr__(self, item):
        if item.startswith('_'):
            super(FUSEDComponent, self).__getattribute__(item)
        else:
            if hasattr(self, '_fused_unknowns') and hasattr(self, '_fused_params'):
                if item in self._fused_params:
                    return self._fused_params[item]
                elif item in self._fused_unknowns:
                    return self._fused_unknowns[item]
                else:
                    self.__getattribute__(item)
            else:
                super(FUSEDComponent, self).__getattribute__(item)

    def __setattr__(self, item, value):
        if item.startswith('_'):
            return super(FUSEDComponent, self).__setattr__(item, value)
        if hasattr(self, '_fused_unknowns') and hasattr(self, '_fused_params'):
            if item in self._fused_unknowns:
                if item in dir(self):
                    raise KeyError(item + 'is both defined in unkowns and as an internal instance variable. Your variables are going to collite')
                self._fused_unknowns[item] = value
            else:
                if item in self._fused_params:
                    self._fused_params[item] = value
#                if item in self._fused_params:
#                   raise KeyError(item + ' is defined as an input and cant be changed')
                return super(FUSEDComponent, self).__setattr__(item, value)
                # raise KeyError(item + 'is not an unknown or a , and therefore can\'t be changed')
        else:
            return super(FUSEDComponent, self).__setattr__(item, value)



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
        params = self._fused_params
        unknowns = self._fused_unknowns
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


class WrappedFuncComponent(Component):
    def __init__(self, inputs, outputs, wrapped_function):
        super(WrappedFuncComponent, self).__init__()
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
        params = copy(self._params_dict)
        unknowns = copy(self._unknowns_dict)
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
            c = WrappedFuncComponent(inputs, outputs, wrapped_function)
            input_keys = list(c._params_dict.keys())
            for i, a in enumerate(args):
                k = input_keys[i]
                c._params_dict[k] = a
            for k,v in kwargs.items():
                c._params_dict[k] = v
            return c

        return _initialize
    return _outer_wrapper

def fused_yaml(io, func):
    iod = yaml.load(io)
    inputs_str = iod[func.__name__]['inputs']
    outputs_str= iod[func.__name__]['outputs']
    inputs = []
    for i in inputs_str:
        d, o = i.split('.')
        inputs.append(fall[d][o])

    outputs = []
    for i in outputs_str:
        d, o = i.split('.')
        outputs.append(fall[d][o])

    return fused_func(inputs, outputs)(func)
from openmdao.api import IndepVarComp, Component, Problem, Group
from fusedwind.core.FUSEDobj import FUSEDobj
from functools import reduce

class FUSEDBasedComponent(Component):
    def _add_inputs(self, inputs):
        for a in inputs:
            self.add_param(a.address, val=a.value)

    def _add_outputs(self, outputs):
        for a in outputs:
            self.add_output(a.address, val=a.value)


class FUSEDComponent(Component):
    def __init__(self):
        super(FUSEDComponent, self).__init__()
        self.fused_objects = {}
        self.inverse_map = {}
        #self.auto_configure()

    def __setattr__(self, key, value):
        if hasattr(self, 'fused_objects'):
            if key in self.fused_objects:
                self.fused_objects[key].update_value(value)
            else:
                if isinstance(value, FUSEDobj):
                    super(FUSEDComponent, self).__setattr__(key, value)
                    value.inputs = set()
                    value.outputs = set()
                    self.fused_objects[key] = value
                    if value._type in ('list(dict)', 'dict'):
                        for k in value.resolves('*'):
                            self.inverse_map[k.address] = k
                    else:
                        self.inverse_map[value.address] = value
                else:
                    super(FUSEDComponent, self).__setattr__(key, value)
        else:
            super(FUSEDComponent, self).__setattr__(key, value)

    def auto_configure(self):
        for v in self.fused_objects.values():
            v.outputs = set()
            v.inputs = set()

        self._execute()

        outputs = reduce(lambda a,b:a.union(b), [v.outputs for k, v in self.fused_objects.items()])
        inputs = reduce(lambda a,b:a.union(b), [v.inputs for k, v in self.fused_objects.items()]) - outputs

        for i in inputs:
            if i in self.inverse_map:
                self.add_param(i, val=self.inverse_map[i].value)
                print('input:', i, self.inverse_map[i].value)

        for o in outputs:
            if o in self.inverse_map:
                self.add_output(o, val=self.inverse_map[o].value)
                print('output:', o, self.inverse_map[o].value)

    def _execute(self):
        raise Exception("_execute methode hasn't been implemented")

    def solve_nonlinear(self, params, unknowns, resids):
        for k, v in params.items():
            self.inverse_map[k].value = v['val']

        self._execute()

        for k in unknowns.keys():
            unknowns[k] = self.inverse_map[k].value

class FUSEDWrapper(FUSEDComponent):
    def __init__(self, func, inputs, outputs):
        super(FUSEDWrapper, self).__init__()
        self.func = func
        self._inputs = inputs
        self._outputs = outputs
        self.auto_configure()

    def auto_configure(self):
        for v in self._inputs:
            if v._type in ('list(dict)', 'dict'):
                for k in v.resolves('*'):
                    self.inverse_map[k.address] = k
            else:
                self.inverse_map[v.address] = v

            v.inputs = set()

        for v in self._outputs:
            if v._type in ('list(dict)', 'dict'):
                for k in v.resolves('*'):
                    self.inverse_map[k.address] = k
            else:
                self.inverse_map[v.address] = v

            v.outputs = set()

        outs = self.func(*self._inputs)

        ## Not needed to figure out what are the outputs
        #if not isinstance(outs, list):
        #    outs = [outs]
        #for o1, o2 in zip(self._outputs, outs):
        #    o1.update_value(o2)

        outputs = reduce(lambda a,b:a.union(b), [v.outputs for v in self._outputs])
        inputs = reduce(lambda a,b:a.union(b), [v.inputs for v in self._inputs]) - outputs

        for i in inputs:
            if i in self.inverse_map:
                self.add_param(i, val=self.inverse_map[i].value)
                print('input:', i, self.inverse_map[i].value)

        for o in outputs:
            if o in self.inverse_map:
                self.add_output(o, val=self.inverse_map[o].value)
                print('output:', o, self.inverse_map[o].value)


    def params2inputs(self, params):
        """Convert the openmdao  `params` pseudo dict into a fusedwind compatible dict"""
        for k in self._inputs:
            for i in k.resolves('*'):
                if i.address in params:
                    i.value = params[i.address]
            yield k

    def outputs2unknowns(self, fused_outputs):
        """Convert the a fusedwind compatible output list into an `unknowns` pseudo dict"""
        if len(self._outputs) == 1:
            # If there is only one output, then we wrap it inside a function
            yield self._outputs[0], fused_outputs
        else:
            for k, v in zip(self._outputs, fused_outputs):
                yield k, v

    def solve_nonlinear(self, params, unknowns, resids):
        for k, v in self.outputs2unknowns(self.func(*list(self.params2inputs(params)))):
            for i in k.resolves('*'):
                if i.address in self.unknowns:
                    unknowns[i.address] = i.value
            k.update_value(v)

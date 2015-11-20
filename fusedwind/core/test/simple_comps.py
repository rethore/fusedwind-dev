
from openmdao.api import Component


class SimpleComp(Component):
    """ The simplest component you can imagine. """

    def __init__(self, multiplier=2.0):
        super(SimpleComp, self).__init__()

        self.multiplier = multiplier

        # Params
        self.add_param('x', 3.0)

        # Unknowns
        self.add_output('y', 5.5)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """
        unknowns['y'] = self.multiplier*params['x']

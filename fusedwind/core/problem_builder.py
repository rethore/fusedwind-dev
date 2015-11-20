
import importlib
import yaml

from openmdao.api import Problem, Group, IndepVarComp, ExecComp

def load_class(full_class_string):
    """
    dynamically load a class from a string
    """

    class_data = full_class_string.split(".")
    module_path = ".".join(class_data[:-1])
    class_str = class_data[-1]

    module = importlib.import_module(module_path)
    # Finally, we retrieve the Class
    return getattr(module, class_str)

def readyml(filename):
    """read a yaml file"""

    with open(filename, 'r') as f:
        return yaml.load(f.read())


class FUSEDProblem(Problem):
    """
    Instantiate an OpenMDAO problem defined in a yaml file

    TODO: add methods to deal with drivers, desvars, explicit connections etc!
    """

    def __init__(self, problem=None, filename=None):
        super(FUSEDProblem, self).__init__()

        if filename is not None:
            pb = readyml(filename)
            self.load_problem(pb)
        if problem is not None:
            self.load_problem(problem)

    def load_problem(self, pb):

        if 'root' in pb:
            if pb['root']['class'] == 'Group':
                self.root = Group()
            if 'components' in pb['root']:
                for c in pb['root']['components']:
                    if c['class'] == 'IndepVarComp':
                        self.root.add(c['name'], IndepVarComp(c['parameter'][0], c['parameter'][1]), promotes=c['promotes'])
                    elif c['class'] == 'ExecComp':
                        self.root.add(c['name'], ExecComp(c['expr'], **c['parameters']), promotes=c['promotes'])
                    else:
                        if 'parameters' not in c:
                            c['parameters'] = {}
                        self.root.add(c['name'], load_class(c['class'])(**c['parameters']), promotes=c['promotes'])

    def load_inputs(self, filename):

        inputs = readyml(filename)
        for k,v in inputs.items():
            self[k] = v

    def list_indepvars(self):

        indeps = []
        for c in self.root.components():
            if isinstance(c, IndepVarComp):
                indeps.extend(c._unknowns_dict.keys())
        return indeps

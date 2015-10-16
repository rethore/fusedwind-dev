
import time
import numpy as np
from scipy.interpolate import pchip

from openmdao.core import Component, Group, ParallelGroup
from openmdao.core.problem import Problem
from openmdao.components import IndepVarComp


def read_bladestructure(filebase):
    """
    input file reader of BladeStructureVT3D data

    parameters
    ----------
    filebase: str
        data files' basename

    returns
    -------
    st3d: dict
        dictionary containing geometric and material properties
        definition of the blade structure
    """

    st3d = {}

    # read mat file
    fid = open(filebase + '.mat', 'r')
    materials = fid.readline().split()[1:]
    st3d['materials'] = {name:i for i, name in enumerate(materials)}
    data = np.loadtxt(fid)
    st3d['matprops'] = data

    # read failmat file
    failcrit = {1:'maximum_strain', 2:'maximum_stress', 3:'tsai_wu'}
    fid = open(filebase + '.failmat', 'r')
    materials = fid.readline().split()[1:]
    data = np.loadtxt(fid)
    st3d['failmat'] = data[:, 1:]
    st3d['failcrit'] = [failcrit[mat] for mat in data[:, 0]]

    # read the dp3d file containing region division points
    dpfile = filebase + '.dp3d'

    dpfid = open(dpfile, 'r')

    # read webs
    wnames = dpfid.readline().split()[1:]
    iwebs = []
    for w, wname in enumerate(wnames):
        line = dpfid.readline().split()[1:]
        line = [int(entry) for entry in line]
        iwebs.append(line)
    st3d['web_def'] = iwebs
    nwebs = len(iwebs)
    header = dpfid.readline()
    dpdata = np.loadtxt(dpfile)
    nreg = dpdata.shape[1] - 2
    try:
        regions = header.split()[1:]
        assert len(regions) == nreg
    except:
        regions = ['region%02d' % i for i in range(nreg)]
    st3d['s'] = dpdata[:, 0]
    st3d['DPs'] = dpdata[:, 1:]

    # read the st3d files containing thicknesses and orientations
    st3d['regions'] = []
    for i, rname in enumerate(regions):
        r = {}
        layup_file = '_'.join([filebase, rname]) + '.st3d'
        fid = open(layup_file, 'r')
        rrname = fid.readline().split()[1]
        lheader = fid.readline().split()[1:]

        cldata = np.loadtxt(fid)
        layers = lheader[1:]
        nl = len(layers)

        r['layers'] = layers
        r['thicknesses'] = cldata[:, 1:nl + 1]
        if cldata.shape[1] == nl*2 + 1:
            r['angles'] = cldata[:, nl + 2:2*nl + 2]
        else:
            r['angles'] = np.zeros((cldata.shape[0], nl))
        st3d['regions'].append(r)

    st3d['webs'] = []
    for i, rname in enumerate(wnames):
        r = {}
        layup_file = '_'.join([filebase, rname]) + '.st3d'
        fid = open(layup_file, 'r')
        rrname = fid.readline().split()[1]
        lheader = fid.readline().split()[1:]

        cldata = np.loadtxt(fid)
        layers = lheader[1:]
        nl = len(layers)

        r['layers'] = layers
        r['thicknesses'] = cldata[:, 1:nl + 1]
        if cldata.shape[1] == nl*2 + 1:
            r['angles'] = cldata[:, nl + 2:2*nl + 2]
        else:
            r['angles'] = np.zeros((cldata.shape[0], nl))
        st3d['webs'].append(r)

    return st3d

def write_bladestructure(st3d, filebase):
    """
    input file writer for a blade structure definition

    parameters
    ----------
    st3d: dict
        dictionary containing geometric and material properties
        definition of the blade structure
    filebase: str
        data files' basename
    """

    # write material properties
    fid = open(filebase + '.mat', 'w')
    fid.write('# %s\n' % (' '.join(st3d['materials'].keys())))
    fid.write('# E1 E2 E3 nu12 nu13 nu23 G12 G13 G23 rho\n')
    fmt = ' '.join(10*['%.20e'])
    np.savetxt(fid, st3d['matprops'], fmt=fmt)

    failcrit = dict(maximum_strain=1, maximum_stress=2, tsai_wu=3)
    fid = open(filebase + '.failmat', 'w')
    fid.write('# %s\n' % (' '.join(st3d['materials'])))
    fid.write('# failcrit s11_t s22_t s33_t s11_c s22_c s33_c'
              't12 t13 t23 e11_t e22_t e33_t e11_c e22_c e33_c g12 g13 g23'
              'gM0 C1a C2a C3a C4a\n')
    data = np.zeros((st3d['failmat'].shape[0], st3d['failmat'].shape[1]+1))
    data[:, 0] = [failcrit[mat] for mat in st3d['failcrit']]
    data[:, 1:] = st3d['failmat']
    fmt = '%i ' + ' '.join(23*['%.20e'])
    np.savetxt(fid, np.asarray(data), fmt=fmt)

    # write dp3d file with region division points
    fid = open(filebase + '.dp3d', 'w')
    webs = ['web%02d' % i for i in range(len(st3d['webs']))]
    fid.write('# %s\n' % ('  '.join(webs)))
    for web in st3d['web_def']:
        fid.write('# %i %i\n' % (web[0], web[1]))
    DPs = ['DP%02d' % i for i in range(st3d['DPs'].shape[1])]
    fid.write('# s %s\n' % (' '.join(DPs)))
    data = np.array([st3d['s']]).T
    data = np.append(data, st3d['DPs'], axis=1)
    np.savetxt(fid, data)
    fid.close()

    # write st3d files with material thicknesses and angles
    for i, reg in enumerate(st3d['regions']):
        rname = 'region%02d' % i
        fname = '_'.join([filebase, rname]) + '.st3d'
        fid = open(fname, 'w')
        lnames = '    '.join(reg['layers'])
        fid.write('# %s\n' % rname)
        fid.write('# s    %s\n' % lnames)
        data = np.array([st3d['s']]).T
        data = np.append(data, reg['thicknesses'], axis=1)
        data = np.append(data, reg['angles'], axis=1)
        np.savetxt(fid, data)
        fid.close()
    for i, reg in enumerate(st3d['webs']):
        rname = 'web%02d' % i
        fname = '_'.join([filebase, rname]) + '.st3d'
        fid = open(fname, 'w')
        lnames = '    '.join(reg['layers'])
        fid.write('# %s\n' % rname)
        fid.write('# s    %s\n' % lnames)
        data = np.array([st3d['s']]).T
        data = np.append(data, reg['thicknesses'], axis=1)
        data = np.append(data, reg['angles'], axis=1)
        np.savetxt(fid, data)
        fid.close()

def interpolate_bladestructure(st3d, s_new):
    """
    interpolate a blade structure definition onto
    a new spanwise distribution using pchip

    parameters
    ----------
    st3d: dict
        dictionary with blade structural definition
    s_new: array
        1-d array with new spanwise distribution

    returns
    -------
    st3dn: dict
        blade structural definition interpolated onto s_new distribution
    """

    st3dn = {}
    sorg = st3d['s']
    st3dn['s'] = s_new
    st3dn['materials'] = st3d['materials']
    st3dn['matprops'] = st3d['matprops']
    st3dn['failmat'] = st3d['failmat']
    st3dn['failcrit'] = st3d['failcrit']
    st3dn['web_def'] = st3d['web_def']
    st3dn['regions'] = []
    st3dn['webs'] = []

    DPs = np.zeros((s_new.shape[0], st3d['DPs'].shape[1]))
    for i in range(st3d['DPs'].shape[1]):
        tck = pchip(sorg, st3d['DPs'][:, i])
        DPs[:, i] = tck(s_new)
    st3dn['DPs'] = DPs

    for r in st3d['regions']:
        rnew = {}
        rnew['layers'] = r['layers']
        Ts = r['thicknesses']
        As = r['angles']
        tnew = np.zeros((s_new.shape[0], Ts.shape[1]))
        anew = np.zeros((s_new.shape[0], As.shape[1]))
        for i in range(Ts.shape[1]):
            tck = pchip(sorg, Ts[:, i])
            tnew[:, i] = tck(s_new)
            tck = pchip(sorg, As[:, i])
            anew[:, i] = tck(s_new)
        rnew['thicknesses'] = tnew.copy()
        rnew['angles'] = anew.copy()
        st3dn['regions'].append(rnew)
    for r in st3d['webs']:
        rnew = {}
        rnew['layers'] = r['layers']
        Ts = r['thicknesses']
        As = r['angles']
        tnew = np.zeros((s_new.shape[0], Ts.shape[1]))
        anew = np.zeros((s_new.shape[0], As.shape[1]))
        for i in range(Ts.shape[1]):
            tck = pchip(sorg, Ts[:, i])
            tnew[:, i] = tck(s_new)
            tck = pchip(sorg, As[:, i])
            anew[:, i] = tck(s_new)
        rnew['thicknesses'] = tnew.copy()
        rnew['angles'] = anew.copy()
        st3dn['webs'].append(rnew)

    return st3dn


# this shouldn't be here, but probably in fusedwind.models.structure...?
class CSPropsBase(Component):
    """
    Template for components that can compute
    beam structural properties using a cross-sectional
    code such as PreComp, BECAS or VABS.

    parameters
    ----------
    config: dict
        dictionary of model specific inputs
    coords: array
        cross-sectional shape. size ((ni_chord, 3))
    matprops: array
        material stiffness properties. Size ((10, nmat)).
    failmat: array
        material strength properties. Size ((18, nmat)).
    DPs: array
        vector of DPs. Size: (nDP)
    coords: array
        blade section coordinates. Size: ((ni_chord, 3))
    r<xx><lname>T: float
        layer thicknesses, e.g. r01triaxT.
    r<xx><lname>A: float
        layer angles, e.g. r01triaxA.
    cs_invar: float
        test variable added to test input promotions

    outputs
    -------
    cs_props: array
        vector of cross section properties. Size (19).
    cs_outvar: float
        test variable added to test output promotions
    """

    def __init__(self, config, st3d, ni_chord):
        super(CSPropsBase, self).__init__()

        nr = len(st3d['regions'])

        # add materials properties array ((10, nmat))
        self.add_param('matprops', st3d['matprops'])

        # add materials strength properties array ((18, nmat))
        self.add_param('failmat', st3d['failmat'])

        # add DPs array
        self.add_param('DPs', np.zeros(nr + 1))

        # add coords coords
        self.add_param('coords', np.zeros((ni_chord, 3)))

        for ireg, reg in enumerate(st3d['regions']):
            for i, lname in enumerate(reg['layers']):
                varname = 'r%i%s' % (ireg, lname)
                self.add_param(varname + 'T', 0.)
                self.add_param(varname + 'A', 0.)

        # add outputs
        self.add_output('cs_props', np.zeros(19))

        # test vars
        self.add_param('cs_invar', 0.)
        self.add_output('cs_outvar', 0.)

    def solve_nonlinear(self, params, unknowns, resids):

        time.sleep(1.)
        if self.comm:
            print 'rank %i computing props for section %3.3f %i %i' % \
                                    (self.comm.rank, params['coords'][0, 2],
                                     params['coords'].shape[0], params['coords'].shape[1])
            print 'rank %i DPs'%self.comm.rank, params['DPs']
        else:
            print 'computing props for section', params['coords'][0, 2]
        unknowns['cs_props'][0] = params['coords'][0, 2]

        # test vars
        unknowns['cs_outvar'] = params['cs_invar'] * 2.


class Slice(Component):
    """
    simple component for slicing arrays into vectors
    for passing to sub-comps computing the csprops

    parameters
    ----------
    DPs: array
        2D array of DPs. Size: ((nsec, nDP))
    surface: array
        blade surface. Size: ((ni_chord, nsec, 3))

    outputs
    -------
    sec<xxx>DPs: array
        Vector of DPs along chord for each section. Size (nDP)
    sec<xxx>coords: array
        Array of cross section coords shapes. Size ((ni_chord, 3))
    """

    def __init__(self, DPs, surface):
        super(Slice, self).__init__()

        self.nsec = surface.shape[1]

        self.add_param('DPs', DPs)
        self.add_param('surface', surface)
        for i in range(self.nsec):
            self.add_output('sec%03dDPs' % i, DPs[i, :])
            self.add_output('sec%03dcoords' % i, surface[:, i, :])

    def solve_nonlinear(self, params, unknowns, resids):

        for i in range(self.nsec):
            unknowns['sec%03dDPs' % i] = params['DPs'][i, :]
            unknowns['sec%03dcoords' % i] = params['surface'][:, i, :]


class Postprocess(Component):
    """
    component for gathering cross section props
    into array as function of span

    parameters
    ----------
    cs_props<xxx>: array
        array of cross section props. Size (19).
    blade_s: array
        dimensionalised running length of blade
    hub_radius: float
        dimensionalised hub radius

    outputs
    -------
    beam_structure: array
        array of beam structure properties. Size ((nsec, 19)).
    """

    def __init__(self, nsec):
        super(Postprocess, self).__init__()

        self.nsec = nsec

        for i in range(nsec):
            self.add_param('cs_props%03d' % i, np.zeros(19))
        self.add_param('blade_s', np.zeros(nsec))
        self.add_param('hub_radius', 0.)


        self.add_output('beam_structure', np.zeros((nsec, 19)))
        self.add_output('blade_mass', 0.)
        self.add_output('blade_mass_moment', 0.)

    def solve_nonlinear(self, params, unknowns, resids):

        for i in range(self.nsec):
            cname = 'cs_props%03d' % i
            cs = params[cname]
            unknowns['beam_structure'][i, :] = cs


class CSBeamStructure(Group):
    """
    Group for computing beam structure properties
    using a cross-section structure code

    parameters
    ----------
    matprops: array
        material stiffness properties. Size (10, nmat).
    failmat: array
        material strength properties. Size (18, nmat).
    DPs: array
        2D array of DPs. Size: ((nsec, nDP))
    surface: array
        blade surface. Size: ((ni_chord, nsec, 3))
    r<xx><lname>T: array
        region layer thicknesses, e.g. r01triaxT. Size (nsec)
    r<xx><lname>A: array
        region layer angles, e.g. r01triaxA. Size (nsec)
    w<xx><lname>T: array
        web layer thicknesses, e.g. r01triaxT. Size (nsec)
    w<xx><lname>A: array
        web layer angles, e.g. r01triaxA. Size (nsec)

    outputs
    -------
    beam_structure: array
        array of beam structure properties. Size ((nsec, 19)).
    blade_mass: float
        blade mass integrated from beam_structure dm
    """

    def __init__(self, config, st3d, surface):
        """
        initializes parameters and adds a csprops component
        for each section

        parameters
        ----------
        config: dict
            dictionary of inputs for the cs_code class
        st3d: dict
            dictionary of blade structure properties
        surface: array
            blade surface. Size: ((ni_chord, nsec, 3))
        """
        super(CSBeamStructure, self).__init__()



        self.st3d = st3d
        nr = len(st3d['regions'])
        nsec = st3d['s'].shape[0]

        # add materials properties array ((10, nmat))
        self.add('matprops_c', IndepVarComp('matprops', st3d['matprops']), promotes=['*'])

        # add materials strength properties array ((18, nmat))
        self.add('failmat_c', IndepVarComp('failmat', st3d['failmat']), promotes=['*'])

        # add DPs array with s, DP0, DP1, ... DP<nr>
        self.add('DPs_c', IndepVarComp('DPs', st3d['DPs']), promotes=['*'])

        # add array containing blade section coords
        self.add('surface_c', IndepVarComp('surface', surface), promotes=['*'])

        # add comp to slice the 2D arrays DPs and surface
        self.add('slice', Slice(st3d['DPs'], surface), promotes=['*'])

        self._varnames = []
        for ireg, reg in enumerate(st3d['regions']):
            for i, lname in enumerate(reg['layers']):
                varname = 'r%i%s' % (ireg, lname)
                self.add(varname+'T_c', IndepVarComp(varname + 'T', np.zeros(nsec)), promotes=['*'])
                self.add(varname+'A_c', IndepVarComp(varname + 'A', np.zeros(nsec)), promotes=['*'])
                self._varnames.append(varname)

        # now add a component for each section
        cid = self.add('cid', ParallelGroup())
        CSCode = config['cs_props']['model']
        try:
            promotions = config['cs_props']['promotes']
        except:
            promotions = []
        for i in range(nsec):
            secname = 'sec%03d' % i
            cid.add(secname, CSCode(config['cs_props'], st3d, surface.shape[0]), promotes=promotions)
            # create connections
            self.connect('matprops', 'cid.%s.matprops' % secname)
            self.connect('failmat', 'cid.%s.failmat' % secname)
            self.connect(secname+'DPs', 'cid.%s.DPs' % secname)
            self.connect(secname+'coords', 'cid.%s.coords' % secname)

            for name in self._varnames:
                self.connect(name + 'T', 'cid.%s.%sT' % (secname, name), src_indices=([i]))
                self.connect(name + 'A', 'cid.%s.%sA' % (secname, name), src_indices=([i]))

        self.add('postpro', Postprocess(nsec), promotes=['hub_radius',
                                                         'blade_s',
                                                         'beam_structure',
                                                         'blade_mass',
                                                         'blade_mass_moment'])
        for i in range(nsec):
            secname = 'sec%03d' % i
            self.connect('cid.%s.cs_props' % secname, 'postpro.cs_props%03d' % i)

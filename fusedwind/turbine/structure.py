
import time
import re
import numpy as np
from scipy.interpolate import pchip

from openmdao.api import Component, Group, ParallelGroup
from openmdao.core.problem import Problem
from openmdao.api import IndepVarComp

from fusedwind.turbine.geometry import FFDSpline


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


class SplinedBladeStructure(Group):
    """
    class that adds structural geometry variables to the analysis
    either as either as splines with user defined control points
    or arrays according to the initial structural data
    """

    def __init__(self, st3d):
        """
        parameters
        ----------
        st3d: dict
            dictionary with blade structural definition
        """
        super(SplinedBladeStructure, self).__init__()

        self._vars = []
        self._allvars = []
        self.st3dinit = st3d

        # add materials properties array ((10, nmat))
        self.add('matprops_c', IndepVarComp('matprops', st3d['matprops']), promotes=['*'])

        # add materials strength properties array ((18, nmat))
        self.add('failmat_c', IndepVarComp('failmat', st3d['failmat']), promotes=['*'])

    def add_spline(self, name, Cx, spline_type='bezier', symm=True, scaler=1.):
        """
        adds a 1D FFDSpline for the given variable
        with user defined spline type and control point locations.

        parameters
        ----------
        name: str
            name of the variable, which should be of the form
            r04uniaxT or r04uniaxA for region 4 uniax thickness
            and angle, respectively.
        Cx: array
            spanwise distribution of control points
        spline_type: str
            spline type used in FFD, options:
            | bezier
            | pchip
        """

        st3d = self.st3dinit

        # decode the name
        if 'DP' in name:
            # try:
            iDP = int(re.match(r"([a-z]+)([0-9]+)", name, re.I).groups()[-1])
            # except:
            #     print('Variable name %s not understood' % name)
            #     return

            var = st3d['DPs'][:, iDP]
            self.add(name + '_c', IndepVarComp(name + '_C', np.zeros(len(Cx))), promotes=['*'])
            c = self.add(name + '_s', FFDSpline(name, st3d['s'],
                                                      var,
                                                      Cx, scaler=scaler),
                                                      promotes=['*'])
            c.spline_options['spline_type'] = spline_type
            self._vars.append(name)

        elif name.startswith('r') or name.startswith('w'):
            l_index = None
            try:
                ireg = int(name[1:3])
                try:
                    split = re.match(r"([a-z]+)([0-9]+)([a-z]+)", name[3:], re.I).groups()
                    l_index = split[1]
                except:
                    split = re.match(r"([a-z]+)([a-z]+)", name[3:], re.I).groups()
                layername = split[0]
                stype = split[-1]
            except:
                print('Variable name %s not understood' % name)
                return

            if name.startswith('r'):
                r = st3d['regions'][ireg]
                rname = 'r%02d' % ireg
            elif name.startswith('w'):
                r = st3d['webs'][ireg]
                rname = 'w%02d' % ireg
            if symm:
                lnames = [layername, layername + '01']
            else:
                lnames = [layername]
            if symm:
                self.add(name + '_c', IndepVarComp(name + '_C', np.zeros(len(Cx))), promotes=['*'])
            for lname in lnames:
                varname = '%s%s%s' % (rname, lname, stype)
                ilayer = r['layers'].index(lname)
                if stype == 'T':
                    var = r['thicknesses'][:, ilayer]
                elif stype == 'A':
                    var = r['angles'][:, ilayer]
                if symm:
                    self.connect(name + '_C', varname + '_s.' + varname + '_C')
                    promotes = [varname]
                else:
                    self.add(varname + '_c', IndepVarComp(varname + '_C', np.zeros(len(Cx))), promotes=['*'])
                    promotes = [varname, varname + '_C']
                c = self.add(varname + '_s', FFDSpline(varname, st3d['s'],
                                                          var,
                                                          Cx, scaler=scaler),
                                                          promotes=promotes)
                c.spline_options['spline_type'] = spline_type
                self._vars.append(varname)

    def configure(self):
        """
        add IndepVarComp's for all remaining planform variables
        """
        st3d = self.st3dinit

        for i in range(st3d['DPs'].shape[1]):
            varname = 'DP%02d' % i
            var = st3d['DPs'][:, i]
            if varname not in self._vars:
                self.add(varname + '_c', IndepVarComp(varname, var), promotes=['*'])

        for ireg, reg in enumerate(st3d['regions']):
            for i, lname in enumerate(reg['layers']):
                varname = 'r%02d%s' % (ireg, lname)
                if varname+'T' not in self._vars:
                    self.add(varname + 'T_c', IndepVarComp(varname + 'T', reg['thicknesses'][:, i]), promotes=['*'])
                if varname+'A' not in self._vars:
                    self.add(varname + 'A_c', IndepVarComp(varname + 'A', reg['angles'][:, i]), promotes=['*'])
        for ireg, reg in enumerate(st3d['webs']):
            for i, lname in enumerate(reg['layers']):
                varname = 'w%02d%s' % (ireg, lname)
                if varname+'T' not in self._vars:
                    self.add(varname + 'T_c', IndepVarComp(varname + 'T', reg['thicknesses'][:, i]), promotes=['*'])
                if varname+'A' not in self._vars:
                    self.add(varname + 'A_c', IndepVarComp(varname + 'A', reg['angles'][:, i]), promotes=['*'])

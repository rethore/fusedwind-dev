
import numpy as np

from openmdao.recorders.base_recorder import BaseRecorder
from fusedwind.turbine.structure import write_bladestructure

def get_structure_recording_vars(st3d, with_props=False, with_CPs=False):
    """
    convenience method for generating list of variable names
    of the blade structure for adding to a recorder

    params
    ------
    st3d: dict
        dictionary of blade structure
    with_props: bool
        also add variable names with blade structure props
        computed by `BladeStructureProperties`

    returns
    -------
    recording_vars: list
        list of strings with names of all DPs, layer names,
        and material props.
    """
    recording_vars = []
    s = st3d['s']
    nsec = s.shape[0]
    nDP = st3d['DPs'].shape[1]

    DPs = ['DP%02d' % i for i in range(nDP)]
    DPs_C = ['DP%02d_C' % i for i in range(nDP)]
    regions = []
    webs = []
    for ireg, reg in enumerate(st3d['regions']):
        layers = []
        for i, lname in enumerate(reg['layers']):
            varname = 'r%02d%s' % (ireg, lname)
            layers.extend([varname + 'T', varname + 'A'])
            if with_CPs:
                layers.extend([varname + 'T_C', varname + 'A_C'])
        regions.extend(layers)
        if with_props:
            regions.append('r%02d_thickness' % ireg)
            regions.append('r%02d_width' % ireg)
    for ireg, reg in enumerate(st3d['webs']):
        layers = []
        for i, lname in enumerate(reg['layers']):
            varname = 'w%02d%s' % (ireg, lname)
            layers.extend([varname + 'T', varname + 'A'])
            if with_CPs:
                layers.extend([varname + 'T_C', varname + 'A_C'])
        if with_props:
            regions.append('r%02d_thickness' % ireg)
            regions.append('r%02d_width' % ireg)
        webs.extend(layers)

    recording_vars.extend(DPs)
    if with_CPs:
        recording_vars.extend(DPs_C)
    recording_vars.extend(regions)
    recording_vars.extend(webs)
    recording_vars.append('matprops')
    recording_vars.append('failmat')
    recording_vars.append('s_st')

    if with_props:
        recording_vars.extend(['pacc_u',
                               'pacc_l',
                               'pacc_u_curv',
                               'pacc_l_curv'])
        for ireg, reg in enumerate(st3d['webs']):
            recording_vars.extend(['web_angle%02d' % ireg, 'web_offset%02d'%ireg])

    return recording_vars

def write_recorded_bladestructure(st3d, db, coordinate, filebase):

    data = db[coordinate]['Unknowns']

    s = st3d['s']
    stnew = {}
    stnew['s'] = s
    stnew['materials'] = st3d['materials']
    stnew['matprops'] = st3d['matprops']
    stnew['failcrit'] = st3d['failcrit']
    stnew['failmat'] = st3d['failmat']
    stnew['web_def'] = st3d['web_def']
    stnew['version'] = st3d['version']

    nsec = s.shape[0]
    nDP = st3d['DPs'].shape[1]

    DPs = np.array([data['DP%02d' % i] for i in range(nDP)]).T
    stnew['DPs'] = DPs
    stnew['regions'] = []
    stnew['webs'] = []
    for ireg, reg in enumerate(st3d['regions']):
        rnew = {}
        rnew['layers'] = reg['layers']
        nl = len(reg['layers'])
        Ts = np.zeros((nsec, nl))
        As = np.zeros((nsec, nl))

        for i, lname in enumerate(reg['layers']):
            varname = 'r%02d%s' % (ireg, lname)
            Ts[:, i] = data[varname + 'T']
            As[:, i] = data[varname + 'A']
        rnew['thicknesses'] = Ts
        rnew['angles'] = As
        stnew['regions'].append(rnew)

    for ireg, reg in enumerate(st3d['webs']):
        rnew = {}
        rnew['layers'] = reg['layers']
        nl = len(reg['layers'])
        Ts = np.zeros((nsec, nl))
        As = np.zeros((nsec, nl))

        for i, lname in enumerate(reg['layers']):
            varname = 'w%02d%s' % (ireg, lname)
            Ts[:, i] = data[varname + 'T']
            As[:, i] = data[varname + 'A']
        rnew['thicknesses'] = Ts
        rnew['angles'] = As
        stnew['webs'].append(rnew)

    write_bladestructure(stnew, filebase)


def get_planform_recording_vars(suffix='', with_CPs=False):
    """
    convenience method for generating list of variable names
    of the blade planform for adding to a recorder

    params
    ------
    suffix: str
        to record pf vars with e.g. _st appended to the variable names
    with_CPs: bool
        flag for also adding spline CPs arrays to list

    returns
    recording_vars: list
        list of strings with names of all planform vars
    """

    recording_vars = []

    names = ['x', 'y', 'z', 'rot_z', 'rot_y', 'rot_z',
                      'chord', 'rthick', 'p_le']


    if suffix != '':
        pf_vars = [name + suffix for name in names]
    else:
        pf_vars = names
    curv_vars = [name + '_curv' for name in pf_vars]

    cp_vars = []
    if with_CPs:
        cp_vars = [name + '_C' for name in names]

    recording_vars.extend(pf_vars)
    recording_vars.extend(curv_vars)
    recording_vars.extend(cp_vars)

    return recording_vars

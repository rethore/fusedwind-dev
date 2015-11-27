import numpy as np
from collections import OrderedDict

class Material(object):
    ''' Defines and handles material input data such as properties, resistances,
    and safety factors.
    
    Properties (Values should be set as mean values from measurements in SI units.)
    ----------
    :param E1: Young's modulus parallel (||) to fiber direction
    :param E2: Young's modulus perpendicular (_|_) to fiber direction (in lamina plane)
    :param E3: Young's modulus perpendicular (_|_) to fiber direction (out of lamina plane)
    :param nu12: major Poisson's ratio between fiber direction and perpendicular to fiber
        direction (in lamina plane)
    :param nu13: major Poisson's ratio between fiber direction and perpendicular to fiber
        direction (out of lamina plane)
    :param nu23: major Poisson's ratio between both in and out of lamina plane fiber
        directions
    :param nu21: minor Poisson's ratio between perpendicular direction to fiber and fiber
        direction (in lamina plane)
    :param nu31: minor Poisson's ratio between perpendicular direction to fiber and fiber
        direction (out of lamina plane)
    :param nu32: minor Poisson's ratio between both out of and in lamina plane fiber
        directions
        
    .. note:: 1st index = loading, 2nd index = contraction
              That is for a uniaxial layer nu12 is the larger (major) and nu21 is the
              smaller (minor) value.

    :param G12: Shear Modules (in lamina plane)
    :param G13: Shear Modules parallel to fiber direction and out of lamina plane
    :param G23: Shear Modules perpendicular to fiber direction and out of lamina plane
    :param rho: Density 
    :param failcrit: Failure criterion to be used for this material
        {1:'maximum_strain', 2:'maximum_stress', 3:'tsai_wu'}
        
    Resistances:
    -----------
    :param s11_t: allowable tensile stress parallel to fiber direction
    :param s22_t: allowable tensile stress perpendicular to fiber direction (in lamina plane)
    :param s33_t: allowable tensile stress perpendicular to fiber direction (out-of lamina plane)
    :param s11_c: allowable compressive stress parallel to fiber direction
    :param s22_c: allowable compressive stress perpendicular to fiber direction (in lamina plane)
    :param s33_c: allowable compressive stress perpendicular to fiber direction (out-of lamina plane)
    :param t12: allowable shear stress in lamina plane
    :param t13: allowable shear stress out of lamina plane, parallel to fiber direction
    :param t23: allowable shear stress out of lamina plane, perpendicular to fiber direction 
    :param e11_t: allowable tensile strain parallel to fiber direction
    :param e22_t: allowable tensile strain perpendicular to fiber direction (in lamina plane)
    :param e33_t: allowable tensile strain perpendicular to fiber direction (out-of lamina plane)
    :param e11_c: allowable compressive strain parallel to fiber direction
    :param e22_c: allowable compressive strain perpendicular to fiber direction (in lamina plane)
    :param e33_c: allowable compressive strain perpendicular to fiber direction (out-of lamina plane)
    :param g12: allowable shear strain in lamina plane
    :param g13: allowable shear strain out of lamina plane, parallel to fiber direction
    :param g23: allowable shear strain out of lamina plane, perpendicular to fiber direction 
    
    Safety Factors according to GL2010 scheme
    -----------------------------------------
    :param gM0: general material safety factor
    :param C1a: safety factor for influence of ageing
    :param C2a: safety factor for temperature effects
    :param C3a: safety factor for the manufacturing process
    :param C4a: safety factor for the effect of post-curing
    '''
    
    def _minor_poissons_ratios(self):
        ''' Derives minor Poisson's ratios
        '''
        self.nu31 = self.nu13 * self.E3 / self.E1
        self.nu21 = self.nu12 * self.E2 / self.E1
        self.nu32 = self.nu23 * self.E3 / self.E2
        
    def set_props_iso(self, E1, nu12, rho):
        ''' Sets isotropic material properties.
        '''
        self.rho = rho
        self.E1 = E1
        self.nu12 = nu12
        
        # derived
        self.E2 = self.E1
        self.E3 = self.E1
        self.nu23 = self.nu12
        self.nu13 = self.nu12
        self.G12 = self.E1 / (2* (1 + self.nu12))
        self.G23 = self.G12
        self.G13 = self.G12
        self._minor_poissons_ratios()

    def set_props_uniax(self, E1, E2, nu12, G12, nu23, rho):
        ''' Sets material properties for uniax.
        '''
        self.rho = rho
        self.E1 = E1
        self.E2 = E2
        self.nu12 = nu12
        self.G12 = G12
        self.nu23 = nu23
        
        # derived
        self.E3 = self.E2
        self.G23 = self.E2 / (2 * (1 + self.nu23)) # Schuermann, p.202, eq. 8.35
        self.nu13 = self.nu12
        self.G13 = self.G12
        self._minor_poissons_ratios()
        
    def set_props(self, E1, E2, E3, nu12, nu13, nu23, G12, G13, G23, rho):
        ''' Set 3D material properties.
        '''
        self.rho = rho
        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.nu12 = nu12
        self.nu13 = nu13
        self.nu23 = nu23
        self.G12 = G12
        self.G13 = G13
        self.G23 = G23
        
        self._minor_poissons_ratios()
        
    def matprops(self):
        ''' Returns the list of material properties.
        
        :return: A list of material properties suitable for the st3d dict
        
        '''
        return [self.E1,
                 self.E2,
                 self.E3,
                 self.nu12,
                 self.nu13,
                 self.nu23,
                 self.G12,
                 self.G13,
                 self.G23,
                 self.rho]
    
    def set_resists_strains_iso(self, failcrit, e11_t, e11_c, g12):
        ''' Sets the characteristic allowable strains for an isotropic material.
        '''
        self.failcrit = failcrit
        self.e11_t = e11_t
        self.e11_c = e11_c
        self.g12 = g12
        
        # derived
        self.e22_t = self.e11_t
        self.e33_t = self.e11_t
        self.e22_c = self.e11_c
        self.e33_c = self.e11_c
        self.g13 = self.g12
        self.g23 = self.g12
        self._resists_stresses()
        
    def set_resists_strains_uniax(self, failcrit, e11_t, e22_t, e11_c, e22_c, g12):
        ''' Sets the characteristic allowable strains for a uniax material.
        '''
        self.failcrit = failcrit
        self.e11_t = e11_t
        self.e11_c = e11_c
        self.e22_t = e22_t
        self.e22_c = e22_c
        self.g12 = g12
        
        # derived
        self.e33_t = self.e22_t
        self.e33_c = self.e22_c
        self.g13 = self.g12
        self.g23 = self.g12
        self._resists_stresses()

    def set_resists_strains(self, failcrit, e11_t, e22_t, e33_t, e11_c, e22_c,
                            e33_c, g12, g13, g23):
        ''' Sets the characteristic allowable strains.
        '''
        self.failcrit = failcrit
        self.e11_t = e11_t
        self.e22_t = e22_t
        self.e33_t = e33_t
        self.e11_c = e11_c
        self.e22_c = e22_c
        self.e33_c = e33_c
        self.g12 = g12
        self.g13 = g13
        self.g23 = g23
        
        self._resists_stresses()
        
    def _resists_stresses(self):
        ''' Determines stress resistances from strain resistances and stiffnesses
        '''
        self.s11_t = self.e11_t * self.E1
        self.s22_t = self.e22_t * self.E2
        self.s33_t = self.e33_t * self.E3
        self.s11_c = self.e11_c * self.E1
        self.s22_c = self.e22_c * self.E2
        self.s33_c = self.e33_c * self.E3
        self.t12 = self.g12 * self.G12
        self.t13 = self.g13 * self.G13
        self.t23 = self.g23 * self.G23
        
    def set_safety_GL2010(self, gM0, C1a, C2a, C3a, C4a):
        ''' Sets the material safety factors.
        '''
        self.gM0 = gM0
        self.C1a = C1a
        self.C2a = C2a
        self.C3a = C3a
        self.C4a = C4a
        
    def failmat(self):
        ''' Returns the list of material resistances and safety factors.
        
        :return: A list of resistances and safety factors suitable for the 
                 st3d dict
        '''
        return [self.s11_t,
                self.s22_t,
                self.s33_t,
                self.s11_c,
                self.s22_c,
                self.s33_c,
                self.t12,
                self.t13,
                self.t23,
                self.e11_c,
                self.e22_c,
                self.e33_c,
                self.e11_t,
                self.e22_t,
                self.e33_t,
                self.g12,
                self.g13,
                self.g23,
                self.gM0,
                self.C1a,
                self.C2a,
                self.C3a,
                self.C4a]
        
class DivisionPoint(object):
    '''Holds a division point's arc positions on the blade surface.
    
    :param arc: arc length positions on airfoil's surface 
            -1.0 = trailing edge suction side
            1.0 = trailing edge pressure side
            0.0 = leading edge
    :type arc: array 
    '''
    def __init__(self):
        self.arc = None
       
class Layer(object):
    """ Holds a layer's thickness and angle along the blade.

    :param thickness: layer thickness
    :type thickness: array
    :param angle: layup angle (deg)
    :type angle: array
    
    .. note:: A layer thickness can go to zero if material disappears at
              a certain spanwise position.
    """
    def __init__(self):
        self.thickness = None
        self.angle = None 

class Region(object):
    """ Holds a region's layers along the blade.
    
    :param layers: Dictionary of Layer3D objects
    :type layers: dict
    """
    def __init__(self):
        self.layers = OrderedDict()

    def add_layer(self, name):
        ''' Inserts a layer into layers dict.
        
        :param name: Name of the material
        :return: The layer added to the region
        '''
        dubl = 0
        for k in self.layers.iterkeys():
            if name in k:
                dubl += 1

        if dubl > 0:
            lname = '%s%02d' % (name, dubl)
        else:
            lname = name
        
        layer = Layer()
        self.layers[lname] = layer
        return layer
    
class BladeLayup(object):
    """ Span-wise layup definition of a blade.
    
    :param s: Spanwise discretization of the blade's layup
    :type s: array
    :param regions: Dictionary of Region3D objects representing regions on blade
                    surface
    :param webs: Dictionary of Region3D objects representing regions as webs
    :param iwebs: DP indices connecting webs to the surface
    :param DPs: Dictionary of DivisionPoint objects
    :param materials: Dictionary of Material objects
    
    """
    def __init__(self):
        self.s = None
        self.regions = OrderedDict()
        self.webs = OrderedDict()
        self.iwebs = None
        self.DPs = OrderedDict()
        self.materials = OrderedDict()

    def init_regions(self, nr, names=[]):
        ''' Initialize a number of nr regions.
        
        :param nr: Number of regions to be initialized
        :type nr: integer
        :param names: Names of regions (optional), must have the length of nr
        '''

        for i in range(nr + 1):
            self.DPs['DP%02d' % i] = DivisionPoint()

        for i in range(nr):
            try:
                name = names[i]
            except:
                name = 'region%02d' % i
            self._add_region(name)

    def init_webs(self, nw, iwebs, names=[]):
        ''' Initialize a number of nw webs.
        
        :param nw: Number of webs to be initialized
        :type nw: integer
        :param iwebs: List of DP index pairs connecting a web
            Example: [[2, 3], [1, 4]] means 2 webs, web1 uses DP02 and DP03,
            web2 uses DP01 and DP04
        :param names: Names of webs (optional), must have the length of nw
        '''
        self.iwebs = iwebs

        for i in range(nw):
            try:
                name = names[i]
            except:
                name = 'web%02d' % i
            self._add_web(name)

    def _add_region(self, name):
        ''' Adds region to the blade
        '''
        region = Region()
        self.regions[name] = region
        return region

    def _add_web(self, name):
        ''' Adds web to the blade
        '''
        region = Region()
        self.webs[name] = region
        return region
    
    def add_material(self, name):
        ''' Inserts material into materials dict.
        
        :param name: Name of the material.
        :return: The added material object.
        '''
        material = Material()
        self.materials[name] = material
        return material
        
        
def create_bladestructure(bl):
    """ Creator for BladeStructureVT3D data from a BladeLayup object

    :param bl: BladeLayupShell object
    :return: The st3d dictionary containing geometric and material properties
        definition of the blade structure
    """
    
    st3d = {}
    
    st3d['materials'] = {name:i for i, name in enumerate(bl.materials.iterkeys())}

    matprops = []
    failmat = []
    failcrit = []
    for v in bl.materials.itervalues():
        matprops.append(v.matprops())
        failmat.append(v.failmat())
        failcrit.append(v.failcrit)
        
    st3d['matprops'] = np.r_[matprops]
    st3d['failmat'] =  np.r_[failmat]
    st3d['failcrit'] = failcrit
    st3d['web_def'] = bl.iwebs
    st3d['s'] = bl.s
    
    dpdata = []
    for v in bl.DPs.itervalues():
        dpdata.append(v.arc)
    st3d['DPs'] = np.rot90(np.r_[dpdata], 1)
    
    def create_regions(dictionary):
        ''' create regions list
        
        :param dictionary: bl.regions or bl.webs
        :return: List of regions
        '''
        regs = []
        for k, v in dictionary.iteritems():
            r = {}
            r['layers'] = []
            andata = []
            thdata = []
            for k, v in v.layers.iteritems():
                r['layers'].append(k)
                thdata.append(v.thickness)
                andata.append(v.angle)
            r['thicknesses'] = np.rot90(np.r_[thdata], -1)
            r['angles'] = np.rot90(np.r_[andata], 1)
            regs.append(r)
        return regs
    
    st3d['regions'] = create_regions(bl.regions)
    st3d['webs'] = create_regions(bl.webs)
    
    return st3d
        

if __name__ == '__main__':
    
    mat1 = Material()
    mat1.set_props(E1 = 1.392E+10,
                     E2 = 1.392E+10,
                     E3 = 1.2099E+10,
                     nu12 = 0.533,
                     nu13 = 0.275,
                     nu23 = 0.332899,
                     G12 = 1.15E+10,
                     G13 = 4.53864E+09,
                     G23 = 4.53864E+09,
                     rho = 1.845E+03)
    
    mat1.set_resists_strains(failcrit = 1,
                             e11_t = 9.52E-03,
                             e22_t = 1.00E+06,
                             e33_t = 1.00E+06,
                             e11_c = 6.80E-03,
                             e22_c = 1.00E+06,
                             e33_c = 1.00E+06,
                             g12 = 1.00E+06,
                             g13 = 1.00E+06,
                             g23 = 1.00E+06)
    
    mat1.set_safety_GL2010(gM0 = 1.25,
                           C1a = 1.0,
                           C2a = 1.0,
                           C3a = 1.0,
                           C4a = 1.0)
    
    print(mat1.matprops())
    
    mat2 = Material()
    mat2.set_props_uniax(E1 = 4.163000000000000000e+10,
                         E2 = 1.493000000000000000e+10,
                         nu12 = 2.409999999999999920e-01,
                         G12  = 5.047000000000000000e+09,
                         nu23  = 3.301000000000000045e-01,
                         rho = 1.9155E+03)
    print(mat2.matprops())
    
    mat3 = Material()
    mat3.set_props_iso(E1 = 3.00E+09, nu12 = .3800, rho = 1.180E+03)
    print(mat3.matprops())
    print(mat1.failmat())
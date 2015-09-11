
from openmdao.core import Problem, Group

from turbine_costsse.turbine_costsse_2015 import *
from turbine_costsse.nrel_csm_tcc_2015 import *

from seamloads.SEAMLoads import SEAMLoads
from seamtower.SEAMTower import SEAMTower
from seamrotor.seamrotor import SEAMBladeStructure
from seamaero.seam_aep import SEAM_PowerCurve


class FUSEDTurbineCostsModel(Group):

    def __init__(self, config):

        super(FUSEDTurbineCostsModel, self).__init__()

        if config['tower'] == 'seam' or config['blade'] == 'seam':
            self.add('loads', SEAMLoads(26), promotes=['*'])

        if config['blade'] == 'csm':
            self.add('blade',BladeMass(), promotes=['*'])
        elif config['blade'] == 'seam':
            self.add('blade', SEAMBladeStructure(), promotes=['*'])

        self.add('hub',HubMass(), promotes=['*'])
        self.add('pitch',PitchSystemMass(), promotes=['*'])
        self.add('spinner',SpinnerMass(), promotes=['*'])
        self.add('lss',LowSpeedShaftMass(), promotes=['*'])
        self.add('bearing',BearingMass(), promotes=['*'])
        self.add('gearbox',GearboxMass(), promotes=['*'])
        self.add('hss',HighSpeedSideMass(), promotes=['*'])
        self.add('generator',GeneratorMass(), promotes=['*'])
        self.add('bedplate',BedplateMass(), promotes=['*'])
        self.add('yaw',YawSystemMass(), promotes=['*'])
        self.add('hvac',HydraulicCoolingMass(), promotes=['*'])
        self.add('cover',NacelleCoverMass(), promotes=['*'])
        self.add('other',OtherMainframeMass(), promotes=['*'])
        self.add('transformer',TransformerMass(), promotes=['*'])

        if config['tower'] == 'csm':
            self.add('tower',TowerMass(), promotes=['*'])
        elif config['tower'] == 'seam':
            self.add('tower', SEAMTower(21), promotes=['*'])

        self.add('turbine',turbine_mass_adder(), promotes=['*'])

        self.add('blade_c',BladeCost2015(), promotes=['*'])
        self.add('hub_c',HubCost2015(), promotes=['*'])
        self.add('pitch_c',PitchSystemCost2015(), promotes=['*'])
        self.add('spinner_c',SpinnerCost2015(), promotes=['*'])
        self.add('hub_adder',HubSystemCostAdder2015(), promotes=['*'])
        self.add('rotor_adder',RotorCostAdder2015(), promotes=['*'])
        self.add('lss_c',LowSpeedShaftCost2015(), promotes=['*'])
        self.add('bearing_c',BearingsCost2015(), promotes=['*'])
        self.add('gearbox_c',GearboxCost2015(), promotes=['*'])
        self.add('hss_c',HighSpeedSideCost2015(), promotes=['*'])
        self.add('generator_c',GeneratorCost2015(), promotes=['*'])
        self.add('bedplate_c',BedplateCost2015(), promotes=['*'])
        self.add('yaw_c',YawSystemCost2015(), promotes=['*'])
        self.add('hvac_c',HydraulicCoolingCost2015(), promotes=['*'])
        self.add('controls_c',ControlsCost2015(), promotes=['*'])
        self.add('vs_c', VariableSpeedElecCost2015(), promotes=['*'])
        self.add('elec_c', ElecConnecCost2015(), promotes=['*'])
        self.add('cover_c',NacelleCoverCost2015(), promotes=['*'])
        self.add('other_c',OtherMainframeCost2015(), promotes=['*'])
        self.add('transformer_c',TransformerCost2015(), promotes=['*'])
        self.add('nacelle_adder',NacelleSystemCostAdder2015(), promotes=['*'])
        self.add('tower_c',TowerCost2015(), promotes=['*'])
        self.add('tower_adder',TowerCostAdder2015(), promotes=['*'])
        self.add('turbine_c',TurbineCostAdder2015(), promotes=['*'])

def example():

    with_seam = False
    if with_seam:
        config = {'blade': 'seam', 'tower': 'seam'}
    else:
        config = {'blade': 'csm', 'tower': 'csm'}

    turbine = FUSEDTurbineCostsModel(config)
    prob = Problem(turbine)
    prob.setup()

    prob['rotor_diameter'] = 126.0
    prob['blade_number'] = 3
    prob['machine_rating'] = 5000.0
    prob['hub_height'] = 90.0
    prob['bearing_number'] = 2
    prob['crane'] = True
    prob['offshore'] = False

    # Rotor force calculations for nacelle inputs
    maxTipSpd = 80.0
    maxEfficiency = 0.90

    ratedHubPower  = prob['machine_rating']*1000. / maxEfficiency
    rotorSpeed     = (maxTipSpd/(0.5*prob['rotor_diameter'])) * (60.0 / (2*np.pi))
    prob['rotor_torque'] = ratedHubPower/(rotorSpeed*(np.pi/30))

    # other inputs
    prob['machine_rating'] = 5000.0
    prob['blade_number'] = 3
    prob['crane'] = True
    prob['offshore'] = True
    prob['bearing_number'] = 2

    if not with_seam:
        prob['turbine_class'] = 1
        prob['blade_has_carbon'] = False
    else:
        prob['tsr'] = 8.0
        prob['rated_power'] = 5.
        prob['max_tipspeed'] = 62.
        prob['min_wsp'] = 0.
        prob['max_wsp'] = 25.
        prob['project_lifetime'] = 20.

        prob['tower_bottom_diameter'] = 4.
        prob['tower_top_diameter'] = 2.

        # loads inputs
        prob['Iref'] = 0.16
        prob['F'] = 0.777
        prob['wohler_exponent_blade_flap'] = 10.0
        prob['wohler_exponent_tower'] = 4.
        prob['nSigma4fatFlap'] = 1.2
        prob['nSigma4fatTower'] = 0.8
        prob['dLoad_dU_factor_flap'] = 0.9
        prob['dLoad_dU_factor_tower'] = 0.8
        prob['lifetime_cycles'] = 1.0e07
        prob['EdgeExtDynFact'] = 2.5
        prob['EdgeFatDynFact'] = 0.75
        prob['WeibullInput'] = True
        prob['WeiA_input'] = 11.
        prob['WeiC_input'] = 2.00
        prob['Nsections'] = 21
        prob['lifetime_cycles'] = 1e7
        prob['wohler_exponent_blade_flap'] = 10.0
        prob['PMtarget'] = 1.0

        prob['MaxChordrR'] = 0.2
        prob['TIF_FLext'] = 1.
        prob['TIF_EDext'] = 1.
        prob['TIF_FLfat'] = 1.
        prob['sc_frac_flap'] = 0.3
        prob['sc_frac_edge'] = 0.8
        prob['SF_blade'] = 1.1
        prob['Slim_ext_blade'] = 200.0
        prob['Slim_fat_blade'] = 27
        prob['AddWeightFactorBlade'] = 1.2
        prob['blade_density'] = 2100.

        prob['wohler_exponent_tower'] = 4.
        prob['stress_limit_extreme_tower'] = 235.0
        prob['stress_limit_fatigue_tower'] = 14.885
        prob['safety_factor_tower'] = 1.5


    return prob

if __name__ == "__main__":

    prob = example()
    prob.run()

    print 'Turbine cost:', prob['turbine_cost']
    print 'Blade mass:', prob['blade_mass']
    print 'Tower mass:', prob['tower_mass']

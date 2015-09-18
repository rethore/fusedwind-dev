
from openmdao.core import Group

from turbine_costsse.turbine_costsse_2015 import *
from turbine_costsse.nrel_csm_tcc_2015 import *

from seamloads.SEAMLoads import SEAMLoads
from seamtower.SEAMTower import SEAMTower
from seamrotor.seamrotor import SEAMBladeStructure
from seamaero.seam_aep import SEAM_PowerCurve, SEAM_AEP

class FUSEDTurbineDesignModel(Group):

    def __init__(self, config):

        super(FUSEDTurbineDesignModel, self).__init__()

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

        self.add('mass_adder',turbine_mass_adder(), promotes=['*'])


class FUSEDTurbineAEPModel(Group):

    def __init__(self, config):
        super(FUSEDTurbineAEPModel, self).__init__()

        self.add('rotor_aero', SEAM_PowerCurve(), promotes=['*'])
        self.add('aep_calc', SEAM_AEP(), promotes=['*'])


class FUSEDTurbineCapexModel(Group):

    def __init__(self, config):

        super(FUSEDTurbineCapexModel, self).__init__()

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

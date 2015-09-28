
from openmdao.core import Problem, Group

from seamloads.SEAMLoads import SEAMLoads
from seamtower.SEAMTower import SEAMTower
from seamrotor.seamrotor import SEAMBladeStructure
from seamaero.seam_aep import SEAM_PowerCurve


if __name__ == '__main__':

    prob = Problem(root=Group())

    prob.root.add('loads', SEAMLoads(26), promotes=['*'])
    prob.root.add('tower', SEAMTower(21), promotes=['*'])
    prob.root.add('blade', SEAMBladeStructure(), promotes=['*'])
    prob.root.add('power_curve', SEAM_PowerCurve(26), promotes=['*'])

    prob.setup()

    # global variables
    prob['tsr'] = 8.0
    prob['rated_power'] = 3.
    prob['max_tipspeed'] = 62.
    prob['min_wsp'] = 0.
    prob['max_wsp'] = 25.
    prob['project_lifetime'] = 20.

    prob['rotor_diameter'] = 101.0
    prob['hub_height'] = 100.0
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

    prob.run()

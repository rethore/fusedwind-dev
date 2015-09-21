import yaml
import numpy as np

from openmdao.core import Problem, Group

from fusedwind.workflows.turbine_design import FUSEDModelBuilder

main_config = yaml.load(open('turbine_costs.json','r'))

prob = Problem(root=Group())
prob.root.add('turbine_design', FUSEDModelBuilder(main_config['turbine_design']), promotes=['*'])
prob.setup()

prob['rotor_diameter'] = 126.0
prob['blade_number'] = 3
prob['machine_rating'] = 5000.0
prob['hub_height'] = 90.0
# prob['bearing_number'] = 2
# prob['crane'] = True
# prob['offshore'] = False

# Rotor force calculations for nacelle inputs
maxTipSpd = 80.0
maxEfficiency = 0.90

ratedHubPower  = prob['machine_rating']*1000. / maxEfficiency
rotorSpeed     = (maxTipSpd/(0.5*prob['rotor_diameter'])) * (60.0 / (2*np.pi))
prob['rotor_torque'] = ratedHubPower/(rotorSpeed*(np.pi/30))

# other inputs

config = main_config['turbine_design']

# if config['blade'] == 'csm':
#     prob['turbine_class'] = 1
#     prob['blade_has_carbon'] = False
prob['tsr'] = 8.0
prob['rated_power'] = 5.
prob['max_tipspeed'] = 62.
prob['min_wsp'] = 0.
prob['max_wsp'] = 25.
prob['project_lifetime'] = 20.

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

prob['tower_bottom_diameter'] = 6.
prob['tower_top_diameter'] = 3.78
prob['wohler_exponent_tower'] = 4.
prob['stress_limit_extreme_tower'] = 235.0
prob['stress_limit_fatigue_tower'] = 14.885
prob['safety_factor_tower'] = 1.5


prob.run()

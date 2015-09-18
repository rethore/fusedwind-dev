
import numpy as np
import unittest

from openmdao.core import Problem, Group

from fusedwind.workflows.turbine_costs import FUSEDTurbineDesignModel, \
                                              FUSEDTurbineAEPModel, \
                                              FUSEDTurbineCapexModel


def configure_problem(config):


    prob = Problem(root=Group())
    prob.root.add('turbine_design', FUSEDTurbineDesignModel(config), promotes=['*'])
    # prob.root.add('turbine_aep', FUSEDTurbineAEPModel({}), promotes=['*'])
    prob.root.add('turbine_cost', FUSEDTurbineCapexModel({}), promotes=['*'])

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

    if config['blade'] == 'csm':
        prob['turbine_class'] = 1
        prob['blade_has_carbon'] = False
    else:
        prob['tsr'] = 8.0
        prob['rated_power'] = 5.
        prob['max_tipspeed'] = 62.
        prob['min_wsp'] = 0.
        prob['max_wsp'] = 25.
        prob['project_lifetime'] = 20.

    if config['blade'] == 'seam' or config['tower'] == 'seam':
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

    if config['blade'] == 'seam':
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

    if config['tower'] == 'seam':
        prob['tower_bottom_diameter'] = 6.
        prob['tower_top_diameter'] = 3.78
        prob['wohler_exponent_tower'] = 4.
        prob['stress_limit_extreme_tower'] = 235.0
        prob['stress_limit_fatigue_tower'] = 14.885
        prob['safety_factor_tower'] = 1.5

    # if config['aep'] == 'seam':
    #     prob['n_wsp'] = 26
    #     prob['min_wsp'] = 0  #[m/s]
    #     prob['max_wsp'] = 25 #[m/s]
    #     prob['air_density'] = 1.225      # [kg / m^3]
    #     prob['turbulence_int'] = 0.1     # Fraction
    #     prob['max_Cp'] = 0.49            # Fraction
    #     prob['gearloss_const'] = 0.01    # Fraction
    #     prob['gearloss_var'] = 0.014     # Fraction
    #     prob['genloss'] = 0.03          # Fraction
    #     prob['convloss'] = 0.03         # Fraction
    #
    #     prob['WeibullInput'] = True        # 1(true) or 0 (false) if true WeiA and WeiC overrules MeanWSP. If false MeanWSP is used with Rayleigh distribution
    #     prob['WeiA_input'] = 11.       #[m/s]
    #     prob['WeiC_input'] = 2.00       #[-]

    return prob

class TestCostModel(unittest.TestCase):

    def test_with_csm(self):

        config = {'blade': 'csm', 'tower': 'csm'}

        prob = configure_problem(config)
        prob.run()

        self.assertAlmostEqual(prob['turbine_cost'], 4087716.4195355722, places = 6)

    def test_with_seam(self):

        config = {'blade': 'seam', 'tower': 'seam', 'aep':''}

        prob = configure_problem(config)
        prob.run()

        self.assertAlmostEqual(prob['turbine_cost'], 4171073.1873904169, places = 6)


if __name__ == "__main__":

    # # config = {'blade': 'seam', 'tower': 'seam', 'aep':'seam'}
    # config = {'blade': 'csm', 'tower': 'csm', 'aep':'csm'}
    #
    # prob = configure_problem(config)
    # prob.run()
    # print 'Turbine cost:', prob['turbine_cost']
    # print 'Blade mass:', prob['blade_mass']
    # print 'Tower mass:', prob['tower_mass']

    unittest.main()

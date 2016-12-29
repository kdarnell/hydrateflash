#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 14:52:32 2016

@author: kdarnell
"""

import numpy as np

"""
Treat each possible guest/water molecule as an object
and return all of its properties for every EOS.
"""
# Possible (lowercase) aliases of each component
h2olist = ['h2o', 'h_2o', 'h20', 'water']
ch4list = ['ch4', 'ch_4', 'c1', 'methane']
c2h6list = ['c2h6', 'c_2h_6', 'c2', 'ethane']
c3h8list = ['c3h8', 'c_3h_8', 'c3', 'propane']
co2list = ['co2', 'co_2', 'carbon dioxide']
n2list = ['n2', 'n_2', 'nitrogen']


class component:
    def __init__(self, compstr):
        self.description = "Intrinsic and empirical properties"
        self.warningflag = False
        if compstr.lower() in h2olist:
            self.compname = 'h2o'
        elif compstr.lower() in ch4list:
            self.compname = 'ch4'
        elif compstr.lower() in c2h6list:
            self.compname = 'c2h6'
        elif compstr.lower() in c3h8list:
            self.compname = 'c3h8'
        elif compstr.lower() in co2list:
            self.compname = 'co2'
        elif compstr.lower() in n2list:
            self.compname = 'n2'
        else:
            self.warningflag = True
            print("Warning: " + compstr +
                  " component is not currently supported!!")
        """
        Set all relevant properties for each component.
        These properties are as follows:
            Generic:
            Tc - 'critical temperature', units = 'Kelvin'
            Pc - 'critical pressure', units = 'bar'
            Vc - 'critical volume', units = 'cm^3'
            MW - 'molecular weight', units = 'g/mol'
            N_carb - 'number of carbon atoms', units = 'none'
            Diam - 'molecular diamter, units = 'angstrom'
            gi0 -
            'ideal gas gibb free energy at standard state', units = 'J/mol'
            gi0_ast -
            'partial molar gibb free energy at standard state', units = 'J/mol'
            hi0 - 'ideal gas enthalpy at standard state', units = 'J/mol'
            hi0_ast -
            'partial molar enthalpy at standard state', units = 'J/mol'
            cp -
            'ideal gas heat capacity parameters',
            units = 'Kelvin'[to power of constant]
            SRK EOS:
            SRK_omega - 'acentricity factor', units = 'none'
            SRK_S2 - 'SRK factor', units = 'none'
            SRK_kij = 'interaction factor', units = 'none'
            Hydrate (Van der Waals-Plaateuw Modifed) EOS:
                Structure 1:
                    HVdPM_S1_kappa -
                    'compressibility constant', units = 'bar^-1'
                    HVdPM_S1_large - 'large cage factor', units = ''
                    HVdPM_S1_small - 'small cage factor', units = ''
                    Structure 2:
                        HVdPM_S2_kappa -
                        'compressibility constant', units = 'bar^-1'
                        HVdPM_S2_large - 'large cage factor', units = ''
                        HVdPM_S2_small - 'small cage factor', units = ''
                    Kihara potential:
                        HVdPM_kih_a - 'well-radius, units = 'angstrom'
                        HVdPM_kih_sigma - 'well-depth', units = 'angstrom'
                        HVdPM_kih_epsk - 'well paramter, units = 'Kelvin'
                        ideal_HS1 -
                        'hydrate structure 1 factors for ideal case, units = ''
                        ideal_HS2 -
                        'hydrate structure 2 factors for ideal case, units = ''
            Aqueous EOS:
                omega_born - 'Born constant', units = 'J/mol'
                aq_cp -
                'partial molar heat capacity factors in aqueous phase',
                units = 'variable'
                aq_v -
                'partial molar volume in aqueous phase', units = 'variable'            
        """
        if not self.warningflag:
            if self.compname == 'h2o':
                self.Tc = 647.3
                self.Pc = 220.483
                self.Vc = 0.0559
                self.MW = 18.015
                self.diam = np.nan
                self.N_carb = 0
                self.gi0 = -228700
                self.hi0 = -242000
                self.gi0_ast = np.nan
                self.hi0_ast = np.nan
                self.omega_born = np.nan
                self.cp = {
                           'a0': 3.8747,
                           'a1': 0.0231e-2,
                           'a2': 0.1269e-5,
                           'a3': -0.4321e-9
                           }
                self.aq_cp = {
                              'c1': np.nan,
                              'c2': np.nan}
                self.aq_v = {
                             'v1': np.nan,
                             'v2': np.nan,
                             'v3': np.nan,
                             'v4': np.nan}
                self.SRK_omega = 0.344
                self.SRK_S2 = -0.2018
                self.SRK_kij = {
                                'h2o': 0.0,
                                'ch4': 0.4965,
                                'co2': -0.07,
                                'n2': 0.5063,
                                'c2h6': 0.5975,
                                'c3h8': 0.5612
                                }
                self.HVdPM_S1_kappa = np.nan
                self.HVdPM_S2_kappa = np.nan
                self.HVdPM_S1_small = np.nan
                self.HVdPM_S2_small = np.nan
                self.HVdPM_S1_large = np.nan
                self.HVdPM_S2_large = np.nan
                self.HVdPM_kih_a = np.nan
                self.HVdPM_kih_sig = np.nan
                self.HVdPM_kih_epsk = np.nan
                self.ideal_HS1 = {
                                  'a1': np.nan,
                                  'a2': np.nan,
                                  'a3': np.nan,
                                  'a4': np.nan,
                                  'a5': np.nan,
                                  'a6': np.nan,
                                  'a7': np.nan,
                                  'a8': np.nan,
                                  'a9': np.nan,
                                  'a10': np.nan,
                                  'a11': np.nan,
                                  'a12': np.nan,
                                  'a13': np.nan,
                                  }
                self.ideal_HS2 = {
                                  'a1': np.nan,
                                  'a2': np.nan,
                                  'a3': np.nan,
                                  'a4': np.nan,
                                  'a5': np.nan,
                                  'a6': np.nan,
                                  'a7': np.nan,
                                  'a8': np.nan,
                                  'a9': np.nan,
                                  'a10': np.nan,
                                  'a12': np.nan,
                                  'a13': np.nan,
                                  'a14': np.nan,
                                  'a15': np.nan,
                                  'a16': np.nan,
                                  'a17': np.nan,
                                  'a18': np.nan,
                                  'a19': np.nan,
                                    }
            elif self.compname == 'ch4':
                self.Tc = 190.56
                self.Pc = 45.991
                self.Vc = 0.09865
                self.MW = 16.043
                self.diam = 4.247
                self.N_carb = 1
                self.gi0 = -50830
                self.hi0 = -74900
                self.gi0_ast = -34451
                self.hi0_ast = -87906
                self.omega_born = -133009
                self.cp = {
                           'a0': 2.3902,
                           'a1': 0.6039e-2,
                           'a2': 0.1525e-5,
                           'a3': -1.3234e-9
                           }
                self.aq_cp = {
                              'c1': 176.12,
                              'c2': 6310762}
                self.aq_v = {
                             'v1': 2.829,
                             'v2': 3651.8,
                             'v3': 9.7119,
                             'v4': -131365}
                self.SRK_omega = 0.0115
                self.SRK_S2 = -0.012223
                self.SRK_kij = {
                                'h2o': 0.4965,
                                'ch4': 0.0,
                                'co2': 0.0936,
                                'n2': 0.0291,
                                'c2h6': 0.0,
                                'c3h8': 0.0
                                }
                self.HVdPM_S1_kappa = 1e-5
                self.HVdPM_S2_kappa = 5e-5
                self.HVdPM_S1_small = 0.017668
                self.HVdPM_S2_small = 0.0020998
                self.HVdPM_S1_large = 0.010316
                self.HVdPM_S2_large = 0.011383
                self.HVdPM_kih_a = 0.3834
                self.HVdPM_kih_sig = 3.14393
                self.HVdPM_kih_epsk = 155.593
                self.ideal_HS1 = {
                                  'a1': 27.474169,
                                  'a2': -0.8587468,
                                  'a3': 0.0,
                                  'a4': 6604.6088,
                                  'a5': 50.8806,
                                  'a6': 1.57577,
                                  'a7': -1.4011858,
                                  'a8': 0.0,
                                  'a9': 0.0,
                                  'a10': 0.0,
                                  'a11': 0.0,
                                  'a12': 0.0,
                                  'a13': 0.0,
                                  }
                self.ideal_HS2 = {
                                  'a1': -0.45872,
                                  'a2': 0.0,
                                  'a3': 0.0,
                                  'a4': 31.6621,
                                  'a5': -3.4028,
                                  'a6': -7.7e-5,
                                  'a7': 0.0,
                                  'a8': 0.0,
                                  'a9': 1.8641,
                                  'a10': -0.78338,
                                  'a11': 0.0,
                                  'a12': 0.0,
                                  'a13': 0.0,
                                  'a14': -77.6955,
                                  'a15': 0.0,
                                  'a16': -2.3e-7,
                                  'a17': 6.1e-7,
                                  'a18': 0.0,
                                  'a19': 0.0,
                                    }
            elif self.compname == 'c2h6':
                self.Tc = 305.32
                self.Pc = 48.721
                self.Vc = 0.1455
                self.MW = 30.07
                self.diam = 5.076
                self.N_carb = 2
                self.SRK_omega = 0.0995
                self.SRK_S2 = -0.0124
                self.SRK_kij = {
                                'h2o': 0.5975,
                                'ch4': 0.0,
                                'co2': 0.132,
                                'n2': 0.0082,
                                'c2h6': 0.0,
                                'c3h8': 0.0
                                }
                self.HVdPM_S1_kappa = 1e-8
                self.HVdPM_S2_kappa = 1e-7
                self.HVdPM_S1_small = 0.0
                self.HVdPM_S2_small = 0.025154
                self.HVdPM_S1_large = 0.025154
                self.HVdPM_S2_large = 0.014973
                self.HVdPM_kih_a = 0.5651
                self.HVdPM_kih_sig = 3.24693
                self.HVdPM_kih_epsk = 188.181
                self.ideal_HS1 = {
                                  'a1': 14.81962,
                                  'a2': 6.813994,
                                  'a3': 0.0,
                                  'a4': 3463.9937,
                                  'a5': 2215.3,
                                  'a6': 0.0,
                                  'a7': 0.0,
                                  'a8': 0.0,
                                  'a9': 0.0,
                                  'a10': 0.0,
                                  'a11': 0.0,
                                  'a12': 0.0,
                                  'a13': 0.0,
                                  }
                self.ideal_HS2 = {
                                  'a1': 3.21799,
                                  'a2': 0.0,
                                  'a3': 0.0,
                                  'a4': -290.283,
                                  'a5': 181.2694,
                                  'a6': 0.0,
                                  'a7': 0.0,
                                  'a8': -1.89e-5,
                                  'a9': 1.882,
                                  'a10': -1.19703,
                                  'a11': -402.166,
                                  'a12': -4.897688,
                                  'a13': 0.0411205,
                                  'a14': -68.8018,
                                  'a15': 25.6306,
                                  'a16': 0.0,
                                  'a17': 0.0,
                                  'a18': 0.0,
                                  'a19': 0.0,
                                    }
            elif self.compname == 'c3h8':
                self.Tc = 369.83
                self.Pc = 42.481
                self.Vc = 0.2001
                self.MW = 44.097
                self.diam = 5.745
                self.N_carb = 3
                self.SRK_omega = 0.1523
                self.SRK_S2 = -0.0038
                self.SRK_kij = {
                                'h2o': 0.5612,
                                'ch4': 0.0,
                                'co2': 0.13,
                                'n2': 0.0865,
                                'c2h6': 0.0,
                                'c3h8': 0.0
                                }
                self.HVdPM_S1_kappa = 1e-7
                self.HVdPM_S2_kappa = 1e-6
                self.HVdPM_S1_small = 0.0
                self.HVdPM_S2_small = 0.0
                self.HVdPM_S1_large = 0.0
                self.HVdPM_S2_large = 0.025576
                self.HVdPM_kih_a = 0.6502
                self.HVdPM_kih_sig = 3.4167
                self.HVdPM_kih_epsk = 192.855
                self.ideal_HS1 = {
                                  'a1': 1e6,
                                  'a2': 1e6,
                                  'a3': 1e6,
                                  'a4': 1e6,
                                  'a5': 1e6,
                                  'a6': 1e6,
                                  'a7': 1e6,
                                  'a8': 1e6,
                                  'a9': 1e6,
                                  'a10': 1e6,
                                  'a11': 1e6,
                                  'a12': 1e6,
                                  'a13': 1e6,
                                  }
                self.ideal_HS2 = {
                                  'a1': 7.51966,
                                  'a2': 0.0,
                                  'a3': 0.0,
                                  'a4': 47.056,
                                  'a5': 0.0,
                                  'a6': 1.7e-5,
                                  'a7': 0.0007145,
                                  'a8': 0.0,
                                  'a9': 0.0,
                                  'a10': 0.12348,
                                  'a11': 79.34,
                                  'a12': 0.0,
                                  'a13': 0.0160778,
                                  'a14': 0.0,
                                  'a15': -14.684,
                                  'a16': 5.5e-6,
                                  'a17': 0.0,
                                  'a18': 0.0,
                                  'a19': 0.0,
                                    }
            elif self.compname == 'co2':
                self.Tc = 304.21
                self.Pc = 73.831
                self.Vc = 0.09396
                self.MW = 44.01
                self.diam = 4.603
                self.N_carb = 1
                self.gi0 = -394600
                self.hi0 = -393800
                self.gi0_ast = -385974
                self.hi0_ast = -413798
                self.omega_born = -8368
                self.cp = {
                           'a0': 2.6751,
                           'a1': 0.7188e-2,
                           'a2': -0.4208e-5,
                           'a3': 0.8977e-9
                           }
                self.aq_cp = {
                              'c1': 167.50,
                              'c2': 5304066}
                self.aq_v = {
                             'v1': 2.614,
                             'v2': 3125.9,
                             'v3': 11.7721,
                             'v4': -129198}
                self.SRK_omega = 0.2236
                self.SRK_S2 = -0.004474
                self.SRK_kij = {
                                'h2o': -0.07,
                                'ch4': 0.0936,
                                'co2': 0.0,
                                'n2': -0.0462,
                                'c2h6': 0.132,
                                'c3h8': 0.13
                                }
                self.HVdPM_S1_kappa = 1e-6
                self.HVdPM_S2_kappa = 1e-5
                self.HVdPM_S1_small = 0.0
                self.HVdPM_S2_small = 0.002758
                self.HVdPM_S1_large = 0.0058282
                self.HVdPM_S2_large = 0.012242
                self.HVdPM_kih_a = 0.6805
                self.HVdPM_kih_sig = 2.97638
                self.HVdPM_kih_epsk = 175.405
                self.ideal_HS1 = {
                                  'a1': 15.8336435,
                                  'a2': 3.119,
                                  'a3': 0.0,
                                  'a4': 3760.6324,
                                  'a5': 1090.27777,
                                  'a6': 0.0,
                                  'a7': 0.0,
                                  'a8': 0.0,
                                  'a9': 0.0,
                                  'a10': 0.0,
                                  'a11': 0.0,
                                  'a12': 0.0,
                                  'a13': 0.0,
                                  }
                self.ideal_HS2 = {
                                  'a1': 9.0242,
                                  'a2': 0.0,
                                  'a3': 0.0,
                                  'a4': -207.033,
                                  'a5': 0.0,
                                  'a6': 0.00067588,
                                  'a7': -0.006992,
                                  'a8': -0.0006079,
                                  'a9': -0.09026,
                                  'a10': 0.0,
                                  'a11': 0.0,
                                  'a12': 0.0,
                                  'a13': 0.0186833,
                                  'a14': 0.0,
                                  'a15': 0.0,
                                  'a16': 8.82e-5,
                                  'a17': 0.00778015,
                                  'a18': 0.0,
                                  'a19': 0.0,
                                    }
            elif self.compname == 'n2':
                self.Tc = 126.2
                self.Pc = 34.001
                self.Vc = 0.08919
                self.MW = 28.013
                self.diam = 4.177
                self.N_carb = 0.0
                self.gi0 = 0.0
                self.hi0 = 0.0
                self.gi0_ast = 18188
                self.hi0_ast = -10439
                self.omega_born = -145101
                self.cp = {
                           'a0': 3.4736,
                           'a1': -0.0189e-2,
                           'a2': 0.0971e-5,
                           'a3': -0.3453e-9
                           }
                self.aq_cp = {
                              'c1': 149.75,
                              'c2': 5046230}
                self.aq_v = {
                             'v1': 2.596,
                             'v2': 3083.0,
                             'v3': 11.9407,
                             'v4': -129018}
                self.SRK_omega = 0.0377
                self.SRK_S2 = -0.011016
                self.SRK_kij = {
                                'h2o': 0.5063,
                                'ch4': 0.0291,
                                'co2': -0.0462,
                                'n2': 0.0,
                                'c2h6': 0.0082,
                                'c3h8': 0.0862
                                }
                self.HVdPM_S1_kappa = 1.1e-5
                self.HVdPM_S2_kappa = 1.1e-5
                self.HVdPM_S1_small = 0.017377
                self.HVdPM_S2_small = 0.0020652
                self.HVdPM_S1_large = 0.0
                self.HVdPM_S2_large = 0.011295
                self.HVdPM_kih_a = 0.3526
                self.HVdPM_kih_sig = 3.13512
                self.HVdPM_kih_epsk = 127.426
                self.ideal_HS1 = {
                                  'a1': 173.2164,
                                  'a2': -0.5996,
                                  'a3': 0.0,
                                  'a4': 24751.6667,
                                  'a5': 0.0,
                                  'a6': 0.0,
                                  'a7': 0.0,
                                  'a8': 1.441,
                                  'a9': -37.0696,
                                  'a10': -0.0287444,
                                  'a11': -2.07e-5,
                                  'a12': 0.0,
                                  'a13': 0.0,
                                  }
                self.ideal_HS2 = {
                                  'a1': 1.78857,
                                  'a2': 0.0,
                                  'a3': -0.019667,
                                  'a4': -6.187,
                                  'a5': 0.0,
                                  'a6': 0.0,
                                  'a7': 0.0,
                                  'a8': 5.26e-5,
                                  'a9': 0.0,
                                  'a10': 0.0,
                                  'a12': 0.0,
                                  'a11': 0.0,
                                  'a13': 0.0,
                                  'a14': 0.0,
                                  'a15': 192.39,
                                  'a16': 0.0,
                                  'a17': 3.05e-5,
                                  'a18': 1.1e-7,
                                  'a19': 0.0,
                                    }

                
                
            

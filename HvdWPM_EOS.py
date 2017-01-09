#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 21:30:38 2017

@author: kdarnell
"""
import numpy as np
from scipy.integrate import quad

"""
    This is an equation of state (EOS) for water in the hydrate phase
    in the presence of multiple gases using the Ballard-modified
    van der Waals Platteeuw EOS.
"""
# Constants
R = 8.3144621  # universal gas constant
T_0 = 298.15
P_0 = 1.0
k = 1.3806488e-23


# Parent class for a series of slightly different equations of state
# --however, they will all be based on van der Waals-Platteeuw
# The parent class is desigend to reduce redundancy. Some functions
# will modified in the child class.
class HydrateEos(object):
    def __init__(self, compobjs, T, P, structure='s1'):
        self.description = (
            'Object for calculating fugacity of water in hydrate' +
            ' with a mixture of gases.'
        )
        # Set up arrays and matrices for future calculation
        self.Nc = len(compobjs)
        self.compobjs = compobjs
        self.gwbeta_RT_cons = np.zeros(1)
        self.volume_int = np.zeros(1)
        self.volume = np.zeros(1)
        self.activity = np.zeros(1)
        self.delta_mu_RT = np.zeros(1)
        self.gw0_RT = np.zeros(1)
        self.compnames = [c.compname for c in compobjs]
        self.h2oind = [ii for ii, name in enumerate(self.compnames)
                       if name == 'h2o'][0]
        self.Hs = HydrateStructure(structure)
        self.alt_fug_vec = np.zeros(self.Nc)
        self.Y_small = np.zeros(self.Nc)
        self.Y_large = np.zeros(self.Nc)

        # Transfer everything from the eos-specific dictionary to
        # the Hs object for use in the rest of the class
        # Note: 'eos_key' will be defined within sub-class
        for k, v in dict.items(self.Hs.eos[self.eos_key]):
            setattr(self.Hs, k, v)


    def make_constant_mats(self, compobjs, T, P):
        # Create all values that are invariant wrt T and P
        self.T = T
        self.P = P
        self.gw0_RT = compobjs[self.h2oind].gibbs_ideal(T, P)
        # Add additional stuff for specific EOS's here
        return self

    def langmuir_consts(self, compobjs, T, P):
        # Each child EOS will have a different way of calculating this.
        C_small = np.zeros(self.Nc)
        C_large = np.zeros(self.Nc)
        return C_small, C_large

    def calc_langmuir(self, C):
        Y = np.zeros(self.Nc)
        denominator = (1.0 + np.sum(C*self.alt_fug_vec))
        for ii, comp in enumerate(self.compobjs):
            if comp.compname != 'h2o':
                Y[ii] = C[ii]*self.alt_fug_vec[ii]/denominator
            else:
                Y[ii] = 0
        return Y

    def delta_mu_func(self, compobjs, T, P):
        delta_mu = (
            self.Hs.Nm['small']*np.log(1-np.sum(self.Y_small))
            + self.Hs.Nm['large']*np.log(1-np.sum(self.Y_large))
        )/self.Hs.Num_h2o
        return delta_mu

    def fugacity_calc(self, compobjs, T, P, x, alt_fug):
        pass

    def calc(self, compobjs, T, P, x, alt_fug):
        # Raise flag if components change.
        if compobjs != self.compobjs:
            print('Warning: Action not supported.' +
                  '\nComponents have changed. ' +
                  '\nPlease create a new fugacity object.')
            return None
        else:
            # Re-calculate constants if pressure or temperature changes.
            if self.T != T or self.P != P:
                self.make_constant_mats(compobjs, T, P)

        fug = self.fugacity_calc(compobjs, T, P, x, alt_fug)
        return fug


class HvdwpmEos(HydrateEos):
    cp = {'a0': 0.735409713*R,
          'a1': 1.4180551e-2*R,
          'a2': -1.72746e-5*R,
          'a3': 63.5104e-9*R}
    eos_key = 'hvdwpm'

    def __init__(self, compobjs, T, P, structure='s1'):
        # Inheret all prroperties from HydrateEos
        super().__init__(compobjs, T, P, structure)
        self.kappa = np.zeros(1)
        self.kappa0 = self.Hs.kappa
        self.a0_cubed = self.volume_func(self.Hs.a0_ast)
        self.kappa_vec = np.zeros(self.Nc)
        self.rep_sm_vec = np.zeros(self.Nc)
        self.rep_lg_vec = np.zeros(self.Nc)
        self.D_vec = np.zeros(self.Nc)
        self.a_new = self.Hs.a_norm

        self.make_constant_mats(compobjs, T, P)
        for ii, comp in enumerate(compobjs):
            self.kappa_vec[ii] = comp.HvdWPM[self.Hs.hydstruc]['kappa']
            self.rep_sm_vec[ii] = comp.HvdWPM[self.Hs.hydstruc]['rep']['small']
            self.rep_lg_vec[ii] = comp.HvdWPM[self.Hs.hydstruc]['rep']['large']
            self.D_vec[ii] = comp.diam

    def volume_func(self, a_param):
        if self.Hs.hydstruc != 'sH':
            v = 6.0221413e23/self.Hs.Num_h2o/1e24*(a_param)**3
        else:
            v = self.Hs.v0
        return v

    def make_constant_mats(self, compobjs, T, P):
        # Do standard stuff
        super().make_constant_mats(compobjs, T, P)

        self.gwbeta_RT = (
            self.Hs.gw_0beta/(R*T_0) 
            - (12*T*self.Hs.hw_0beta - 12*T_0*self.Hs.hw_0beta 
               + 12*T_0**2*self.cp['a0'] + 6*T_0**3*self.cp['a1'] 
               + 4*T_0**4*self.cp['a2'] + 3*T_0**5*self.cp['a3'] 
               - 12*T*T_0*self.cp['a0'] - 12*T*T_0**2*self.cp['a1'] 
               + 6*T**2*T_0*self.cp['a1'] - 6*T*T_0**3*self.cp['a2'] 
               + 2*T**3*T_0*self.cp['a2'] - 4*T*T_0**4*self.cp['a3'] 
               + T**4*T_0*self.cp['a3'] + 12*T*T_0*self.cp['a0']*np.log(T)
               - 12*T*T_0*self.cp['a0']*np.log(T_0))/(12*R*T*T_0)
            + (self.vol_int(T, P, self.a0_cubed, self.kappa0)
               - self.vol_int(T, P_0, self.a0_cubed, self.kappa0))*1e-1/(R*T)
        )
            
        self.PT_factor = np.exp(self.Hs.alf[1]/3*(T-T_0) 
                                + self.Hs.alf[2]/3*(T-T_0)**2 
                                + self.Hs.alf[3]/3*(T-T_0)**3 
                                - self.kappa0/3*(P-P_0))
        return self

    def vol_int(self, T, P, v, kappa):
        v_int = (
            v*np.exp(self.Hs.alf[1]*(T - T_0)+ self.Hs.alf[2]*(T - T_0)**2
                     + self.Hs.alf[3]*(T - T_0)**3 - kappa*(P - P_0))
        )/(-kappa)
        return v_int
        
    # Distortion of cage from occupancy
    def distortion_func(self):
        small_const = (
            (1 + self.Hs.etam['small']/self.Hs.Num_h2o)*self.Y_small
            / (1 + (self.Hs.etam['small']/self.Hs.Num_h2o)*self.Y_small)
        )
        if self.Nc>2:
            self.repulsive_small = (small_const*np.exp(
                self.D_vec - np.sum(self.Y_small*self.D_vec)
            ))
        else:
            self.repulsive_small = small_const
            
        self.repulsive_large = (
            (1 + self.Hs.etam['large']/self.Hs.Num_h2o)*self.Y_large
            / (1 + (self.Hs.etam['large']/self.Hs.Num_h2o)*self.Y_large)
        )
        return self

    # TODO: Determine when and where 'linear' vs. 'volumetric' kappas
    # are being used/provided.
    # TODO: Additionally, determine which kappa should be used in which 
    # places...the hydrate structure or the guest specific one??
    def kappa_func(self):
        if self.Nc>2:
            kappa = 3.0*np.sum(self.kappa_vec*self.Y_large)
        else:
            kappa = 3.0*np.sum(self.kappa_vec)
            
        return kappa

    def activity_func(self, T, P, v_H_0):
        kappa_wtavg = self.kappa_func()
        activity = (
            (v_H_0 - self.a0_cubed)/R*(self.Hs.a_fit/T_0
                                       + self.Hs.b_fit*(1/T - 1/T_0))
            + ((self.vol_int(T, P, v_H_0, kappa_wtavg)
                - self.vol_int(T, P, self.a0_cubed, self.kappa0))
              - (self.vol_int(T, P_0, v_H_0, kappa_wtavg)
                 - self.vol_int(T, P_0, self.a0_cubed, self.kappa0)))*1e-1/(R*T)
        )
        return activity


    def convert_a_param(self):
        self.distortion_func()
        self.a_new = (self.Hs.a0_ast
             + (self.Hs.Nm['small']
                * np.sum(self.repulsive_small*self.rep_sm_vec))
             + (self.Hs.Nm['large']
                *np.sum(self.repulsive_large*self.rep_lg_vec)))
        return self

    def delta_func(self, N, Rn, aj, r):
        delta = ((1.0 - r/Rn - aj/Rn)**(-N) - (1.0 + r/Rn - aj/Rn)**(-N))/N
        return delta

    def w_func(self, zn, eps_k, r, Rn, sigma, aj):
        if self.Hs.hydstruc == 's1':
            w = (2*zn*eps_k*(sigma**12/(Rn**11*r)
                             * (self.delta_func(10, Rn, aj, r)
                                + (aj/Rn)*self.delta_func(11, Rn, aj, r))
                             - sigma**6/(Rn**5*r)
                             * (self.delta_func(4, Rn, aj, r)
                                + (aj/Rn)*self.delta_func(5, Rn, aj, r))))
        return w

    def integrand_sm(self, r, R1, R2, z1, z2, eps_k, sigma, aj, T):
        output = r**2*np.exp((-1.0/T)
                             * (self.w_func(z1, eps_k, r, R1, sigma, aj)
                             + self.w_func(z2, eps_k, r, R2, sigma, aj)))
        return output

    def integrand_lg(self, r, R1, R2, R3, R4, z1, z2, z3, z4, 
                     eps_k, sigma, aj, T):
        output = r**2*np.exp((-1.0/T)
                             * (self.w_func(z1, eps_k, r, R1, sigma, aj)
                             + self.w_func(z2, eps_k, r, R2, sigma, aj)
                             + self.w_func(z3, eps_k, r, R3, sigma, aj)
                             + self.w_func(z4, eps_k, r, R4, sigma, aj)))
        return output

    def compute_integral_constants(self):
        a_factor = (self.a_new/self.Hs.a_norm)*self.PT_factor
        self.R1_sm = self.Hs.R['sm'][1]*a_factor
        self.R2_sm = self.Hs.R['sm'][2]*a_factor
        self.R1_lg = self.Hs.R['lg'][1]*a_factor
        self.R2_lg = self.Hs.R['lg'][2]*a_factor
        self.R3_lg = self.Hs.R['lg'][3]*a_factor
        self.R4_lg = self.Hs.R['lg'][4]*a_factor
        return self
        
    def langmuir_consts(self, compobjs, T, P):
        self.compute_integral_constants()
        C_small = np.zeros(self.Nc)
        C_large = np.zeros(self.Nc)
        C_const = 1e-10**3*4*np.pi/(k*T)*1e5
        for ii, comp in enumerate(compobjs):
            if comp.compname != 'h2o':
                
                small_int = quad(self.integrand_sm,
                                 0,
                                 self.R1_sm - comp.HvdWPM['kih']['a'],
                                 args=(self.R1_sm,
                                       self.R2_sm,
                                       self.Hs.z['sm'][1],
                                       self.Hs.z['sm'][2],
                                       comp.HvdWPM['kih']['epsk'],
                                       comp.HvdWPM['kih']['sig'],
                                       comp.HvdWPM['kih']['a'],
                                       T,))
                large_int = quad(self.integrand_lg,
                                 0,
                                 self.R2_lg - comp.HvdWPM['kih']['a'],
                                 args=(self.R1_lg,
                                       self.R2_lg,
                                       self.R3_lg,
                                       self.R4_lg,
                                       self.Hs.z['lg'][1],
                                       self.Hs.z['lg'][2],
                                       self.Hs.z['lg'][3],
                                       self.Hs.z['lg'][4],
                                       comp.HvdWPM['kih']['epsk'],
                                       comp.HvdWPM['kih']['sig'],
                                       comp.HvdWPM['kih']['a'],
                                       T,))
                C_small[ii] = C_const*small_int[0]
                C_large[ii] = C_const*large_int[0]
            else:
                C_small[ii] = 0
                C_large[ii] = 0

        return C_small, C_large
        
        
    def iterate_function(self, compobjs, T, P):
        self.convert_a_param()
        C_small, C_large = self.langmuir_consts(compobjs, T, P)
        self.Y_small = self.calc_langmuir(C_small)
        self.Y_large = self.calc_langmuir(C_large) 
        return C_small, C_large
        
    def find_nonlinear_C(self, compobjs, T, P):
        error = 1e6
        TOL = 1e-3
        C_small = np.zeros(self.Nc)
        C_large = np.zeros(self.Nc)
        while error>TOL:
            C_small_new, C_large_new = self.iterate_function(compobjs, T, P)
            error = 0.0
            for ii, comp in enumerate(compobjs):
                if comp.compname != 'h2o':
                    error += (abs(C_small_new[ii] 
                                  - C_small[ii])/C_small_new[ii] 
                              + abs(C_large_new[ii] 
                                    - C_large[ii])/C_large_new[ii])
            C_small = C_small_new
            C_large = C_large_new
        return self


    def fugacity_calc(self, compobjs, T, P, x, alt_fug):
        alt_fug[self.h2oind] = 0.0
        self.alt_fug_vec = alt_fug
        self.find_nonlinear_C(compobjs, T, P)
        delta_mu_RT = self.delta_mu_func(compobjs, T, P)
        v_H = self.volume_func(self.a_new)
        activity = self.activity_func(T, P, v_H)
        mu_H_RT = self.gwbeta_RT + activity + delta_mu_RT
        fug = np.exp(mu_H_RT - self.gw0_RT)
        return fug



# Properties of each hydrate structure necessary for further calculation.
class HydrateStructure(object):
    # Possible aliases of the hydrate structures
    menu = {'s1': ('s1', '1', 1, 'si', 'i', 'one'),
            's2': ('s2', '2', 2, 'sii', 'ii', 'two'),
            'sh': ('sh', 'h')}
    def __init__(self, hyd_type):
        self.description = ('Object for holding properties for each type of'
                            + ' hydrate structure (1,2,H)')

        if str(hyd_type).lower() in self.menu['s1']:
            self.hydstruc = 's1'
        elif str(hyd_type).lower() in self.menu['s2']:
            self.hydstruc = 's2'
        elif str(hyd_type).lower() in self.menu['sh']:
            self.hydstruc = 'sH'
        else:
            raise RuntimeError(hyd_type + ' is not a supported hydrate '
                               + 'structure!! \nConsult '
                               + '"HydrateStructure.menu" '
                               + 'attribute for valid structures.')

        if self.hydstruc == 's1':
            self.eos = {'hvdwpm': {'v0': 22.7712,
                                   'kappa': 3e-5,
                                   'a0_ast': 11.99245,
                                   'gw_0beta': -235537.85,
                                   'hw_0beta': -291758.77,
                                   'a_fit': 25.74,
                                   'b_fit': -481.32,
                                   'alf': {1: 3.38496e-4,
                                           2: 5.40099e-7,
                                           3: -4.76946e-11},
                                   'a_norm': 12.03,
                                   'R': {'sm': {1: 3.83,
                                                2: 3.96},
                                         'lg': {1: 4.47,
                                                2: 4.06,
                                                3: 4.645,
                                                4: 4.25}},
                                   'z': {'sm': {1: 8,
                                                2: 12},
                                         'lg': {1: 8,
                                                2: 8,
                                                3: 4,
                                                4: 4}}}}
            self.Nm = {'small': 2, 'large': 6}
            self.etam = {'small': 20, 'large': 24}
            self.Num_h2o = 46
        elif self.hydstruc == 's2':
            self.eos = {'hvdwpm': {'v0': 22.9456,
                                   'kappa': 3e-6,
                                   'a0_ast': 17.10000,
                                   'gw_0beta': -235627.53,
                                   'hw_0beta': -292044.10,
                                   'a_fit': 260,
                                   'b_fit': -68.64,
                                   'alf': {1: 2.029776e-4,
                                           2: 1.851168e-7,
                                           3: -1.879455e-10}}}

            self.Nm = {'small': 16, 'large': 8}
            self.etam = {'small': 20, 'large': 28}
            self.Num_h2o = 136
        elif self.hydstruc == 'sH':
            self.eos = {'hvdwpm': {'v0': 24.2126,
                                   'kappa': 3e-7,
                                   'a0_ast': 11.09826,
                                   'gw_0beta': -2355491.02,
                                   'hw_0beta': -291979.26,
                                   'a_fit': 0,
                                   'b_fit': 0,
                                   'alf': {1: 3.575490e-4,
                                           2: 6.294390e-7,
                                           3: 0}}}
            self.Nm = {'small': 3, 'large': 1}
            self.etam = {'small': 20, 'large': 36}
            self.Num_h2o = 46

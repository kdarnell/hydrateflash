#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 21:30:38 2017

@author: kdarnell
"""
import numpy as np

"""
    This is an equation of state (EOS) for water in the hydrate phase
    in the presence of multiple gases using the Ballard-modified 
    van der Waals Platteeuw EOS.
"""
# Constants
R = 8.3144621  # universal gas constant
T_0 = 298.15
P_0 = 1
cp = {'a0': 0.735409713*R,
      'a1': 1.4180551e-2*R,
      'a2': -1.72746e-5*R,
      'a3': 63.5104e-9*R}

class HydrateEos(object):
    def __init__(self, compobjs, T, P, structure='s1'):
        self.description = (
            'Object for calculating fugacity of water in hydrate' +
            ' with a mixture of gases.'
        )
        # Set up arrays and matrices for future calculation
        self.Nc = len(compobjs)
        self.compobjs = compobjs
        self.gwbeta_RT = np.zeros(1)
        self.activity = np.zeros(1)
        self.delta_mu_RT = np.zeros(1)
        self.gw0_RT = np.zeros(1)
        self.compnames = [c.compname for c in compobjs]
        self.h2oind = [ii for ii, name in enumerate(self.compnames)
                       if name == 'h2o'][0]
        self.Hs = HydrateStructure(structure)
        self.make_constant_mats(compobjs, T, P)
        
    def make_constant_mats(self, compobjs, T, P):
        # Create all values that are invariant wrt T and P
        self.T = T
        self.P = P
        self.gw0_RT = compobjs[self.h2oind].gibbs_ideal(T, P)
        
        # Add additional stuff for specific EOS's here

        return self

    def Langmuir_constants(self, compobjs, T, P, x):
        return self
        
        
    def calc(self, compobjs, T, P, x):
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
                    
        mu_H_RT = (self.gwbeta_RT + self.activity 
                   + self.delta_mu_RT)
        fug = np.exp(mu_H_RT - self.gw0_RT)
        return fug
        
class HydrateStructure(object):
    # Possible aliases of the hydrate structures
    menu = {'s1': ('s1', '1', 'si', 'i', 'one'),
            's2': ('s2', '2', 'sii', 'ii', 'two'),
            'sh': ('sh', 'h')}
    def __init__(self, hyd_type):
        self.description = ('Object for holding properties for each type of'
                            + ' hydrate structure (1,2,H)')

        if hyd_type.lower() in self.menu['s1']:
            self.hydstruc = 's1'
        elif hyd_type.lower() in self.menu['s2']:
            self.hydstruc = 's2'
        elif hyd_type.lower() in self.menu['sh']:
            self.hydstruc = 'sH'
        else:
            raise RuntimeError(hyd_type + ' is not a supported hydrate '
                               + 'structure!! \nConsult '
                               + '"HydrateStructure.menu" '
                               + 'attribute for valid structures.')
           
        if self.hydstruc == 's1':
            self.v0 = 22.7712
            self.kappa = 3e-5
            self.a0_ast = 11.99245
            self.gw_0beta = -235537.85
            self.hw_0beta = -291758.77
            self.a_fit = 25.74
            self.b_fit = -481.32
            self.Nm = {'small': 2, 'large': 6}
            self.etam = {'small': 20, 'large': 24}
            self.Num_h2o = 46
            self.alf = {'1': 3.38496e-4, '2': 5.40099e-7, '3': -4.76946e-11}
        elif self.hydstruc == 's2':
            self.v0 = 22.9456
            self.kappa = 3e-6
            self.a0_ast = 17.10000
            self.gw_0beta = -235627.53
            self.hw_0beta = -292044.10
            self.a_fit = 260
            self.b_fit = -68.64
            self.Nm = {'small': 16, 'large': 8}
            self.etam = {'small': 20, 'large': 28}
            self.Num_h2o = 136
            self.alf = {'1': 2.029776e-4, '2': 1.851168e-7, '3': -1.879455e-10}
        elif self.hydstruc == 'sH':
            self.v0 = 24.2126
            self.kappa = 3e-7
            self.a0_ast = 11.09826
            self.gw_0beta = -2355491.02
            self.hw_0beta = -291979.26
            self.a_fit = 0
            self.b_fit = 0
            self.Nm = {'small': 3, 'large': 1}
            self.etam = {'small': 20, 'large': 36}
            self.Num_h2o = 46
            self.alf = {'1': 3.575490e-4, '2': 6.294390e-7, '3': 0} 
#
#% Calculate cage occupancies at T_0,P_0
#
#[~,Y_small,Y_large] = calc_Langmuir(comps,datatable,kij_param,T_0,P_0,phases,x,struc_typ,a0_ast,key);
#
#% Integrated solid volume
#vol_wbeta_int = @(P,v,kappa,T) (v*exp(alf1*(T - T_0) + alf2*(T - T_0)**2 + ...
#    alf3*(T - T_0)**3 - kappa*(P - P_0))/(-kappa));
#
#% Volume dependence of guest
#if length(gas_components)==1
#    func_small = @ (argY,argeta) ((1 + argeta./NH2O).*argY./(1 + (argeta./NH2O).*argY));%.*exp(D_array - sum(argY.*D_array)); %Not sure what to do about this...perhaps it should only be valid for more than one guest!!
#else
#    func_small = @ (argY,argeta) ((1 + argeta./NH2O).*argY./(1 + (argeta./NH2O).*argY)).*exp(D_array - sum(argY.*D_array));
#end
#func_large = @ (argY,argeta) ((1 + argeta./NH2O).*argY./(1 + (argeta./NH2O).*argY));
#
#% Lattice parameter of "real" (non-empty) hydrate
#a_param = a0_ast + Nm_small*sum(func_small(Y_small,etam_small).*delr_small) + ...
#    Nm_large*sum(func_large(Y_large,etam_large).*delr_large);
#
#% Volume dependence of guest with real lattice parameter
#[~,Y_small,Y_large] = calc_Langmuir(comps,datatable,kij_param,...
#    T_0,P_0,phases,x,struc_typ,a_param,key);
#
#a_param = a0_ast + Nm_small*sum(func_small(Y_small,etam_small).*delr_small) + ...
#    Nm_large*sum(func_large(Y_large,etam_large).*delr_large);
#
#% Perhaps I would need to actually iterate the above until the lattice
#% parameter reached convergence
#
#% Convert lattice parameter to cm**3/mol
#v_H_0 = 6.0221413e23/NH2O/1e24*(a_param).**3; 
#if strcmp(struc_typ,'H')
#    v_H_0 = v_0;
#end
#
#
#% Equation 3.47 of Ballard Thesis
#% Standard hydrate:
#a0_cubed = a0_ast**3*6.0221413e23/NH2O/1e24;
#g_wbeta_RT = gw_0beta/(R*T_0) - (12*T*hw_0beta - 12*T_0*hw_0beta + 12*T_0**2*cp_a0 + 6*T_0**3*cp_a1 + ...
#    4*T_0**4*cp_a2 + 3*T_0**5*cp_a3 - 12*T*T_0*cp_a0 - 12*T*T_0**2*cp_a1 + 6*T**2*T_0*cp_a1 - ...
#    6*T*T_0**3*cp_a2 + 2*T**3*T_0*cp_a2 - 4*T*T_0**4*cp_a3 + T**4*T_0*cp_a3 + 12*T*T_0*cp_a0*log(T) - ...
#    12*T*T_0*cp_a0*log(T_0))/(12*R*T*T_0) + ...
#    (1/(R*T))*1e-1*(vol_wbeta_int(P,a0_cubed,kappa0,T) - vol_wbeta_int(P_0,a0_cubed,kappa0,T));
#
#% Determine cage occupancty at T,P
#[delta_mu,Y_small,Y_large] = calc_Langmuir(comps,datatable,kij_param,...
#    T,P,phases,x,struc_typ,a_param,key);
#
#if length(gas_components)==1
#    kappa_real = kappa_array;
#else
#    kappa_real = sum(kappa_array.*Y_large);
#end
#
#% Determine activity as deviation from standard volume
#% I'm still not sure if I should be taking v_beta as v_0 or a0_cubed...
#% The text is very confusing in this regard.
#activity = (v_H_0 - a0_cubed)/R*(a/T_0 + b*(1/T - 1/T_0)) + ...
#    (1/(R*T))*1e-1*((vol_wbeta_int(P,v_H_0,kappa_real,T) - vol_wbeta_int(P,a0_cubed,kappa0,T)) - ...
#    (vol_wbeta_int(P_0,v_H_0,kappa_real,T) - vol_wbeta_int(P_0,a0_cubed,kappa0,T)));
#
#v_s_aparam = vol_wbeta_int(P,v_H_0,kappa_real,T)*-kappa_real;
#v_s_a0 = vol_wbeta_int(P,a0_cubed,kappa_real,T)*-kappa_real;
#v_s_v0 = vol_wbeta_int(P,a0_cubed,kappa_real,T)*-kappa_real;
#
#
#
#
#% Determine chemical potential
#mu_H_RT = g_wbeta_RT + activity + delta_mu;
#%
#produce_aqueous_factors;
#h2o=s.h2o;
#% Gibbs Free Energy in ideal gas phase
#gw_0_RT = h2o.g_io0/(R*T_0) - (12*T*h2o.h_io0 - 12*T_0*h2o.h_io0 + 12*T_0**2*h2o.cp_a0 + 6*T_0**3*h2o.cp_a1 + ...
#    4*T_0**4*h2o.cp_a2 + 3*T_0**5*h2o.cp_a3 - 12*T*T_0*h2o.cp_a0 - 12*T*T_0**2*h2o.cp_a1 + 6*T**2*T_0*h2o.cp_a1 - ...
#    6*T*T_0**3*h2o.cp_a2 + 2*T**3*T_0*h2o.cp_a2 - 4*T*T_0**4*h2o.cp_a3 + T**4*T_0*h2o.cp_a3 + 12*T*T_0*h2o.cp_a0*log(T) - ...
#    12*T*T_0*h2o.cp_a0*log(T_0))/(12*R*T*T_0);
#fug = exp(mu_H_RT - gw_0_RT);
#
#end

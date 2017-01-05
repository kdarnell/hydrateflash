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
# Possible aliases of the 
s1alias = ('s1', '1', 'sI', 'I', 'one')
s2alias = ('s2', '2', 'sII', 'II', 'two')
sHalias = ('sH', 'H')

# Constants
R = 8.3144621  # universal gas constant
T_0 = 298.15
P_0 = 1
cp = {'a0': 0.735409713*R,
      'a1': 1.4180551e-2*R,
      'a2': -1.72746e-5*R,
      'a3': 63.5104e-9*R}

class VdwpmEOSEos(object):
    def __init__(self, compobjs, T, P, structure='S1'):
        self.description = (
            'Object for calculating fugacity of water in hydrate' +
            'with a mixture of gases using the van der Waals' +
            'Platteeuw-Modifed equation of state.'
        )
        # Set up arrays and matrices for future calculation
        self.Nc = len(compobjs)
        self.compobjs = compobjs
        Hs = HydStructure(structure)

#        self.S1_vec = np.zeros(self.Nc)
#        self.kij_mat = np.zeros([self.Nc, self.Nc])
#        self.a_vec = np.zeros(self.Nc)
#        self.b_vec = np.zeros(self.Nc)
#        self.a_mat = np.zeros([self.Nc, self.Nc])
#        self.alf_vec = np.zeros(self.Nc)
#        self.Tr_vec = np.zeros(self.Nc)
#        self.Pr_vec = np.zeros(self.Nc)
#
#        # Assuming pressure and temperature won't change, compute terms that
#        # are not functions of compostion.
#        self.make_constant_mats(compobjs, T, P)
#
#    # Define all constants wrt to composition.
#    def make_constant_mats(self, compobjs, T, P):
#        for ii, comp in enumerate(compobjs):
#            self.T = T
#            self.P = P
#            self.Tr_vec[ii] = T/comp.Tc
#            self.Pr_vec[ii] = P/comp.Pc
#            self.S1_vec[ii] = (0.48508 + 1.55171*comp.SRK['omega']
#                               - 0.15613*comp.SRK['omega']**2)
#            self.alf_vec[ii] = (
#                (1.0 + self.S1_vec[ii]*(1.0 - np.sqrt(self.Tr_vec[ii]))
#                 + comp.SRK['S2']*(1.0 - np.sqrt(self.Tr_vec[ii]))
#                 / np.sqrt(self.Tr_vec[ii]))**2
#            )
#            self.a_vec[ii] = 0.42747*R**2*comp.Tc**2 / comp.Pc
#            self.b_vec[ii] = 0.08664*R*comp.Tc / comp.Pc
#
#        for ii, compouter in enumerate(compobjs):
#            for jj, compinner in enumerate(compobjs):
#                self.kij_mat[ii, jj] = compouter.SRK['kij'][compinner.compname]
#                self.a_mat[ii, jj] = (
#                    (1 - self.kij_mat[ii, jj])
#                     * np.sqrt(self.alf_vec[ii]*self.a_vec[ii]
#                     * self.alf_vec[jj]*self.a_vec[jj])
#                )
#
#    # Weighted-sum of b
#    def b_tot(self, x):
#        b = np.sum(self.b_vec*x)
#        return b
#
#    # Weighted-sum of a
#    def a_tot(self, x):
#        a = 0.0
#        for ii in range(len(x)):
#            for jj in range(len(x)):
#                a += x[ii]*x[jj]*self.a_mat[ii, jj]
#        return a
#
#    # Function for fugacity calculation in terms of pre-computed values,
#    # composition, and the Z-factor calculated in "calc".
#    def fugacity(self, x, Z):
#        fug = (x*self.P
#               * np.exp((self.b_frac)*(Z - 1.0) - np.log(Z - self.B)
#                        - self.A/self.B*(2.0*self.a_frac - self.b_frac)
#                        * np.log(1.0 + self.B/Z))
#               )
#        return fug
#
#    # Main calculation that will call "fugacity". Option to specify phase.
#    def calc(self, compobjs, T, P, x, phase='general'):
#        # Raise flag if components change.
#        if compobjs != self.compobjs:
#            print('Warning: Action not supported.' +
#                  '\nComponents have changed. ' +
#                  '\nPlease create a new fugacity object.')
#            return None
#        else:
#            # Re-calculate constants if pressure or temperature changes.
#            if self.T != T or self.P != P:
#                self.make_constant_mats(compobjs, T, P)
#
#            self.A = self.a_tot(x)*P / (R**2 * T**2)
#            self.B = self.b_tot(x)*P / (R*T)
#            coeffs = [1, -1, self.A - self.B - self.B**2, -(self.A*self.B)]
#            Z = np.roots(coeffs)
#            self.b_frac = self.b_vec/self.b_tot(x)
#            self.a_x_sum = np.matmul(self.a_mat, x)
#            self.a_frac = self.a_x_sum/self.a_tot(x)
#
#            if np.isreal(Z).all():
#                if phase.lower() in liquidalias:
#                    fug = self.fugacity(x, Z.min())
#                elif phase.lower() in vaporalias or phase.lower() == 'general':
#                    fug = self.fugacity(x, Z.max())
#            elif np.isreal(Z).any():
#                # There should actually only be one real number if any
#                # imaginary roots exists, so the np.max() is redundant.
#                fug = self.fugacity(x, np.real(np.max(Z[np.isreal(Z)])))
#            else:
#                fug = np.nan
#                print('Something is wrong.' +
#                      '\nSolver returned imaginary numbers')
#        return fug
        
        
class HydStructure(object):
    def __init__(self, hyd_type):
    self.description = ('Object for holding properties for each type of'
                        + ' hydrate structure (1,2,H)')
    
    self.supportflag = False
        if hyd_type.lower() in s1alias:
            self.hydstruc = 'S1'
        elif hyd_type.lower() in s2alias:
            self.hydstruc = 'S2'
        elif hyd_type.lower() in sHalias:
            self.hydstruc = 'SH'
        else:
            self.supporflag = True
            self.hydstruc = None
            print('Warning: ' + hyd_tyep +
                  ' hydrate structure is not currently supported!!')
           
    if not self.supportflag:
        if self.hydstruc == 'S1':
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
        elif self.hydstruc == 'S2':
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
        elif self.hydstruc == 'SH':
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
#vol_wbeta_int = @(P,v,kappa,T) (v*exp(alf1*(T - T_0) + alf2*(T - T_0)^2 + ...
#    alf3*(T - T_0)^3 - kappa*(P - P_0))/(-kappa));
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
#% Convert lattice parameter to cm^3/mol
#v_H_0 = 6.0221413e23/NH2O/1e24*(a_param).^3; 
#if strcmp(struc_typ,'H')
#    v_H_0 = v_0;
#end
#
#
#% Equation 3.47 of Ballard Thesis
#% Standard hydrate:
#a0_cubed = a0_ast^3*6.0221413e23/NH2O/1e24;
#g_wbeta_RT = gw_0beta/(R*T_0) - (12*T*hw_0beta - 12*T_0*hw_0beta + 12*T_0^2*cp_a0 + 6*T_0^3*cp_a1 + ...
#    4*T_0^4*cp_a2 + 3*T_0^5*cp_a3 - 12*T*T_0*cp_a0 - 12*T*T_0^2*cp_a1 + 6*T^2*T_0*cp_a1 - ...
#    6*T*T_0^3*cp_a2 + 2*T^3*T_0*cp_a2 - 4*T*T_0^4*cp_a3 + T^4*T_0*cp_a3 + 12*T*T_0*cp_a0*log(T) - ...
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
#H20=s.H20;
#% Gibbs Free Energy in ideal gas phase
#gw_0_RT = H20.g_io0/(R*T_0) - (12*T*H20.h_io0 - 12*T_0*H20.h_io0 + 12*T_0^2*H20.cp_a0 + 6*T_0^3*H20.cp_a1 + ...
#    4*T_0^4*H20.cp_a2 + 3*T_0^5*H20.cp_a3 - 12*T*T_0*H20.cp_a0 - 12*T*T_0^2*H20.cp_a1 + 6*T^2*T_0*H20.cp_a1 - ...
#    6*T*T_0^3*H20.cp_a2 + 2*T^3*T_0*H20.cp_a2 - 4*T*T_0^4*H20.cp_a3 + T^4*T_0*H20.cp_a3 + 12*T*T_0*H20.cp_a0*log(T) - ...
#    12*T*T_0*H20.cp_a0*log(T_0))/(12*R*T*T_0);
#fug = exp(mu_H_RT - gw_0_RT);
#
#end

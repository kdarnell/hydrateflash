#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:31:01 2017

@author: kdarnell
"""

import numpy as np

# Constants
R = 8.3144621  # Gas constant in J/mol-K
T_0 = 298.15  # Reference temperature in K
P_0 = 1  # Reference pressure in bar

# Constants for solute model
theta = 228.0
psi = 2600
s10 = 243.9576
s11 = -0.7520846
s12 = 6.60648e-4
s20 = 0.039037
s21 = -2.12309e-4
s22 = 3.18021e-7
s30 = -1.0126e-5
s31 = 6.04961e-8
s32 = -9.3334e-11

# Constants produced by symbolic integration for h_ast function
f1 = 73786976294838206464
f2 = 8151985141053725
f3 = 3249460376862603
f4 = 9444732965739290427392
f5 = 4722366482869645213696
f6 = 1043454098054876800

# Constants for pure water
gw_pure = -237129
hw_pure = -285830
cp_a0 = R*8.712
cp_a1 = R*1e-2*0.125
cp_a2 = R*1e-5*-0.018
cp_a3 = 0
a10 = 31.1251
a11 = -1.14154e-1
a12 = 3.10034e-4
a13 = -2.48318e-7
a20 = -2.46176e-2
a21 = 2.15663e-4
a22 = -6.48160e-7
a23 = 6.47521e-10
a30 = 8.69425e-6
a31 = -7.96939e-8
a32 = 2.45391e-10
a33 = -2.51773e-13
a40 = -6.03348e-10
a41 = 5.57791e-12
a42 = -1.72577e-14
a43 = 1.77978e-17


class HegBromEos(object):
    def __init__(self, compobjs, T, P):
        self.description = (
            'Object for calculating fugacity of aqueous phase mixtures ' +
            'using a modified Helgeson equation of state with a ' +
            'Bromley activity model.'
        )
        
        self.compnames = [c.compname for c in compobjs]
        try:
            self.h2oind = [ii for ii, name in enumerate(self.compnames)
                           if name == 'h2o'][0]
        except IndexError: 
            raise RuntimeError(
                'Aqueous EOS requires water to be present!'
                 + '\nPlease provide water in your component list.')
            

        self.Nc = len(compobjs)
        self.g_io_vec = np.zeros(self.Nc)
        self.molality_vec = np.zeros(self.Nc)
        self.activity_vec = np.zeros(self.Nc)
        self.gamma_P1_vec = np.zeros(self.Nc)
        self.mu_ik_RT_cons = np.zeros(self.Nc)
        self.compobjs = compobjs


        # Assuming pressure and temperature won't change, compute terms that
        # are not functions of compostion.
        self.make_constant_mats(compobjs, T, P)

    # Define all constants wrt to composition.
    def make_constant_mats(self, compobjs, T, P):
        self.T = T
        self.P = P
        for ii, comp in enumerate(compobjs):
            self.g_io_vec[ii] = comp.gibbs_ideal(T, P)

            if comp.compname != 'h2o':
                c1 = comp.AqHB['cp']['c1']
                c2 = comp.AqHB['cp']['c2']
                omega = comp.AqHB['omega_born']
                h_io_ast = comp.h_io_ast

                # Output of symbolic integration.
                h_ast = (np.log(T)*(f1*c1 + f2*omega)/(f1*R)
                         - (np.log(T_0)*(f1*c1 + f2*omega))/(f1*R)
                         - (f3*T*omega)/(f4*R) + (f3*T_0*omega)/(f4*R)
                         - (f5*T_0*c2
                            - T_0*(f4*c2 + f4*T_0*h_io_ast - f4*T_0**2*c1
                                   - f6*T_0**2*omega + f3*T_0**3*omega)
                            )/(f4*R*T_0**3)
                         + (f5*T_0*c2
                            - T*(f4*c2 + f4*T_0*h_io_ast - f4*T_0**2*c1
                                 - f6*T_0**2*omega + f3*T_0**3*omega)
                            )/(f4*R*T**2*T_0))

                solute_vol_Pfunc = self.solutevol_integrated(comp, T)
                self.mu_ik_RT_cons[ii] = (
                    comp.g_io_ast/(R*T_0) - h_ast
                    + solute_vol_Pfunc(P) - solute_vol_Pfunc(P_0)
                )

                if comp.compname == 'co2':
                    self.gamma_P1_vec[ii] = (0.107 - 4.5e-4*T)
            else:
                self.mu_ik_RT_cons[ii] = (
                    gw_pure/(R*T_0)
                    - (12*T*hw_pure - 12*T_0*hw_pure + 12*T_0**2*cp_a0
                       + 6*T_0**3*cp_a1 + 4*T_0**4*cp_a2 + 3*T_0**5*cp_a3
                       - 12*T*T_0*cp_a0 - 12*T*T_0**2*cp_a1
                       + 6*T**2*T_0*cp_a1 - 6*T*T_0**3*cp_a2
                       + 2*T**3*T_0*cp_a2 - 4*T*T_0**4*cp_a3
                       + T**4*T_0*cp_a3 + 12*T*T_0*cp_a0*np.log(T)
                       - 12*T*T_0*cp_a0*np.log(T_0))/(12*R*T*T_0)
                    + (self.purewatervol_integrated(P, T)
                       - self.purewatervol_integrated(P_0, T))*1e-1/(R*T)
                )

    def purewatervol_integrated(self, P, T):
        v_w = ((a10*P + a20*P**2/2 + a30*P**3/3 + a40*P**4/4)
               + (a11*P + a21*P**2/2 + a31*P**3/3 + a41*P**4/4)*T
               + (a12*P + a22*P**2/2 + a32*P**3/3 + a42*P**4/4)*T**2
               + (a13*P + a23*P**2/2 + a33*P**3/3 + a43*P**4/4)*T**3)
        return v_w
        

    def purewatervol(self, P, T):
        v_w = ((a10 + a20*P + a30*P**2 + a40*P**3)
               + (a11 + a21*P + a31*P**2 + a41*P**3)*T
               + (a12 + a22*P + a32*P**2 + a42*P**3)*T**2
               + (a13 + a23*P + a33*P**2 + a43*P**3)*T**3)
        return v_w


    def solutevol_integrated(self, comp, T):
        omega = comp.AqHB['omega_born']
        v1 = comp.AqHB['v']['v1']
        v2 = comp.AqHB['v']['v2']
        v3 = comp.AqHB['v']['v3']
        v4 = comp.AqHB['v']['v4']
        tau = ((5.0/6.0)*T - theta)/(1.0 + np.exp((T - 273.15)/5.0))

        v_ast_P = (
            lambda P: (
                       v1*P + v2*np.log(psi + P)
                       + (v3*P + v4*np.log(psi + P))*(1.0/(T - theta - tau))
                       + omega/self.dielectric_const(T, P))/(R*T)

        )
        return v_ast_P

    def dielectric_const(self, T, P):
        eps = ((s10 + s20*P + s30*P**2)
               + (s11 + s21*P + s31*P**2)*T
               + (s12 + s22*P + s32*P**2)*T**2)
        return eps

    def fugacity(self, compobjs, x):
        xw = x[self.h2oind]

        for ii, comp in enumerate(compobjs):
            if comp.compname != 'h2o':
                self.molality_vec[ii] = self.molality(x[ii], xw)
                self.activity_vec[ii] = (
                    np.log(self.molality_vec[ii])
                    + 2.0*self.molality_vec[ii]*self.gamma_P1_vec[ii]
                )

        self.activity_vec[self.h2oind] = (
            np.sum(-0.018015*(self.molality_vec**2*(self.gamma_P1_vec)
                              + self.molality_vec))
        )

        mu_ik_RT = self.mu_ik_RT_cons + self.activity_vec
        fug = np.exp(mu_ik_RT - self.g_io_vec)

        return fug

    def molality(self, xc, xw):
        return xc/(xw*0.018015)

    # Main calculation that will call "fugacity". Option to specify phase.
    def calc(self, compobjs, T, P, x):
        if len(x) != len(compobjs):
            if len(x) > len(compobjs):
                raise RuntimeError('Length of mole fraction vector "x" '
                                   + 'exceeds number of components!')
            elif not x:
                raise RuntimeError('Mole fraction vector "x" is empty!')
            else:
                raise RuntimeError('Mole fraction vector "x" contains less '
                                   +'values than component length!')
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

            fug = self.fugacity(compobjs, x)
        return fug



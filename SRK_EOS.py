#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:40:41 2016

@author: kdarnell
"""
import numpy as np

"""
    This is an equation of state (EOS) for vapor and liquid hydrocarbon phases
    using the Soave-Redlich-Kwong (SRK) cubic equation.
    At present (30 Dec. 16), there is no correction for the liquid volume.
"""
R = 83.144621  # universal gas constant
# Possible aliases for describing the particular phase requested.
liquidalias = ('liquid', 'liq', 'l')
vaporalias = ('vapor', 'vap', 'v', 'gas', 'g')


class SrkFugs(object):
    def __init__(self, compobjs, T, P):
        self.description = (
            'Object for calculating fugacity of mixtures of gases' +
            'using the Soave-Relich-Kwong (SRK) cubic equation of state.'
        )
        # Set up arrays and matrices for future calculation
        self.Nc = len(compobjs)
        self.S1_vec = np.zeros(self.Nc)
        self.kij_mat = np.zeros([self.Nc, self.Nc])
        self.a_vec = np.zeros(self.Nc)
        self.b_vec = np.zeros(self.Nc)
        self.a_mat = np.zeros([self.Nc, self.Nc])
        self.alf_vec = np.zeros(self.Nc)
        self.Tr_vec = np.zeros(self.Nc)
        self.Pr_vec = np.zeros(self.Nc)
        # Store these, but continue feeding them into subsequent functions
        # to make sure they don't change!!
        self.compobjs = compobjs

        # Assuming pressure and temperature won't change, compute terms that
        # are not functions of compostion.
        self.make_constant_mats(compobjs, T, P)

    # Define all constants wrt to composition.
    def make_constant_mats(self, compobjs, T, P):
        for ii, comp in enumerate(compobjs):
            self.T = T
            self.P = P
            self.Tr_vec[ii] = T/comp.Tc
            self.Pr_vec[ii] = P/comp.Pc
            self.S1_vec[ii] = (0.48508 + 1.55171*comp.SRK['omega']
                               - 0.15613*comp.SRK['omega']**2)
            self.alf_vec[ii] = (
                (1.0 + self.S1_vec[ii]*(1.0 - np.sqrt(self.Tr_vec[ii]))
                 + comp.SRK['S2']*(1.0 - np.sqrt(self.Tr_vec[ii]))
                 / np.sqrt(self.Tr_vec[ii]))**2
            )
            self.a_vec[ii] = 0.42747*R**2*comp.Tc**2 / comp.Pc
            self.b_vec[ii] = 0.08664*R*comp.Tc / comp.Pc

        for ii, compouter in enumerate(compobjs):
            for jj, compinner in enumerate(compobjs):
                self.kij_mat[ii, jj] = compouter.SRK['kij'][compinner.compname]
                self.a_mat[ii, jj] = (
                    (1 - self.kij_mat[ii, jj])
                     * np.sqrt(self.alf_vec[ii]*self.a_vec[ii]
                     * self.alf_vec[jj]*self.a_vec[jj])
                )

    # Weighted-sum of b
    def b_tot(self, x):
        b = np.sum(self.b_vec*x)
        return b

    # Weighted-sum of a
    def a_tot(self, x):
        a = 0.0
        for ii in range(len(x)):
            for jj in range(len(x)):
                a += x[ii]*x[jj]*self.a_mat[ii, jj]
        return a

    # Function for fugacity calculation in terms of pre-computed values,
    # composition, and the Z-factor calculated in "calc".
    def fugacity(self, x, Z):
        fug = (x*self.P
               * np.exp((self.b_frac)*(Z - 1.0) - np.log(Z - self.B)
                        - self.A/self.B*(2.0*self.a_frac - self.b_frac)
                        * np.log(1.0 + self.B/Z))
               )
        return fug

    # Main calculation that will call "fugacity". Option to specify phase.
    def calc(self, compobjs, T, P, x, phase='general'):
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

            self.A = self.a_tot(x)*P / (R**2 * T**2)
            self.B = self.b_tot(x)*P / (R*T)
            coeffs = [1, -1, self.A - self.B - self.B**2, -(self.A*self.B)]
            Z = np.roots(coeffs)
            self.b_frac = self.b_vec/self.b_tot(x)
            self.a_x_sum = np.matmul(self.a_mat, x)
            self.a_frac = self.a_x_sum/self.a_tot(x)

            if np.isreal(Z).all():
                if phase.lower() in liquidalias:
                    fug = self.fugacity(x, Z.min())
                elif phase.lower() in vaporalias or phase.lower() == 'general':
                    fug = self.fugacity(x, Z.max())
            elif np.isreal(Z).any():
                # There should actually only be one real number if any
                # imaginary roots exists, so the np.max() is redundant.
                fug = self.fugacity(x, np.real(np.max(Z[np.isreal(Z)])))
            else:
                fug = np.nan
                print('Something is wrong.' +
                      '\nSolver returned imaginary numbers')
        return fug

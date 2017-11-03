#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vapor and liquid hydrocarbon  Soave-Redlich-Kwong Equation of State

This file implements a vapor/liquid hydrocarbon equation of state (EOS)
named Soave-Redlich-Kwong (SRK). The file consists of a single
class, 'SrkEos' that will take as arguments a list of components
and a pressure and a temperature. Pressure and temperature can be
modified after an instance of SrkEos is created; however, the number
of components and actual component list cannot. The method 'calc' is the
main calculation of the class, which uses other methods to determine
the partial fugacity of each component given mole fractions, pressure,
and temperature.
"""
import numpy as np


R = 83.144621  # universal gas constant (compatible with bar)
# Possible aliases for describing the particular phase requested.
liquid_alias = ('liquidhc', 'liqhc', 'lhc')
vapor_alias = ('vapor', 'vap', 'v', 'gas', 'g')


class SrkEos(object):
    """The main class for this EOS that perform various calculations.

    Methods
    ----------
    make_constant_mats :
        Performs calculations that only depend on pressure and temperature.
    fugacity :
        Calculates fugacity of each component in the aqueous phase.
    calc:
        Main calculation for aqueous phase EOS.
    """
    def __init__(self, comps, T, P):
        """Vapor and liquid hydrocarbon EOS object for fugacity calculations.

        Parameters
        ----------
        comps : list
            List of components as 'Component' objects created with
            'component_properties.py'.
        T : float
            Temperature at initialization in Kelvin.
        P : float
            Pressure at initialization in bar.

        Attributes
        ----------
        comps : list
            List of 'Component' classes passed into 'SrkEos'.
        num_comps : int
            Number of components.
        T : float
            Temperature at initialization in Kelvin.
        P : float
            Pressure at initialization in bar.
        s1_vec : numpy array
            Pre-allocated array for variable 's1'.
        kij_vec : numpy array
            Pre-allocated array for interaction parameter between
            component 'i' and component 'j'.
        a_vec : numpy array
            Pre-allocated array for variable 'a'.
        b_vec : numpy array
            Pre-allocated array for variable 'b'.
        a_mat : numpy array
            Pre-allocated array for variable 'a' derived from a_vec.
        alf_vec : numpy array
            Pre-allocated array for variable 'alpha'.
        Tr_vec : numpy array
            Pre-allocated array reduced temperature.
        Pr_vec : numpy array
            Pre-allocated array reduced pressure.
        A : float
            Constant used in the calculation of 'Z'.
        B : float
            Constant used in the calculation of 'Z'.
        Z : float
            Root of the cubic equation for molar volume.
        a_frac : numpy array
            Fraction of 'a' parameter for each component.
        b_frac : numpy array
            Fraction of 'b' parameter for each component.
        a_x_sum : numpy array
            Sum of the a_mat with the molar fraction vector.
        """
        self.comps = comps
        self.num_comps = len(comps)
        self.T = T
        self.P = P
        self.s1_vec = np.zeros(self.num_comps)
        self.kij_mat = np.zeros([self.num_comps, self.num_comps])
        self.a_vec = np.zeros(self.num_comps)
        self.b_vec = np.zeros(self.num_comps)
        self.a_mat = np.zeros([self.num_comps, self.num_comps])
        self.alf_vec = np.zeros(self.num_comps)
        self.Tr_vec = np.zeros(self.num_comps)
        self.Pr_vec = np.zeros(self.num_comps)
        self.A = None
        self.B = None
        self.Z = None
        self.a_x_sum = None
        self.a_frac = np.zeros(self.num_comps)
        self.b_frac = np.zeros(self.num_comps)
        self.make_constant_mats(comps, T, P)

    def make_constant_mats(self, comps, T, P):
        """Portion of calculation that only depends on P and T.

        Parameters
        ----------
        comps : list
            List of components as 'Component' objects created with
            'component_properties.py'.
        T : float
            Temperature in Kelvin.
        P : float
            Pressure in bar.

        Notes
        ----------
        Calculation assumes that pressure and temperature won't change
        upon successive iteration of EOS. Instead, the calculation will
        adjust molar fractions of each component at a fixed T and P.
        However, if T and P do change then, it will recalculate these
        constants.
        """
        for ii, comp in enumerate(comps):
            self.T = T
            self.P = P
            self.Tr_vec[ii] = T/comp.Tc
            self.Pr_vec[ii] = P/comp.Pc
            self.s1_vec[ii] = (0.48508 + 1.55171 * comp.SRK['omega']
                               - 0.15613 * comp.SRK['omega'] ** 2)
            if comp.compname == 'h2o':
                self.s1_vec[ii] = 1.2440
            self.alf_vec[ii] = (
                (1.0 + self.s1_vec[ii] * (1.0 - np.sqrt(self.Tr_vec[ii]))
                 + comp.SRK['S2'] * (1.0 - np.sqrt(self.Tr_vec[ii]))
                 / np.sqrt(self.Tr_vec[ii]))**2
            )

            # Potential change from not fitting lhc-hydrate
            # self.alf_vec[ii] = (
            #     (1.0 + self.s1_vec[ii] * (1.0 - np.sqrt(self.Tr_vec[ii])
            #      / np.sqrt(self.Tr_vec[ii]))) ** 2
            # )

            self.a_vec[ii] = 0.42747*R**2*comp.Tc**2 / comp.Pc
            self.b_vec[ii] = 0.08664*R*comp.Tc / comp.Pc

        for ii, comp_outer in enumerate(comps):
            for jj, comp_inner in enumerate(comps):
                self.kij_mat[ii, jj] = comp_outer.SRK['kij'][comp_inner.compname]
                self.a_mat[ii, jj] = (
                    (1 - self.kij_mat[ii, jj])
                    * np.sqrt(self.alf_vec[ii]*self.a_vec[ii]
                    * self.alf_vec[jj]*self.a_vec[jj])
                )

    def b_tot(self, x):
        """Molar fraction weighted sum of 'b' parameter.

        Parameters
        ----------
        x : numpy array
            Molar fraction of each component.

        Returns
        ----------
        float
            Molar fraction weighted sum of 'b' parameter.
        """
        return np.sum(self.b_vec*x)

    def a_tot(self, x):
        """Molar fraction weighted sum of 'a' parameter.

        Parameters
        ----------
        x : numpy array
            Molar fraction of each component.

        Returns
        ----------
        float
            Molar fraction weighted sum of 'a' parameter.
        """
        a = np.sum([x[ii]*x[jj]*self.a_mat[ii, jj]
                for ii in range(len(x)) for jj in range(len(x))])
        return a

    def fugacity(self, x):
        """Fugacity of each component in hydrocarbon phase for molar fractions 'x'.

        Parameters
        ----------
        x : list, numpy array
            Molar fractions of each components indexed in the same order
            as comps.

        Returns
        ----------
        fug : numpy array
            Fugacity of each component in aqueous phase.
        """
        fug = (x*self.P
               * np.exp(self.b_frac*(self.Z - 1.0) - np.log(self.Z - self.B)
                        - self.A/self.B*(2.0*self.a_frac - self.b_frac)
                        * np.log(1.0 + self.B/self.Z))
               )
        return fug
        
    @property
    def volume(self):
        """Volume of phase.

        Returns
        ----------
        v : float
            Volume of phase in cm^3.
        """
        v = self.Z*R*self.T/self.P
        return v
        
# TODO    Write this function!
# def enthalpy(self, Z):

    def calc(self, comps, T, P, x, phase='general'):
        """Main calculation for the EOS which returns array of fugacities

        Parameters
        ----------
        comps : list
            List of components as 'Component' classes.
        T : float
            Temperature in Kelvin.
        P : float
            Pressure in bar.
        x : list, numpy array
            Molar fractions of each components indexed in the same order
            as comps.
        phase : str, optional
            Specific phase for the calculation (liquid or vapor). This dictates
            which of the roots are return by the 'Z' calculation.

        Returns
        ----------
        fug : numpy array
            Fugacity of each component in aqueous phase.
        """
        if len(x) != len(comps):
            if len(x) > len(comps):
                raise RuntimeError("""Length of mole fraction vector 'x'
                                   exceeds number of components!""")
            elif not x:
                raise RuntimeError("Mole fraction vector 'x' is empty!")
            else:
                raise RuntimeError(""""Mole fraction vector "x" contains less
                                   values than component length!""")
        
        if comps != self.comps:
            print("""Warning: Action not supported.
                  \nComponents have changed.
                  \nPlease create a new fugacity object.""")
            return None
        else:
            if self.T != T or self.P != P:
                self.make_constant_mats(comps, T, P)

            self.b_frac = self.b_vec / self.b_tot(x)
            self.a_x_sum = np.matmul(self.a_mat, x)
            self.a_frac = self.a_x_sum / self.a_tot(x)
            self.A = self.a_tot(x)*P / (R**2 * T**2)
            self.B = self.b_tot(x)*P / (R*T)
            coefs = [1, -1, self.A - self.B - self.B**2, -(self.A*self.B)]
            Z = np.roots(coefs)

            if np.isreal(Z).all():
                if phase.lower() in liquid_alias:
                    self.Z = Z.min()
                elif phase.lower() in vapor_alias or phase.lower() == 'general':
                    self.Z = Z.max()
            elif np.isreal(Z).any():
                self.Z = np.real(np.max(Z[np.isreal(Z)]))
            else:
                print("""Something is wrong.
                      \nSolver returned imaginary numbers""")
                return None
                
            fug = self.fugacity(x)
        return fug

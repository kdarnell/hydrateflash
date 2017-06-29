#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aqueous Helgeson Equation of State with Bromley activity model

This file implements an aqueous equation of state (EOS) named the
Helgeson EOS with Bromley activity model. It is specifically
the version of the that EOS that is described in Jager et. al. (2003),
which is linked in the readme.md file. The file consists of a single
class, 'HegBromEos' that will take as arguments a list of components
and a pressure and a temperature. Pressure and temperature can be
modified after an instance of HegBromEos is created; however, the number
of components and actual component list cannot. The method 'calc' is the
main calculation of the class, which uses other methods to determine
the partial fugacity of each component given mole fractions, pressure,
and temperature.

    Functions
    ----------
    pure_water_vol_intgrt :
        Calculates integrated change in the volume of pure water from P_0 to P
        at fixed T.
    pure_water_vol :
        Calculates volume of pure water at P and T.
    dielectric_const :
        Calculates dielectric constant of pure water at P and T.
    molality :
        Calculates molality of each solute in the aqueous phase.
    solute_vol_integrated:
        Calculates volume of each component as a solute.
"""

import numpy as np

# Constants for EOS
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
# Constants produced by symbolic integration for h_ast funum_compstion
f1 = 73786976294838206464
f2 = 8151985141053725
f3 = 3249460376862603
f4 = 9444732965739290427392
f5 = 4722366482869645213696
f6 = 1043454098054876800
# Constants for pure water
gw_pure = -237129
hw_pure = -285830
cp_a0 = R * 8.712
cp_a1 = R * 1e-2 * 0.125
cp_a2 = R * 1e-5 * -0.018
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
"""The above global variables may be used throughout the calculation."""


def pure_water_vol_intgrt(T, P):
    """Volume of pure water integrate wrt pressure.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    P : float
        Pressure in bar.

    Returns
    ----------
    v_w : float
        Volume of water integrated in cm^3 - bar.
    """
    v_w = ((a10 * P + a20 * P ** 2 / 2 + a30 * P ** 3 / 3 + a40 * P ** 4 / 4)
           + (a11 * P + a21 * P ** 2 / 2 + a31 * P ** 3 / 3 + a41 * P ** 4 / 4) * T
           + (a12 * P + a22 * P ** 2 / 2 + a32 * P ** 3 / 3 + a42 * P ** 4 / 4) * T ** 2
           + (a13 * P + a23 * P ** 2 / 2 + a33 * P ** 3 / 3 + a43 * P ** 4 / 4) * T ** 3)
    return v_w

def pure_water_vol(T, P):
    """Volume of pure water.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    P : float
        Pressure in bar.

    Returns
    ----------
    v_w : float
        Volume of water in cm^3.
    """
    v_w = ((a10 + a20 * P + a30 * P ** 2 + a40 * P ** 3)
           + (a11 + a21 * P + a31 * P ** 2 + a41 * P ** 3) * T
           + (a12 + a22 * P + a32 * P ** 2 + a42 * P ** 3) * T ** 2
           + (a13 + a23 * P + a33 * P ** 2 + a43 * P ** 3) * T ** 3)
    return v_w


def dielectric_const(T, P):
    """Dielectric constant of pure water.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    P : float
        Pressure in bar.

    Returns
    ----------
    eps : float
        Dielectric constant of water (no units) .
    """
    eps = ((s10 + s20 * P + s30 * P ** 2)
           + (s11 + s21 * P + s31 * P ** 2) * T
           + (s12 + s22 * P + s32 * P ** 2) * T ** 2)
    return eps


def molality(xc, xw):
    """Calculates molality of each solute in the aqueous phase.

    Parameters
    ----------
    xc : float
        Mole fraction of component.
    xw : float
        Mole fraction water.

    Returns
    ----------
    float
        Molality of component in mol[component] / kg[water].

    """
    return xc / (xw * 0.018015)


def solute_vol_integrated(comp, T, P):
    """Volume of solute integrated wrt pressure.

    Parameters
    ----------
    comp : object
        Instance of Component class for each component
    T : float
        Temperature at initialization in Kelvin.
    P : float
        Pressure at initialization in bar.

    Returns
    ----------
    v_ast_P : float
        Volume of solute integrated wrt pressure in cm^3 - bar
    """

    omega = comp.AqHB['omega_born']
    v1 = comp.AqHB['v']['v1']
    v2 = comp.AqHB['v']['v2']
    v3 = comp.AqHB['v']['v3']
    v4 = comp.AqHB['v']['v4']
    tau = ((5.0 / 6.0) * T - theta) / (1.0 + np.exp((T - 273.15) / 5.0))

    v_ast_P = (
        (v1 * P + v2 * np.log(psi + P)
        + (v3 * P + v4 * np.log(psi + P)) * (1.0 / (T - theta - tau))
        + omega / dielectric_const(T, P)) / (R * T)
    )
    return v_ast_P


class HegBromEos(object):
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
        """Aqueous EOS object for fugacity calculations.

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
        water_ind : int
            Index of for water component in all lists.
        comps : list
            List of 'Component' classes passed into 'HegBromEos'.
        comp_names : list
            List of components names.
        num_comps : int
            Number of components.
        T : float
            Temperature at initialization in Kelvin.
        P : float
            Pressure at initialization in bar.
        g_io_vec : numpy array
            Pre-allocated array for gibbs energy of each component
            in ideal gas state.
        molality_vec : numpy array
            Pre-allocated array for molality of each component.
        activity_vec : numpy array
            Pre-allocated array for activity of each component
            in Bromley activity model.
        gamma_p1_vec : numpy array
            Pre-allocated array for gamma_{p1} variable of each
            component in Bromley activity model.
        mu_ik_RT_vec : numpy array
            Pre-allocated array chemical potential of each component.
        """
        try:
            self.water_ind = [ii for ii, x in enumerate(comps)
                              if x.compname == 'h2o'][0]
        except ValueError:
            raise RuntimeError(
                """Aqueous EOS requires water to be present!
                \nPlease provide water in your component list.""")
        self.comps = comps
        self.comp_names = [x.compname for x in comps]
        self.num_comps = len(comps)
        self.T = T
        self.P = P
        self.g_io_vec = np.zeros(self.num_comps)
        self.molality_vec = np.zeros(self.num_comps)
        self.activity_vec = np.zeros(self.num_comps)
        self.gamma_p1_vec = np.zeros(self.num_comps)
        self.mu_ik_rt_cons = np.zeros(self.num_comps)
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
        self.T = T
        self.P = P
        for ii, comp in enumerate(comps):
            self.g_io_vec[ii] = comp.gibbs_ideal(T, P)

            if comp.compname != 'h2o':
                c1 = comp.AqHB['cp']['c1']
                c2 = comp.AqHB['cp']['c2']
                omega = comp.AqHB['omega_born']
                h_io_ast = comp.h_io_ast

                # Output of symbolic integration.
                h_ast = (np.log(T) * (f1 * c1 + f2 * omega) / (f1 * R)
                         - (np.log(T_0) * (f1 * c1 + f2 * omega)) / (f1 * R)
                         - (f3 * T * omega) / (f4 * R) + (f3 * T_0 * omega) / (f4 * R)
                         - (f5 * T_0 * c2
                            - T_0 * (f4 * c2 + f4 * T_0 * h_io_ast - f4 * T_0 ** 2 * c1
                                     - f6 * T_0 ** 2 * omega + f3 * T_0 ** 3 * omega)
                            ) / (f4 * R * T_0 ** 3)
                         + (f5 * T_0 * c2
                            - T * (f4 * c2 + f4 * T_0 * h_io_ast - f4 * T_0 ** 2 * c1
                                   - f6 * T_0 ** 2 * omega + f3 * T_0 ** 3 * omega)
                            ) / (f4 * R * T ** 2 * T_0))

                self.mu_ik_rt_cons[ii] = (
                    comp.g_io_ast / (R * T_0) - h_ast
                    + solute_vol_integrated(comp, T, P)
                    - solute_vol_integrated(comp, T, P_0)
                )

                if comp.compname == 'co2':
                    self.gamma_p1_vec[ii] = (0.107 - 4.5e-4 * T)
            else:
                self.mu_ik_rt_cons[ii] = (
                    gw_pure / (R * T_0)
                    - (12 * T * hw_pure - 12 * T_0 * hw_pure + 12 * T_0 ** 2 * cp_a0
                       + 6 * T_0 ** 3 * cp_a1 + 4 * T_0 ** 4 * cp_a2 + 3 * T_0 ** 5 * cp_a3
                       - 12 * T * T_0 * cp_a0 - 12 * T * T_0 ** 2 * cp_a1
                       + 6 * T ** 2 * T_0 * cp_a1 - 6 * T * T_0 ** 3 * cp_a2
                       + 2 * T ** 3 * T_0 * cp_a2 - 4 * T * T_0 ** 4 * cp_a3
                       + T ** 4 * T_0 * cp_a3 + 12 * T * T_0 * cp_a0 * np.log(T)
                       - 12 * T * T_0 * cp_a0 * np.log(T_0)) / (12 * R * T * T_0)
                    + (pure_water_vol_intgrt(T, P)
                       - pure_water_vol_intgrt(T, P_0)) * 1e-1 / (R * T)
                )

    def fugacity(self, comps, x):
        """Fugacity of each component in aqueous phase for molar fractions 'x'.

        Parameters
        ----------
        comps : list
            List of components as 'Component' classes.
        x : list, numpy array
            Molar fractions of each components indexed in the same order
            as comps.

        Returns
        ----------
        fug : numpy array
            Fugacity of each component in aqueous phase.
        """

        xw = x[self.water_ind]
        for ii, comp in enumerate(comps):
            if comp.compname != 'h2o':
                self.molality_vec[ii] = molality(x[ii], xw)
                self.activity_vec[ii] = (
                    np.log(self.molality_vec[ii])
                    + 2.0 * self.molality_vec[ii] * self.gamma_p1_vec[ii]
                )

        self.activity_vec[self.water_ind] = (
            np.sum(-0.018015 * (self.molality_vec ** 2 * self.gamma_p1_vec
                                + self.molality_vec))
        )

        mu_ik_RT = self.mu_ik_rt_cons + self.activity_vec
        fug = np.exp(mu_ik_RT - self.g_io_vec)
        return fug

    def calc(self, comps, T, P, x):
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
                raise RuntimeError("""Mole fraction vector 'x' contains less
                                   values than component length!""")
        if comps != self.comps:
            print("""Warning: Action not supported.
                  \n Number of_components have changed.
                  \n Please create a new fugacity object.""")
            return None
        else:
            # Re-calculate constants if pressure or temperature changes.
            if self.T != T or self.P != P:
                self.make_constant_mats(comps, T, P)

            fug = self.fugacity(comps, x)
        return fug

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hydrate Modified-van der Waals Platteeuw Equation of State

This file implements a hydrate equation of state (EOS)named
Modified- van der Waals Platteeuw after Ballard and Sloan (2002).
The file consists of a generic hydrate class 'Hydrate EOS' and the
class, 'HvdwpmEos', which incorporates all information for the
Ballard and Sloan EOS. Other hydrate EOS's will share similar features,
since they are all derived from the original van der Waals Platteeuw EOS,
so the generic 'HydrateEos' could be ported to another EOS. The classes will
take in as arguments a list of components, pressure, temperature, and
hydrate structure. Pressure and temperature can be modified after an
instance of SrkEos is created; however, the number of components, the
actual component list, and the hydrate structure cannot. The method 'calc'
is the main calculation of the class, which uses other methods to determine
the partial fugacity of each component given mole fractions, pressure,
temperature, and structure.

    Functions
    ----------
    delta_func :
        Special calculation used in Kihara potential
    w_func :
        Spherical kihara potential
"""
import numpy as np
from scipy.integrate import quad
import pdb

# Constants
R = 8.3144621  # universal gas constant in J/mol-K
T_0 = 298.15  # reference temperature in K
P_0 = 1.0  # reference pressure in bar
k = 1.3806488e-23  # Boltzmann's constant in J/K


def delta_func(N, Rn, aj, r):
    """Function specifically used in langmuir constant calculation

    Parameters
    ----------
    N : int
        Exponent used in calculation
    Rn : float
        Radius of cage
    aj : float
        Radius of guest molecule
    r : float
        General radius

    Returns
    ----------
    delta : float
        Distance measure
    """
    delta = ((1.0 - r/Rn - aj/Rn)**(-N) - (1.0 + r/Rn - aj/Rn)**(-N))/N
    return delta

def w_func(zn, eps_k, r, Rn, sigma, aj):
    """Kihara spherical potential function

    Parameters
    ----------
    zn : int
        Number of water molecules
    eps_k : float
        Kihara potential parameter, \epsilon (normalized by
        Boltzmann's constant), for guest
    r : float
        General radius
    Rn : float
        Radius of cage
    sigma : float
        Kihara potential parameter, sigma, for guest
    aj : float
        Radius of guest molecule


    Returns
    ----------
    w : float
        Kihara potential of guest at a specific radius
    """
    w = (2 * zn * eps_k * (sigma ** 12 / (Rn ** 11 * r)
                           * (delta_func(10, Rn, aj, r)
                              + (aj / Rn) * delta_func(11, Rn, aj, r))
                           - sigma ** 6 / (Rn ** 5 * r)
                           * (delta_func(4, Rn, aj, r)
                              + (aj / Rn) * delta_func(5, Rn, aj, r))))
    return w

class HydrateEos(object):
    """The parent class for this EOS that perform various calculations.

    Methods
    ----------
    make_constant_mats :
        Performs calculations that only depend on pressure and temperature.
    langmuir_consts :
        Performs the langmuir constant calculation.
    calc_langmuir :
        Performs that calculates cage occupancies from langmuir constants.
    delta_mu_func :
        Calculates chemical potential difference according to van der Waals.
    fugacity :
        Calculates fugacity of only water in the hydrate phase.
    calc:
        Main calculation for hydrate phase EOS.
    """
    def __init__(self, comps, T, P, structure='s1'):
        """Hydrate EOS object for fugacity calculations.

        Parameters
        ----------
        comps : list
            List of components as 'Component' objects created with
            'component_properties.py'.
        T : float
            Temperature at initialization in Kelvin.
        P : float
            Pressure at initialization in bar.
        structure : str
            Structure of hydrate ('s1' or 's2').

        Attributes
        ----------
        water_ind : int
            Index of for water component in all lists.
        comps : list
            List of 'Component' classes passed into 'HydrateEos'.
        comp_names : list
            List of components names.
        num_comps : int
            Number of components.
        T : float
            Temperature at initialization in Kelvin.
        P : float
            Pressure at initialization in bar.
        gwbeta_RT : numpy array
            Pre-allocated array for gibbs energy of water
            in empty standard state.
        volume_int : numpy array
            Pre-allocated array for pressure-integrated volume
            of hydrate.
        volume : numpy array
            Pre-allocated array for volume of hydrate.
        activity : numpy array
            Pre-allocated array for activity of water in hydrate.
        delta_mu_RT : numpy array
            Pre-allocated array for chemical potential difference
            of water between aqueous and empty hydrate phases.
        gw0_RT : numpy array
            Pre-allocated array for gibbs energy of water in ideal
            gas state.
        Hs : dict
            Dictionary of structure-specific properties of hydrate.

            Keys and values
            ----------
            R_sm : numpy array
                Pre-allocated array for radii of each shell
                in the small cages.
            z_sm : numpy array
                Pre-allocated array for number of water molecules
                in each shell in the large cages.
            R_lg : numpy array
                Pre-allocated array for radii of each shell
                in the small cages.
            z_lg : numpy array
                Pre-allocated array for number of water molecules
                in each shell in the large cages.
        alt_fug_vec : numpy array
            Pre-allocated array for fugacity of non-water components
            in phase at equilibrium with hydrate.
        Y_small : numpy array
            Pre-allocated array for cage occupancy of small cages for
            each non-water component.
        Y_large : numpy array
            Pre-allocated array for cage occupancy of large cages for
            each non-water component.
        C_small : numpy array
            Pre-allocated array for langmuir constants of small cages for
            each non-water component.
        C_large : numpy array
            Pre-allocated array for langmuir constants of large cages for
            each non-water component.
        fug : float
            Fugacity of water in hydrate.

        Notes
        ----------
        'Hs' will populate based on the 'eos_key' defined in the child class.
        This attribute will store the structure specific information that is
        stored in a class called 'HydrateStructure'.
        """
        try:
            self.water_ind = [ii for ii, x in enumerate(comps)
                              if x.compname == 'h2o'][0]
        except ValueError:
            raise RuntimeError(
                """Hydrate EOS requires water to be present!
                \nPlease provide water in your component list.""")
        self.comps = comps
        self.comp_names = [x.compname for x in comps]
        self.num_comps = len(comps)
        self.T = T
        self.P = P
        self.gwbeta_RT = np.zeros(1)
        self.volume_int = np.zeros(1)
        self.volume = np.zeros(1)
        self.activity = np.zeros(1)
        self.delta_mu_RT = np.zeros(1)
        self.gw0_RT = np.zeros(1)
        self.Hs = HydrateStructure(structure)
        self.alt_fug_vec = np.zeros(self.num_comps)
        self.Y_small = np.zeros(self.num_comps)
        self.Y_large = np.zeros(self.num_comps)
        self.C_small = np.zeros(self.num_comps)
        self.C_large = np.zeros(self.num_comps)
        self.fug = 0
        for key, value in dict.items(self.Hs.eos[self.eos_key]):
            setattr(self.Hs, key, value)
        self.R_sm = np.zeros(len(self.Hs.R['sm']))
        self.z_sm = np.zeros_like(self.R_sm)
        for ii in range(len(self.z_sm)):
            self.z_sm[ii] = self.Hs.z['sm'][ii + 1]
        self.R_lg = np.zeros(len(self.Hs.R['lg']))
        self.z_lg = np.zeros_like(self.R_lg)
        for ii in range(len(self.z_lg)):
            self.z_lg[ii] = self.Hs.z['lg'][ii + 1]


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
        self.gw0_RT = comps[self.water_ind].gibbs_ideal(T, P)

    def langmuir_consts(self, comps, T, P):
        """Calculation of langmuir constants.

        Parameters
        ----------
        comps : list
            List of components as 'Component' objects created with
            'component_properties.py'.
        T : float
            Temperature in Kelvin.
        P : float
            Pressure in bar.

        Returns
        ----------
        C_small : numpy array
            Langmuir constants for component in small cages.
        C_large : numpy array
            Langmuir constants for component in large cages.
        """
        C_small = np.zeros(self.num_comps)
        C_large = np.zeros(self.num_comps)
        return C_small, C_large

    def calc_langmuir(self, C, eq_fug):
        """Calculation of langmuir constants.

        Parameters
        ----------
        C : numpy array
            Array of langmuir constants.
        eq_fug : numpy array
            Fugacity of each component for the phase that is in
            equilibrium with hydrate.

        Returns
        ----------
        Y : numpy array
            Fractional occupancy of each component in unspecified
            cage type.
        """
        Y = np.zeros(self.num_comps)
        denominator = (1.0 + np.sum(C * eq_fug))
        for ii, comp in enumerate(self.comps):
            if comp.compname != 'h2o':
                Y[ii] = C[ii] * eq_fug[ii] / denominator
            else:
                Y[ii] = 0
        return Y

    def delta_mu_func(self, comps, T, P):
        """Calculation of chemical potential difference of water in hydrate.

        Parameters
        ----------
        comps : list
            List of components as 'Component' objects created with
            'component_properties.py'.
        T : float
            Temperature at initialization in Kelvin.
        P : float
            Pressure at initialization in bar.

        Returns
        ----------
        delta_mu : float
            Chemical potential difference of water between aqueous phases
            and empty standard hydrate.
        """
        delta_mu = (
            self.Hs.Nm['small']*np.log(1-np.sum(self.Y_small))
            + self.Hs.Nm['large']*np.log(1-np.sum(self.Y_large))
        )/self.Hs.Num_h2o
        return delta_mu

    def fugacity(self, comps, T, P, x, eq_fug):
        """Calculation of fugacity of water in hydrate.

        Parameters
        ----------
        comps : list
            List of components as 'Component' objects created with
            'component_properties.py'.
        T : float
            Temperature in Kelvin.
        P : float
            Pressure in bar.
        x : list, numpy array
            Dummy list of molar fraction in hydrate phase to maintain
            argument parallelism with other EOS's.
        eq_fug : numpy array
            Equilibrium fugacity of each non-water component that will
            be in equilibrium within some other phase.
        """
        pass

    def calc(self, comps, T, P, x, eq_fug):
        """Main calculation for the EOS which returns array of fugacities

        Parameters
        ----------
        comps : list
            List of components as 'Component' objects created with
            'component_properties.py'.
        T : float
            Temperature in Kelvin.
        P : float
            Pressure in bar.
        x : list, numpy array
            Dummy list of molar fraction in hydrate phase to maintain
            argument parallelism with other EOS's.
        eq_fug : numpy array
            Equilibrium fugacity of each non-water component that will
            be in equilibrium within some other phase.

        Returns
        ----------
        fug : numpy array
            Fugacity of water in aqueous phase.
        """
        if comps != self.comps:
            print("""Warning: Action not supported.
                  \n Number of_components have changed.
                  \n Please create a new fugacity object.""")
            return None
        else:
            # Re-calculate constants if pressure or temperature changes.
            if self.T != T or self.P != P:
                self.make_constant_mats(comps, T, P)

            fug = self.fugacity(comps, T, P, x, eq_fug)
        return fug

class HvdwpmEos(HydrateEos):
    """The child class for this EOS that perform various calculations.

    Methods
    ----------
    make_constant_mats :
        Performs calculations that only depend on pressure and temperature.
    find_stdstate_volume :
        Finds the standard state volume of hydrate.
    find_hydrate_properties :
        Performs calculations necessary to determine hydrate properties.
    kappa_func :
        Determines compressibility of hydrate with specific cage occupancies.
    hydrate_size :
        Calculates size of hydrate.
    h_vol_Pint :
        Volume of hydrate integrated with respect to pressure.
    lattice_to_volume :
        Convert the lattice size of hydrate in angstrom to a volume in
        cm^3/mol.
    cage_distortion :
        Calculate distortion of cage size due to filling by guests.
    filled_lattice_size :
        Calculate size of hydrate filled with guests.
    iterate_function :
        Special function that must be iterated due to nonlinear dependencies.
    integrand :
        Setup integrand for calculation of langmuir constant.
    compute_integral_constants :
        Calculate portions of integral that do not depend on composition.
    langmuir_consts :
        Calculate langmuir constants.
    activity_func :
        Calculate activity of water in hydrate due to filling of cages.
    fugacity :
        Calculate fugacity of wate rin hydrate.
    hyd_comp :
        Conversion of cage occupancies to a hydrate composition.

    Constants
    ----------
    cp : dict
        Dictionary of constants for heat capacity
    eos_key : str
        Key for use in HydrateStructure class information retrieval.s
    """
    cp = {'a0': 0.735409713*R,
          'a1': 1.4180551e-2*R,
          'a2': -1.72746e-5*R,
          'a3': 63.5104e-9*R}
    eos_key = 'hvdwpm'

    def __init__(self, comps, T, P, structure='s1'):
        """Hydrate EOS object for fugacity calculations.

        Parameters
        ----------
        comps : list
            List of components as 'Component' objects created with
            'component_properties.py'
        T : float
            Temperature at initialization in Kelvin
        P : float
            Pressure at initialization in bar
        structure : str
            Structure of hydrate ('s1' or 's2')

        Attributes
        ----------
        kappa : numpy array
            Compressibility of filled hydrate
        kappa0 : float
            Compressibility of empty hydrate of specific structure
        a0_cubed : float
            Size of standard empty hydrate
        kappa_vec : numpy array
            Compressibility of hydrate filled by a single guest
        rep_sm_vec : numpy array
            Repulsive constant for each guest in small cage
        rep_lg_vec : numpy array
            Repulsive constant for each guest in large cage
        D_vec : numpy array
            Diameter of each guest
        a_new : float
            Size of hydrate with specific composition
        Y_small_0 : numpy array
            Cage occupancy of small cage at standard state
        Y_large_0 : numpy array
            Cage occupancy of large cage at standard state
        eq_fug : numpy array
            Equilibrium fugacity of each component in another phase at
            equilibrium with hydrate phase
        stdstate_fug : numpy array
            Fugacity of each component at standard state
        vol_Tfactor : float
            Change in hydrate volume from temperature dependence
        lattice_Tfactor : float
            Change in hydrate lattice size from temperature dependence
        kappa_tmp : float
            Temporary storage of kappa during convergence
        a_0 : float
            Lattice size at standard state
        v_H_0 : float
            Hydrate volume at standard state
        v_H : float
            Hydrate volume
        repulsive_small : float
            Repulsive constant in small cages
        epulsive_small : float
            Repulsive constant in large cages
        """

        # Inherit all properties from HydrateEos
        try:
            super().__init__(comps, T, P, structure)
        except:
            super(HvdwpmEos, self).__init__(comps, T, P, structure)

        self.kappa = np.zeros(1)
        self.kappa0 = self.Hs.kappa
        self.a0_cubed = self.lattice_to_volume(self.Hs.a0_ast)
        self.kappa_vec = np.zeros(self.num_comps)
        self.rep_sm_vec = np.zeros(self.num_comps)
        self.rep_lg_vec = np.zeros(self.num_comps)
        self.D_vec = np.zeros(self.num_comps)
        self.a_new = self.Hs.a_norm
        self.Y_small_0 = np.zeros(self.num_comps)
        self.Y_large_0 = np.zeros(self.num_comps)
        self.eq_fug = np.zeros(self.num_comps)
        self.stdstate_fug = np.zeros(self.num_comps)
        self.vol_Tfactor = None
        self.lattice_Tfactor = None
        self.kappa_tmp = None
        self.a_0 = None
        self.v_H_0 = None
        self.v_H = None
        self.repulsive_small = None
        self.repulsive_large = None

        # Retrieve information for components and populate within vectors
        for ii, comp in enumerate(comps):
            self.kappa_vec[ii] = comp.HvdWPM[self.Hs.hydstruc]['kappa']
            self.rep_sm_vec[ii] = comp.HvdWPM[self.Hs.hydstruc]['rep']['small']
            self.rep_lg_vec[ii] = comp.HvdWPM[self.Hs.hydstruc]['rep']['large']
            self.D_vec[ii] = comp.diam
            self.stdstate_fug[ii] = comp.stdst_fug

        # Set up parameters that do not change with the system, which
        # in the case means the fugacity of other phases.
        self.make_constant_mats(comps, T, P)

        # Set up parameters that depend on standard state (T_0, P_0)
        # Now, the lattice size won't change. However, the volume will
        # depend on P, T and kappa. And kappa depends on cage occupancy (Y)
        # and cage occupancy depends on langmuir constants, which in turn
        # depend on the volume. Thus, the langmuir constants will have to be
        # determined at each call to 'calc'.
        self.find_stdstate_volume(comps, T, P)


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
        try:
            super().make_constant_mats(comps, T, P)
        except:
            super(HvdwpmEos, self).make_constant_mats(comps, T, P)

        self.gwbeta_RT = (
            self.Hs.gw_0beta/(R*T_0)
            - (12*T*self.Hs.hw_0beta - 12*T_0*self.Hs.hw_0beta
               + 12*T_0**2*self.cp['a0'] + 6*T_0**3*self.cp['a1']
               + 4*T_0**4*self.cp['a2'] + 3*T_0**5*self.cp['a3']
               - 12*T*T_0*self.cp['a0'] - 12*T*T_0**2*self.cp['a1']
               + 6*T**2*T_0*self.cp['a1'] - 6*T*T_0**3*self.cp['a2']
               + 2*T**3*T_0*self.cp['a2'] - 4*T*T_0**4*self.cp['a3']
               + T**4*T_0*self.cp['a3'] + 12*T*T_0*self.cp['a0']*np.log(T)
               - 12*T*T_0*self.cp['a0']*np.log(T_0)) / (12*R*T*T_0)
            + (self.h_vol_Pint(T, P, self.a0_cubed, self.kappa0)
               - self.h_vol_Pint(T, P_0, self.a0_cubed, self.kappa0)) * 1e-1 / (R * T)
        )

        self.vol_Tfactor = self.hydrate_size(T, P_0, 1.0, self.kappa0)
        self.lattice_Tfactor = self.hydrate_size(T, P_0, 1.0,
                                                 self.kappa0, dim='linear')

    def find_stdstate_volume(self, comps, T, P):
        """Calculation of standard state volume and other properties.

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
        Standard state is an empty hydrate at reference pressure and temperature.
        This still depends on composition.
        """
        error = 1e6
        TOL = 1e-8
        C_small = np.zeros(self.num_comps)
        C_large = np.zeros(self.num_comps)
        # 'Lattice sz' will originally be equal to self.Hs.a0_ast because we
        # set self.Y_*_0 equal to zero.
        # That will produce one set of C's and new Y's. We will then iterate
        # until convergence on 'C'.
        kappa = self.kappa0
        lattice_sz = self.Hs.a0_ast
        while error > TOL:
            out = self.iterate_function(comps, T_0, P_0,
                                        self.stdstate_fug,
                                        lattice_sz, kappa)
            # Update these on every iteration
            C_small_new = out[0]
            C_large_new = out[1]
            self.Y_small_0 = out[2]
            self.Y_large_0 = out[3]
            lattice_sz = self.filled_lattice_size()
            kappa = self.kappa_func(self.Y_large_0)

            error = 0.0
            for ii, comp in enumerate(comps):
                if comp.compname != 'h2o':
                    error += (abs(C_small_new[ii]
                                  - C_small[ii])/C_small_new[ii]
                              + abs(C_large_new[ii]
                                    - C_large[ii])/C_large_new[ii])
            C_small = C_small_new
            C_large = C_large_new
        self.kappa_tmp = kappa
        self.a_0 = lattice_sz
        self.v_H_0 = self.lattice_to_volume(lattice_sz)

    def find_hydrate_properties(self, comps, T, P, eq_fug):
        """Hydrate properties at T, P, and with specific composition

        Parameters
        ----------
        comps : list
            List of components as 'Component' objects created with
            'component_properties.py'
        T : float
            Temperature in Kelvin
        P : float
            Pressure in bar
        eq_fug : numpy array
            Equilibrium fugacity of each non-water component that will
            be in equilibrium within some other phase
        """
        error = 1e6
        TOL = 1e-8

        if hasattr(self, 'C_small'):
            C_small = self.C_small
            C_large = self.C_large
        else:
            C_small = np.zeros(self.num_comps)
            C_large = np.zeros(self.num_comps)

        if hasattr(self.kappa_tmp, 'copy'):
            kappa = self.kappa_tmp.copy()
        else:
            kappa = self.kappa_tmp

        lattice_sz = self.a_0
        while error > TOL:
            out = self.iterate_function(comps, T, P, eq_fug,
                                        lattice_sz, kappa)

            # Update these on every iteration
            C_small_new = out[0]
            C_large_new = out[1]
            self.Y_small = out[2]
            self.Y_large = out[3]
            kappa = self.kappa_func(self.Y_large)

            error = 0.0
            for ii, comp in enumerate(comps):
                if comp.compname != 'h2o':
                    error += (abs(C_small_new[ii]
                                  - C_small[ii])/C_small_new[ii]
                              + abs(C_large_new[ii]
                                    - C_large[ii])/C_large_new[ii])
            C_small = C_small_new
            C_large = C_large_new

        self.C_small = C_small
        self.C_large = C_large
        if hasattr(self.kappa_tmp, 'copy'):
            kappa = self.kappa_tmp.copy()
        else:
            kappa = self.kappa_tmp
        self.v_H = self.hydrate_size(T, P, self.v_H_0, kappa)

    def kappa_func(self, Y_large):
        """Compressibility of hydrate function

        Parameters
        ----------
        Y_large : numpy array
            Fractional occupancy of large hydrate cages
        """
        if self.num_comps > 2:
            kappa = 3.0*np.sum(self.kappa_vec*Y_large)
        else:
            kappa = self.kappa0

        return kappa

    def hydrate_size(self, T, P, v, kappa, dim='volumetric'):
        """Hydrate properties at T, P, and with specific composition

        Parameters
        ----------
        T : float
            Temperature in Kelvin
        P : float
            Pressure in bar
        v : float
            Size of hydrate (either volumetric or linear)
        kappa : float
            Compressibility of hydrate
        dim : str
            Dimension of interest ('volumetric' or 'linear'

        Returns
        ----------
        H_size : float
            Size of hydrate in dimension specified in argument list

        Notes
        ----------
        In various places, either the linear or volumetric sizes are necessary
        components in a particular calculation. A small, constant difference is
        required to switch between the two.
        """
        if dim == 'volumetric':
            factor = 1.0
        elif dim == 'linear':
            factor = 1.0/3.0
        else:
            raise ValueError("Invalid option for optional argument 'dim'!")

        H_size = (v*np.exp(factor*(self.Hs.alf[1]*(T - T_0)
                                   + self.Hs.alf[2]*(T - T_0)**2
                                   + self.Hs.alf[3]*(T - T_0)**3
                                   - kappa*(P - P_0))))
        return H_size

    def h_vol_Pint(self, T, P, v, kappa):
        """Hydrate volume integrated with respect to pressure

        Parameters
        ----------
        T : float
            Temperature in Kelvin
        P : float
            Pressure in bar
        v : float
            Size of hydrate (either volumetric or linear)
        kappa : float
            Compressibility of hydrate

        Returns
        ----------
        v : float
            Hydrate volume integrated wrt pressure in cm^3 - bar /mol
        """
        v = self.hydrate_size(T, P, v, kappa) / (-kappa)
        return v

    def lattice_to_volume(self, lattice_sz):
        """Conversion of linear hydrate size to volumetric size

        Parameters
        ----------
        lattice_sz : float
            Size of hydrate as a radius in angstrom

        Returns
        ----------
        v : float
            Volume of hydrate in cm^3/mol
        """
        if self.Hs.hydstruc != 'sH':
            v = 6.0221413e23/self.Hs.Num_h2o/1e24*lattice_sz**3
        else:
            v = self.Hs.v0
        return v

    def cage_distortion(self):
        """Distortion of cages when filled with guests

        Notes
        ----------
        This depends on composition, but not on temperature or pressure.
        """
        small_const = (
            (1 + self.Hs.etam['small']/self.Hs.Num_h2o)*self.Y_small_0
            / (1 + (self.Hs.etam['small']/self.Hs.Num_h2o)*self.Y_small_0)
        )

        # Ballard did not differentiate between multiple or single guests
        # as we do here. In his version, the weighted exponential is always
        # used. However, we saw better agreement by separating single and
        # multiple guests.
        if self.num_comps > 1:
            # Results of flash calculations for hydrocarbon systems do not produce results that are consistent
            # with CSMGem. On August 31, 2017. I commented out the following lines to see if they are consisent.
            # if self.Hs.hydstruc == 's1':
            #     self.repulsive_small = (small_const * np.exp(
            #         self.D_vec - np.sum(self.Y_small_0  * self.D_vec) / np.sum(self.Y_small_0)
            #     ))
            # else:
            # self.repulsive_small = small_const
            # self.repulsive_small = (small_const * np.exp(
            #         self.D_vec - np.sum(self.Y_small_0  * self.D_vec)
            #     ))
            self.repulsive_small = (small_const * np.exp(
                -(self.D_vec - np.sum(self.Y_small_0  * self.D_vec))
                ))
            self.repulsive_small[self.water_ind] = 0.0
        else:
            self.repulsive_small = small_const

        self.repulsive_large = (
            (1 + self.Hs.etam['large']/self.Hs.Num_h2o) * self.Y_large_0
            / (1 + (self.Hs.etam['large']/self.Hs.Num_h2o) * self.Y_large_0)
        )
        self.repulsive_large[self.water_ind] = 0.0

    # Determine size of lattice due to the filling of cages at T_0, P_0
    def filled_lattice_size(self):
        """Size of hydrate lattice when filled

        Returns
        ----------
        lattice sz : float
            Size of lattice when filled by guests
        """
        self.cage_distortion()
        lattice_sz = (self.Hs.a0_ast
                      + (self.Hs.Nm['small']
                         * np.sum(self.repulsive_small * self.rep_sm_vec))
                      + (self.Hs.Nm['large']
                         * np.sum(self.repulsive_large * self.rep_lg_vec)))
        return lattice_sz

    def iterate_function(self, comps, T, P, fug, lattice_sz, kappa):
        """Function that must be iterated until convergence due to nonlinearity

        Parameters
        ----------
        comps : list
            List of components as 'Component' objects created with
            'component_properties.py'
        T : float
            Temperature in Kelvin
        P : float
            Pressure in bar
        fug : numpy array
            Fugacity of each non-water component that will
            be in equilibrium within some other phase
        lattice_sz : float
            Size of filled hydrate lattice
        kappa : float
            Compressibility of filled hydrate

        Returns
        ----------
        calc_list : list of numpy arrays
            List of output variables such that each element is a
            numpy array. First element is a numpy array of small cage
            langmuir constants, second element is a numpy array of large
            cage langmuir constants, third elements is a numpy array
            of small cage fractional occupancies, and fourth element is
            a numpy array of large cage fractional occupancies.
        """
        C_small, C_large = self.langmuir_consts(comps, T, P,
                                                lattice_sz, kappa)
        Y_small = self.calc_langmuir(C_small, fug)
        Y_large = self.calc_langmuir(C_large, fug)
        calc_list = [C_small, C_large, Y_small, Y_large]
        return  calc_list
    
    def integrand(self, r, Rn, z, eps_k, sigma, aj, T):
        """Kihara spherical potential function

        Parameters
        ----------
        r : float
            General radius
        Rn : list, numpy array
            Radius of each cage in hydrate structure
        z : int
            Number of water molecules
        eps_k : float
            Kihara potential parameter, \epsilon (normalized by
            Boltzmann's constant), of guest
        sigma : float
            Kihara potential parameter, sigma, og guest
        aj : float
            Radius of guest molecule
        T : float
            Temperature in Kelvin


        Returns
        ----------
        integrand_w : numpy array
            Integrand as a function of radius
        """
        integrand_sum = 0
        for ii in range(len(Rn)):
            integrand_sum += w_func(z[ii], eps_k, r, Rn[ii], sigma, aj)
            
        integrand_w = r**2*np.exp((-1.0/T)*integrand_sum)
        return integrand_w

    def compute_integral_constants(self, T, P, lattice_sz, kappa):
        """Function to compute integral for langmuir constant calculation

        Parameters
        ----------
        T : float
            Temperature in Kelvin
        P : float
            Pressure in bar
        lattice_sz : float
            Size of filled hydrate lattice
        kappa : float
            Compressibility of filled hydrate
        """
        Pfactor = self.hydrate_size(T_0, P, 1.0, kappa, dim='linear')
        a_factor = (lattice_sz/self.Hs.a_norm)*self.lattice_Tfactor*Pfactor
        for ii in range(len(self.Hs.R['sm'])):
            self.R_sm[ii] = self.Hs.R['sm'][ii + 1]*a_factor
        for ii in range(len(self.Hs.R['lg'])):
            self.R_lg[ii] = self.Hs.R['lg'][ii + 1]*a_factor

    def langmuir_consts(self, comps, T, P, lattice_sz, kappa):
        """Calculates langmuir constant through many interior calculations

        Parameters
        ----------
        comps : list
            List of components as 'Component' objects created with
            'component_properties.py'
        T : float
            Temperature in Kelvin
        P : float
            Pressure in bar
        lattice_sz : float
            Size of filled hydrate lattice
        kappa : float
            Compressibility of filled hydrate

        Returns
        ----------
        C_small : numpy array
            Langmuir constants for each guest in small cage
        C_large : numpy array
            Langmuir constants for each guest in large cage

        Notes
        ----------
        Calculation will perform numerical integration and is numerically
        expensive. Other methods are possible, but not as accurate given
        the accompanying empirically fit parameter set.
        """
        self.compute_integral_constants(T, P, lattice_sz, kappa)
        C_small = np.zeros(self.num_comps)
        C_large = np.zeros(self.num_comps)
        C_const = 1e-10**3*4*np.pi/(k*T)*1e5
        for ii, comp in enumerate(comps):
            if ii != self.water_ind:
                small_int = quad(self.integrand,
                                 0,
                                 min(self.R_sm) - comp.HvdWPM['kih']['a'],
                                 args=(self.R_sm,
                                       self.z_sm,
                                       comp.HvdWPM['kih']['epsk'],
                                       comp.HvdWPM['kih']['sig'],
                                       comp.HvdWPM['kih']['a'],
                                       T,))
                large_int = quad(self.integrand,
                                 0,
                                 min(self.R_lg) - comp.HvdWPM['kih']['a'],
                                 args=(self.R_lg,
                                       self.z_lg,
                                       comp.HvdWPM['kih']['epsk'],
                                       comp.HvdWPM['kih']['sig'],
                                       comp.HvdWPM['kih']['a'],
                                       T,))

                # Quad returns a tuple of the integral and the error.
                # We want to retrieve the integrated value.
                C_small[ii] = C_const*small_int[0]
                C_large[ii] = C_const*large_int[0]
            else:
                C_small[ii] = 0
                C_large[ii] = 0

        return C_small, C_large


    def activity_func(self, T, P, v_H_0):
        """Calculates activity of water between aqueous phase and filled hydrate

        Parameters
        ----------
        T : float
            Temperature in Kelvin
        P : float
            Pressure in bar
        v_H_0 : float
            Volume of hydrate at standard state

        Returns
        ----------
        activity : float
            Activity of water
        """
        kappa_wtavg = self.kappa_func(self.Y_large)
        activity = (
            (v_H_0 - self.a0_cubed) / R * (self.Hs.a_fit/T_0
                                       + self.Hs.b_fit*(1/T - 1/T_0))
            + ((self.h_vol_Pint(T, P, v_H_0, kappa_wtavg)
                - self.h_vol_Pint(T, P, self.a0_cubed, self.kappa0))
               - (self.h_vol_Pint(T, P_0, v_H_0, kappa_wtavg)
                  - self.h_vol_Pint(T, P_0, self.a0_cubed, self.kappa0))) / (R * T)
        ) * 1e-1
        return activity


    def fugacity(self, comps, T, P, x, eq_fug):
        """Main calculation for the EOS which returns array of fugacities

        Parameters
        ----------
        comps : list
            List of components as 'Component' objects created with
            'component_properties.py'.
        T : float
            Temperature in Kelvin.
        P : float
            Pressure in bar.
        x : list, numpy array
            Dummy list of molar fraction in hydrate phase to maintain
            argument parallelism with other EOS's.
        eq_fug : numpy array
            Equilibrium fugacity of each non-water component that will
            be in equilibrium within some other phase.

        Returns
        ----------
        fug : float
            Fugacity of water in aqueous phase.
        """
        if (eq_fug == self.eq_fug).all():
            fug = self.fug.copy()
        else:
            fug = np.zeros(self.num_comps)
            fug[1:] = eq_fug[1:]

        self.find_hydrate_properties(comps, T, P, eq_fug)
        delta_mu_RT = self.delta_mu_func(comps, T, P)
        activity = self.activity_func(T, P, self.v_H_0)
        mu_H_RT = self.gwbeta_RT + activity + delta_mu_RT
        fug[0] = np.exp(mu_H_RT - self.gw0_RT)
        self.fug = fug.copy()
        return fug

    # Calcualte hydrate composition
    def hyd_comp(self):
        """Hydrate composition as a molar fraction

        Returns
        ----------
        x : numpy
            Molar fraction of each component in hydrate phase
        """
        x_tmp = (self.Hs.Nm['small']*self.Y_small
                 + self.Hs.Nm['large']*self.Y_large)/self.Hs.Num_h2o
        x = x_tmp/(1.0 + np.sum(x_tmp))
        x[self.water_ind] = 1.0 - np.sum(x)
        return x


# Properties of each hydrate structure necessary for further calculation.
class HydrateStructure(object):
    # Possible aliases of the hydrate structures
    menu = {'s1': ('s1', '1', 1, 'si', 'i', 'one'),
            's2': ('s2', '2', 2, 'sii', 'ii', 'two'),
            'sh': ('sh', 'h')}

    def __init__(self, hyd_type):
        self.description = ("""Object for holding properties for each type of
                               hydrate structure (1,2,H)""")

        if str(hyd_type).lower() in self.menu['s1']:
            self.hydstruc = 's1'
        elif str(hyd_type).lower() in self.menu['s2']:
            self.hydstruc = 's2'
        elif str(hyd_type).lower() in self.menu['sh']:
            self.hydstruc = 'sH'
        else:
            raise RuntimeError(hyd_type + """ 
                               is not a supported hydrate structure!!\n
                               Consult "HydrateStructure.menu" attribute 
                               for valid structures.""")

        # 'alf' is presented as a volumetric parameter.
        # Divide by 3 when used for lattice parameter.
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
                                   'a_fit': 260.0,
                                   'b_fit': -68.64,
                                   'alf': {1: 2.029776e-4,
                                           2: 1.851168e-7,
                                           3: -1.879455e-10},
                                   'a_norm': 17.1,
                                   'R': {'sm': {1: 3.748,
                                                2: 3.845,
                                                3: 3.956},
                                         'lg': {1: 4.729,
                                                2: 4.715,
                                                3: 4.635}},
                                   'z': {'sm': {1: 2,
                                                2: 6,
                                                3: 12},
                                         'lg': {1: 4,
                                                2: 12,
                                                3: 12}}}}

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Algorithm for determining phase stability given P, T, and composition

The algorithmic framework presented here constitutes a hydrate
flash algorithm, such that the amount and presence of hydrate can
be predicted given simple input. It follows the general procedure
of CSMGem.

    Functions
    ----------
    stability func :
        Calculation of phase stability compatibility
    objective :
        Calculation to determine objective function that must be minimized
    jacobian :
        Calculation to determine jacobian of objective function that must be minimized
    ideal_LV :
        Calculation of ideal partition coefficients between liquid and vapor phases
    ideal_VAq :
        Calculation of ideal partition coefficients between vapor and aqueous phases
    ideal_Ice Aq :
        Calculation of ideal partition coefficients between ice and aqueous phases
    ideal_VHs1 :
        Calculation of ideal partition coefficients between vapor and structure 1 hydrate phases
    ideal_VHs2 :
        Calculation of ideal partition coefficients between vapor and structure 2 hydrate phases
    make_ideal_K_allmat :
        Use ideal partition coefficient functions to construct a matrix of coefficients

"""
import numpy as np
import time

import component_properties as cp
import aq_hb_eos as aq
import h_vdwpm_eos as h
import vlhc_srk_eos as hc
import pdb

"""Mapping from columns of K_all_mat to corresponding partition coefficient
First phase is numerator, second phase is denominator"""
K_dict = {0: ('lhc', 'vapor'),
          1: ('vapor', 'aqueous'),
          2: ('vapor', 's1'),
          3: ('vapor', 's2'),
          4: ('ice', 'aqueous')}

"""Transformation from K_all_mat to partition coefficients of the form,
K_{j, ref_phase} where keys in K_transform refer to reference phase
and subsequent keys refer to phase j, and tuple describes algebraic
manipulation of K_mat_all. 9 refers to the value 1 0-4 refers to a
column"""
K_transform = {'aqueous': {'vapor': (1),
                           'lhc': (1, 0),
                           'aqueous': (9),
                           'ice': (4),
                           's1': (1, 2),
                           's2': (1, 3)},
               'vapor': {'vapor': (9),
                         'lhc': (9, 0),
                         'aqueous': (9, 1),
                         'ice': (4, 1),
                         's1': (9, 2),
                         's2': (9, 3)},
               'lhc': {'vapor': (0),
                       'lhc': (9),
                       'aqueous': (0, 1),
                       'ice': ((0, 4), 1),
                       's1': (0, 2),
                       's2': (0, 3)},
               'ice': {'vapor': (1, 4),
                       'lhc': (1, (0, 4)),
                       'aqueous': (9, 4),
                       'ice': (9),
                       's1': (1, (2, 4)),
                       's2': (1, (3, 4))}}

def stability_func(alpha, theta):
    """Simple function that should always equal zero

    Parameters
    ----------
    alpha : float, numpy array
        Molar of phase fraction
    theta : float, numpy array
        Stability of phase

    Returns
    ----------
    Calculation output of size identical to alpha, theta
    """
    return alpha * theta / (alpha + theta)

def objective(z, alpha, theta, K):
    """Objective function to be minimized

    Parameters
    ----------
    z : list, numpy array
        Total composition of each component with size Nc
    alpha : list, numpy array
        Molar phase fractions with size Np
    theta : list, numpy array
        Stability of phases with size Np
    K : list, numpy array
        Partition coefficients for each component
        in each phase with size Nc x Np

    Returns
    ----------
    cost : numpy array
        Numerical "cost" or residual of size 2*Np
    """
    if type(z) != np.ndarray:
        z = np.asarray(z)
    if type(alpha) != np.ndarray:
        alpha = np.asarray(alpha)
    if type(theta) != np.ndarray:
        theta = np.asarray(theta)
    if type(K) != np.ndarray:
        K = np.asarray(K)

    numerator = z[:, np.newaxis] * (K * np.exp(theta[np.newaxis, :]) - 1)
    denominator = 1 + np.sum(
        alpha[np.newaxis, :]
        * (K * np.exp(theta[np.newaxis, :]) - 1),
        axis=1)
    e_cost = np.sum(numerator / denominator[:, np.newaxis], axis=0)
    y_cost = stability_func(alpha, theta)
    cost = np.concatenate((e_cost, y_cost))
    return cost

def jacobian(z, alpha, theta, K):
    """Jacobian of objective function to be minimized

    Parameters
    ----------
    z : list, numpy array
        Total composition of each component with size Nc
    alpha : list, numpy array
        Molar phase fractions with size Np
    theta : list, numpy array
        Stability of phases with size Np
    K : list, numpy array
        Partition coefficients for each component
        in each phase with size Nc x Np

    Returns
    ----------
    jacobian : numpy array
        Jacobian matrix of objective function of size 2*Np x 2 *Np
    """
    if type(z) != np.ndarray:
        z = np.asarray(z)
    if type(alpha) != np.ndarray:
        alpha = np.asarray(alpha)
    if type(theta) != np.ndarray:
        theta = np.asarray(theta)
    if type(K) != np.ndarray:
        K = np.asarray(K)


    stability_mat = (K*np.exp(theta[np.newaxis, :]) - 1.0)
    alpha_numerator = (
        z[:, np.newaxis, np.newaxis]
        * stability_mat[:, :, np.newaxis]
        * stability_mat[:, np.newaxis, :])
    theta_numerator = (
        z[:, np.newaxis, np.newaxis]
        * stability_mat[:, :, np.newaxis]
        * K[:, np.newaxis,:]
        * alpha[np.newaxis, np.newaxis, :]
        * np.exp(theta[np.newaxis, np.newaxis, :]))
    denominator = (
        1.0 + (np.sum(alpha[np.newaxis, :]
                      * stability_mat,
                      axis=1))
        )**2
    jac_alpha = -np.sum(
        alpha_numerator / denominator[:, np.newaxis, np.newaxis],
        axis = 0)
    jac_theta = -np.sum(
        theta_numerator / denominator[:, np.newaxis, np.newaxis],
        axis = 0)
    diag_denom = 1.0 + np.sum((K * np.exp(theta[np.newaxis, :]) - 1.0)
                               * alpha[np.newaxis, :],
                              axis=1)
    diag = np.sum(z[:, np.newaxis] * K * np.exp(theta[np.newaxis, :])
                  / diag_denom[:, np.newaxis],
                  axis=0)
    jac_theta += np.diag(diag)
    jac_cost = np.concatenate((jac_alpha, jac_theta), axis=1)
    jac_alpha_y = (theta/(alpha + theta)
                          - alpha*theta/(alpha + theta)**2)
    jac_theta_y = (alpha/(alpha + theta)
                          - alpha*theta/(alpha + theta)**2)
    jac_stability = np.concatenate((np.diag(jac_alpha_y),
                                    np.diag(jac_theta_y)),
                                    axis=1)
    jacobian = np.concatenate((jac_cost, jac_stability), axis=0)
    return jacobian


#TODO: Convert all the ideal stuff into a separate class.
def ideal_LV(compobjs, T, P):
    """Ideal partition coefficients for liquid and vapor phases

    Parameters
    ----------
    compobjs : list, tuple
        List of components
    T : float
        Temperature in Kelvin
    P : float
        Pressure in bar

    Returns
    ----------
    K : numpy array
        Array of partition coefficients for each component
    """

    if not hasattr(compobjs, '__iter__'):
        compobjs = [compobjs]

    K = np.ones([len(compobjs)])
    for ii, comp in enumerate(compobjs):
        if comp.compname != 'h2o':
            K[ii] = ((comp.Pc/P)
                     * np.exp(5.373*(1.0 + comp.SRK['omega'])
                              *(1 - comp.Tc/T)))
        else:
            K[ii] = (-133.67 + 0.63288*T)/P + 3.19211e-3*P
    return K


def ideal_VAq(compobjs, T, P):
    """Ideal partition coefficients for vapor and aqueous phases

    Parameters
    ----------
    compobjs : list, tuple
        List of components
    T : float
        Temperature in Kelvin
    P : float
        Pressure in bar

    Returns
    ----------
    K : numpy array
        Array of partition coefficients for each component
    """
    if not hasattr(compobjs, '__iter__'):
        compobjs = [compobjs]

    K = np.ones([len(compobjs)])

    for ii, comp in enumerate(compobjs):
        if comp.compname != 'h2o':
            gamma_inf = np.exp(0.688 - 0.642*comp.N_carb)
            a1 = (5.927140 - 6.096480*(comp.Tc/T)
                  - 1.288620*np.log(T/comp.Tc) + 0.169347*T**6/comp.Tc**6)

            a2 = (15.25180 - 15.68750*(comp.Tc/T)
                  - 13.47210*np.log(T/comp.Tc) + 0.43577*T**6/comp.Tc**6)
            P_sat = comp.Pc*np.exp(a1 + comp.SRK['omega']*a2)
        else:
            gamma_inf = 1.0
            P_sat = np.exp(12.048399 - 4030.18425/(T + -38.15))
        K[ii] = (P_sat/P)*gamma_inf
    return K


def ideal_IceAq(compobjs, T, P):
    """Ideal partition coefficients for ice and aqueous phases

    Parameters
    ----------
    compobjs : list, tuple
        List of components
    T : float
        Temperature in Kelvin
    P : float
        Pressure in bar

    Returns
    ----------
    K : numpy array
        Array of partition coefficients for each component
    """
    if not hasattr(compobjs, '__iter__'):
        compobjs = [compobjs]

    K = np.ones([len(compobjs)])
    for ii, comp in enumerate(compobjs):
        if comp.compname != 'h2o':
            K[ii] = 0
        else:
            T_0 = 273.1576
            P_0 = 6.11457e-3
            T_ice = T_0 - 7.404e-3*(P - P_0) - 1.461e-6*(P - P_0)**2
            xw_aq = 1 + 8.33076e-3*(T - T_ice) + 3.91416e-5*(T - T_ice)**2
            K[ii] = 1.0/xw_aq
    return K


def ideal_VHs1(compobjs, T, P):
    """Ideal partition coefficients for vapor s1 hydrate phases

    Parameters
    ----------
    compobjs : list, tuple
        List of components
    T : float
        Temperature in Kelvin
    P : float
        Pressure in bar

    Returns
    ----------
    K : numpy array
        Array of partition coefficients for each component
    """
    if not hasattr(compobjs, '__iter__'):
        compobjs = [compobjs]

    K = np.ones([len(compobjs)])
    for ii, comp in enumerate(compobjs):
        if comp.compname != 'h2o':
            s = comp.ideal['Hs1']
            K_wf = np.exp(
                    s['a1'] + s['a2']*np.log(P) + s['a3']*np.log(P)**2
                    - (s['a4'] + s['a5']*np.log(P) + s['a6']*np.log(P)**2
                       + s['a7']*np.log(P)**3)/T
                    + s['a8']/P + s['a9']/P**2 + s['a10']*T + s['a11']*P
                    + s['a12']*np.log(P/T**2) + s['a13']/T**2)
            K[ii] = K_wf/(1 - 0.88)
        else:
            K[ii] = (ideal_VAq(comp, T, P)
                     / (0.88 * ideal_IceAq(comp, T, P)))
    return np.abs(K)


def ideal_VHs2(compobjs, T, P):
    """Ideal partition coefficients for vapor s1 hydrate phases

    Parameters
    ----------
    compobjs : list, tuple
        List of components
    T : float
        Temperature in Kelvin
    P : float
        Pressure in bar

    Returns
    ----------
    K : numpy array
        Array of partition coefficients for each component
    """
    if not hasattr(compobjs, '__iter__'):
        compobjs = [compobjs]

    K = np.ones([len(compobjs)])
    T_Kelvin = T
    T = T*9.0/5.0 - 459.67
    for ii, comp in enumerate(compobjs):
        if comp.compname != 'h2o':
            s = comp.ideal['Hs2']
            K_wf = np.exp(
                    s['a1'] + s['a2']*T + s['a3']*P + s['a4']/T
                    + s['a5']/P + s['a6']*T*P + s['a6']*T**2
                    + s['a8']*P**2 + s['a9']*P/T + s['a10']*np.log(P/T)
                    + s['a11']/P**2 + s['a12']*T/P + s['a13']*T**2/P
                    + s['a14']*P/T**2 + s['a15']*T/P**3 + s['a16']*T**3
                    + s['a17']*P**3/T**2 + s['a18']*T**4
                    + s['a19']*np.log(P))
            K[ii] = K_wf/(1 - 0.90)
        else:
            K[ii] = (ideal_VAq(comp, T_Kelvin, P)
                     / (0.90 * ideal_IceAq(comp, T_Kelvin, P)))
    return K


def make_ideal_K_allmat(compobjs, T, P):
    """Ideal partition coefficients for vapor s1 hydrate phases

    Parameters
    ----------
    compobjs : list, tuple
        List of components
    T : float
        Temperature in Kelvin
    P : float
        Pressure in bar

    Returns
    ----------
    K_all_mat : numpy array
        Matrix of all possible partition coefficients for each component
    """
    if not hasattr(compobjs, '__iter__'):
        compobjs = [compobjs]
    K_all_mat = np.zeros([len(compobjs), 5])
    K_all_mat[:, 0] = ideal_LV(compobjs, T, P)
    K_all_mat[:, 1] = ideal_VAq(compobjs, T, P)
    K_all_mat[:, 2] = ideal_VHs1(compobjs, T, P)
    K_all_mat[:, 3] = ideal_VHs2(compobjs, T, P)
    K_all_mat[:, 4] = ideal_IceAq(compobjs, T, P)
    return K_all_mat


class FlashController(object):
    """Flash calculation and auxiliary components

    Attributes
    ----------
    phase_menu : dict
        Dictionary relating possible phases (keys) and aliases
        for those phases (values).
    eos_menu : dict
        Dictionary relating possible phases (keys) and different
        eos's (values) for each phase that might be used.
    eos_default : dict
        Dictionary setting default eos (values) for each possible
        phase (key).
    """
    phase_menu = {'aqueous': ('aqueous', 'aq', 'water', 'liquid'),
                  'vapor': ('vapor', 'v', 'gas', 'vaporhc', 'hc'),
                  'lhc': ('lhc', 'liquidco2', 'liquid_co2', 'l',
                          'liquidhc'),
                  's1': ('s1', 's1hydrate', 's1h', 'hs1', 'h1',
                         'hydrate', 'hydrate s1', 'structure 1',
                         'structure 1 hydrate', 'str 1'),
                  's2': ('s2', 's2hydrate', 's2h', 'hs2', 'h2',
                         'hydrate s2', 'structure 2',
                         'structure 2 hydrate', 'str 2'),
                  'ice': ('ice')}
    eos_menu = {'aqueous': ('aqhb', 'henryslaw'),
                'vapor': ('srk', 'pr'),
                'lhc': ('srk', 'pr'),
                's1': ('hvdwpm', 'hvdwpped', 'hvdwpvel'),
                's2': ('hvdwpm', 'hvdwpped', 'hvdwpvel'),
                'ice': ('ice')}

    eos_default={'aqueous':'aqhb',
                 'vapor': 'srk',
                 'lhc': 'srk',
                 's1': 'hvdwpm',
                 's2': 'hvdwpm',
                 'ice': 'ice'}


    def __init__(self,
                 components,
                 phases=['aqueous', 'vapor', 'lhc', 's1'],
                 eos=eos_default,
                 T=298.15,
                 P=1.0):
        """Flash controller for a fixed set of components with allowable modification
        to pressure, temperature, or composition


        Parameters
        ----------
        components : list, tuple
            Set of components to be used for the flash controller. This is not
            allowed to change
        phases : list, tuple
            Set of phases to consider during flash
        eos : dict
            Dictionary for relating phases to a specific type of eos
        T : float
            Temperature in Kelvin
        P : float
            Pressure in bar

        Attributes
        ----------
        T : float
            Temperature in Kelvin
        P : float
            Pressure in bar
        ref_phase : str
            Name of phase that is used for partition coefficient denominator
        compobjs : list
            List of components as 'Component' objects created with
            'component_properties.py'
        compname : list
            List of proper component names
        h2oexists : bool
            Boolean variable that describes whether water is one of the components
        h2oind : int
            Index to water in all component-indexed lists
        phases : list
            List of phases to consider in calculation
        fug_list : list
            List of phase-based eos objects for fugacity calculations
        hyd_phases : dictionary
            Dictionary hydrate phases and corresponding indices in 'self.phases'
        nonhyd_phases : list
            List of non-hydrates phases where each element is the corresponding index in 'self.phases'
        feed : list, array
            Total composition (z) of calculation
        ref_comp : numpy array
            Composition of reference phase
        ref_comp : numpy array
            Fugacity of reference phase
        ref_phases_tried : list
            List of reference phases tried during calculation starting empty
        K_calc : numpy array
            Array of reference phase relative partition coefficients wiht size Nc x Np
        x_calc : numpy array
            Array of compositions in each phase with size Nc x Np
        alpha_calc : numpy array
            Array of molar phase fraction of each phase with size Np
        theta_calc : numpy array
            Array of phase stability of each phase with size Np

        Methods
        ----------
        set_feed :
            Take a list or array of size Nc and set the
            total composition such sum(z) == 1
        set_ref_index :
            Determine the index within self.phases of the new reference phases
        set_phases :
            Re-assign phases from a list based argument of phase types
        change_ref_phase :
            Cycles through possible unused reference phases
        main_handler :
            Primary method for calculation logic
        calc_x :
            Calculation of composition for auxiliary variables
        calc_K :
            Calculation of partition coefficients using output of fugacity calculations
        calc_fugacity :
            Calculation of fugacity of each component in each phase
        find_alphatheta_min :
            Calculation that performs minimization of objective function at fixed x and K
        make_ideal_K_mat :
            Determine initial partition coefficients independent of composition
        """

        self.T = T
        self.P = P
        self.ref_phase = None
        self.completed = False
        # Check that components exceed 1.
        if type(components) is str or len(components) == 1:
            raise ValueError("""More than one component is necessary 
                                to run flash algorithm.""")
        elif type(components[0]) is str:
            # Use list or array of component names to populate a new list of
            # component objects
            self.compobjs = []
            for compname in components:
                self.compobjs.append(cp.Component(compname))
        elif isinstance(components[0], cp.Component):
            self.compobjs = components
        else:
            raise ValueError("""Component are not properly defined.
                                Pass a list of strings
                                or a list of component objects""")

        self.compname = []
        self.h2oexists = False
        self.h2oind = None
        for ii, comp in enumerate(self.compobjs):
            self.compname.append(comp.compname)
            if comp.compname == 'h2o':
                self.h2oexists = True
                self.h2oind = ii
        self.Nc = len(self.compobjs)

        # Check that phases exceed 1
        if type(phases) is str or len(phases) == 1:
            raise ValueError(""""More than one component is necessary 
                                to run flash algorithm.""")
        else:

            # Populate phase list and make sure it's a valid phase.
            self.phases = []
            for phase in phases:
                if phase.lower() not in self.phases:
                    phase_found = False
                    for real_phase, alias in self.phase_menu.items():
                        if phase.lower() in alias:
                            self.phases.append(real_phase)
                            phase_found = True

                    if not phase_found:
                        raise ValueError(phase + """
                                is not a supported phase!! \nConsult
                                "FlashController.phase_menu" attribute for
                                \nvalid phases and ssociated call strings.""")
                else:
                    # Perhaps, I should just print warning that eliminates
                    # duplicate phases.
                    raise ValueError("""One or more phases are repeated. 
                                        Distinct phases are necessary to 
                                        run flash algorithm.""")

        # Allow option for changing the default eos for any given phase.
        # Check to make sure the eos is supported and that the phase being
        # modified is a valid phase.
        if eos != self.eos_default:
            if type(eos) is not dict:
                raise TypeError("""
                        "eos" specifies equation of state to be used with
                        a specific phase. Pass a dictionary as eos
                        (e.g., eos={"vapor": "pr"}).""")
            else:
                self.eos = self.eos_default.copy()
                for phase_tmp, eos_tmp in eos.items():
                    if phase_tmp in self.eos_default.keys():
                        if eos_tmp in self.eos_menu[phase_tmp]:
                            self.eos[phase_tmp] = eos_tmp
                        else:
                            raise ValueError(
                                    eos_tmp + ' is not a valid eos for '
                                    + phase_tmp + ' phase.')
                    else:
                        for valid_phase, alias in self.phase_menu.items():
                            phase_found = False
                            if phase_tmp.lower() in alias:
                                if eos_tmp in self.eos_menu[valid_phase]:
                                    self.eos[valid_phase] = eos_tmp
                                    phase_found = True
                                else:
                                    raise ValueError(
                                            eos_tmp + ' is not a valid eos for '
                                            + valid_phase + ' phase.')
                        if not phase_found:
                            raise ValueError(
                                    phase_tmp + ' is not a valid phase.')
        else:
            self.eos = eos

        # Build list for fugacity calculations
        fug_list = list()
        hyd_phases = dict()
        phase_check = list(self.phases)
        for ii, phase in enumerate(phase_check):
            if phase in ('aqueous', 's1', 's2'):
                if self.h2oexists:
                    if phase == 'aqueous':
                        aq_obj = aq.HegBromEos(self.compobjs, self.T, self.P)
                        fug_list.append(aq_obj)
                        self.ref_phase = phase
                    else:
                        hyd_phases[phase] = ii
                        h_obj = h.HvdwpmEos(self.compobjs, self.T, self.P,
                                            structure=phase)
                        fug_list.append(h_obj)
                else:
                    self.phases.remove(phase)

            elif phase == 'ice':
                print('Not currently supported for an ice phase.')
                self.phases.remove(phase)
                if not hyd_phases:
                    for hyd_phase, hyd_ind in hyd_phases.items():
                        hyd_ind += -1
            else:
                hc_obj = hc.SrkEos(self.compobjs, self.T, self.P)
                fug_list.append(hc_obj)
                if (self.ref_phase is None) and ('vapor' in self.phases):
                    self.ref_phase = 'vapor'
                elif (self.ref_phase is None) and ('lhc' in self.phases):
                    self.ref_phase = 'lhc'

        self.fug_list = fug_list
        self.hyd_phases = hyd_phases
        self.nonhyd_phases = [ii for ii in range(len(self.phases))
                              if ii not in self.hyd_phases.values()]
        self.Np = len(self.phases)
        self.feed = None
        self.ref_phases_tried = []
        self.ref_comp = np.zeros([len(self.compobjs)])
        self.ref_fug = np.zeros([len(self.compobjs)])
        self.set_ref_index()
        self.K_calc = np.zeros([len(self.compobjs), len(self.phases)])
        self.x_calc = np.zeros([len(self.compobjs), len(self.phases)])
        self.alpha_calc = np.zeros([len(self.phases)])
        self.theta_calc = np.zeros([len(self.phases)])

    def set_feed(self, z, setref=True):
        """Utility for setting the feed and reference phase based on feed

        Parameters
        ----------
        z : list, tuple, numpy array
            Mole fraction of each components
        setref : bool
            Flag for setting reference phase
        """
        self.feed = np.asarray(z) / sum(z)
        if setref:
            if len(z) != len(self.compobjs):
                raise ValueError("""Feed fraction has different dimension than
                                    initial component list!""")
            elif self.h2oexists:
                if ((z[self.h2oind] > 0.8)
                    or (('vapor' not in self.phases)
                        and ('lhc' not in self.phases))):
                    self.ref_phase = 'aqueous'
                else:
                    if 'vapor' in self.phases:
                        self.ref_phase = 'vapor'
                    elif 'lhc' in self.phases:
                        self.ref_phase = 'lhc'
            elif 'vapor' in self.phases:
                # This is true for now. In practice, this sometimes needs to be
                # lhc, but we will handle that within a different method.
                self.ref_phase = 'vapor'
            elif 'lhc' in self.phases:
                self.ref_phase = 'lhc'

            self.ref_phases_tried = []

    # TODO: Refactor this to be called initialization and make all the same checks.
    def set_phases(self, phases):
        """Utility to reset phases

        Parameters
        ----------
        phases : list, tuple
            List of new phases
        """
        self.phases = phases

    def set_ref_index(self):
        """Utility to set index of reference phases"""
        if self.ref_phase not in self.phases:
            self.phases.append(self.ref_phase)
        self.ref_ind = [ii for ii, phase in enumerate(self.phases)
                        if phase == self.ref_phase].pop()

    # TODO: Make this more robust, it is possibility of breaking!
    def change_ref_phase(self):
        """Utility to change reference phase on calculation stall"""
        self.ref_phases_tried.append(self.ref_phase)
        self.ref_phase = [phase for phase in self.phases
                          if phase not in self.ref_phases_tried 
                          and phase not in ['s1', 's2']].pop(0)
        self.ref_phase_iter = 0
        if not self.ref_phase:
            self.ref_phase = self.ref_phases_tried[
                np.mod(self.ref_phase_iter,
                       len(self.ref_phases_tried))
            ]
            self.ref_phase_iter += 1
        self.set_ref_index()
        
    def main_handler(self, compobjs, z, T, P,
                     K_init=None, verbose=False,
                     initialize=True, run_diagnostics=False,
                     incipient_calc=False,
                     **kwargs):
        """Primary logical utility for performing flash calculation

        Parameters
        ----------
        compobjs : list, tuple
            List of components
        z : list, tuple, numpy array
            Molar composition of each component
        T : float
            Temperature in Kelvin
        P : float
            Pressure in bar
        K_init : numpy array
            Partition coefficient matrix to use at start of calculation
        verbose : bool
            Flag for printing to screen
        initialize : bool
            Flag for initializing the calculation using ideal partition coefficients
        run_diagnostics : bool
            Flag for doing debugging

        Returns
        ----------
        values : list
            List of calculation output of gibbs energy minimum
            values[0] : numpy array
                Composition (x) with size Nc x Np
            values[1] : numpy array
                Molar phase fraction (\alpha) with size Np
            values[2] : numpy array
                Partition coefficient matrix of each component
                in each phase (K) with size Nc x Np
            values[3] : int
                Number of iterations required for convergence
            values[4] : float
                Maximum error on any variable from minimization calculation
        """
        # z = np.asarray(z)
        self.set_feed(z)
        self.set_ref_index()
        if type(z) != np.ndarray:
            z = np.asarray(z)

        if verbose:
            tstart = time.time()
            
        if initialize or not self.completed:
            alpha_0 = np.ones([self.Np]) / self.Np
            theta_0 = np.zeros([self.Np])
            # TODO: Rewrite so that ideal K doesn't have to be re-calculated!
            if not incipient_calc:
                if not K_init:
                    K_0 = self.make_ideal_K_mat(compobjs, T, P)
                else:
                    # Add more code to allow the specification of a partition coefficient
                    print('K is not the default')
                    K_0 = np.asarray(K_init)


                if type(K_init) != np.ndarray:
                    K_init = np.asarray(K_init)
            else:
                K_0 = self.incipient_calc(T, P)

            alpha_new, theta_new = self.find_alphatheta_min(z, alpha_0,
                                                            theta_0, K_0)
            x_new = self.calc_x(z, alpha_new, theta_new, K_0, T, P)
            fug_new = self.calc_fugacity(T, P, x_new)
            x_new = self.calc_x(z, alpha_new, theta_new, K_0, T, P)
            K_new = self.calc_K(T, P, x_new)

            if run_diagnostics:
                print('Initial K:\n', K_0)
                print('First iter K:\n', K_new)
                print('First x:\n', x_new)
                print('First fugacity :\n', fug_new)
                print('First alpha:\n', alpha_new)
                print('First theta:\n', theta_new)

        else:
            alpha_new = self.alpha_calc.copy()
            theta_new = self.theta_calc.copy()
            K_new = self.K_calc.copy()
            x_new = self.x_calc.copy()

        error = 1e6
        TOL = 1e-6
        itercount = 0
        refphase_itercount = 0
        iterlim = 500
        
        alpha_old = alpha_new.copy()
        theta_old = theta_new.copy()
        x_old = x_new.copy()
        K_old = K_new.copy()
        
        while error > TOL and itercount < iterlim:
            # Perform newton iteration to update alpha and theta at
            # a fixed x and K
            alpha_new, theta_new = self.find_alphatheta_min(z, alpha_old, 
                                                             theta_old, K_new)

            # Perform one iteration of successive substitution to update
            # x and K at the new alpha and theta.
            x_error = 1e6
            x_counter = 0
            while x_error > 1e2*TOL and x_counter < 1:
                x_new = self.calc_x(z, alpha_new, theta_new, K_new, T, P)
                K_new = self.calc_K(T, P, x_new)
                x_error = np.sum(abs(x_new - x_old))
                x_counter += 1
            
            if run_diagnostics:
                print('Iter K:\n', K_new)
                print('Iter x:\n', x_new)
                print('Iter alpha:\n', alpha_new)
                print('Iter theta:\n', theta_new)
                print('Iter fug:\n:', self.calc_fugacity(T, P, x_new))
                print('Iter z:\n:', z)

            # Determine error associated new x and K and change in x
            # Set iteration error to the maximum of the two.
            Obj_error = np.sum(objective(z, alpha_new,
                                         theta_new, K_new))
            error = max(Obj_error, x_error)
            
            itercount += 1
            refphase_itercount += 1
            nan_occur = (np.isnan(x_new).any() or np.isnan(K_new).any() 
                         or np.isnan(alpha_new).any() or np.isnan(theta_new).any())
            
            if ((refphase_itercount > 20
                and alpha_new[self.ref_ind] < 0.001)
                or nan_occur) :
                self.change_ref_phase() 
                refphase_itercount = 0
                # TODO change these 3 lines to investigate the effect of changing the reference phase
                # K_new = self.make_ideal_K_mat(compobjs, T, P)
                # alpha_new = np.ones([self.Np])/self.Np
                # theta_new = np.zeros([self.Np])
                K_new = self.calc_K(T, P, x_new)
                alpha_new = np.ones([self.Np]) / self.Np
                theta_new = np.zeros([self.Np])
                if verbose:
                    print('Changed reference phase')
                
            # Set old values using copy 
            # (NOT direct assignment due to 
            # Python quirkiness of memory indexing)
            alpha_old = alpha_new.copy()
            theta_old = theta_new.copy()
            x_old = x_new.copy()
            K_old = K_new.copy()
            
            # Print a bunch of crap if desired.
            if verbose:
                print('\nIteration no: ', itercount)
                print('alpha = ', alpha_new)
                print('theta = ', theta_new)
                print('x = \n', x_new)
                print('K = \n', K_new)
                print('Composition error: ', x_error)
                print('objective function error: ', Obj_error)

        if verbose:
            print('\nElapsed time =', time.time() - tstart, '\n')
        
        self.K_calc = K_new.copy()
        self.x_calc = x_new.copy()
        self.alpha_calc = alpha_new.copy()
        self.theta_calc = theta_new.copy()
        self.completed = True

        values = [x_new, alpha_new, K_new, itercount, error]
        return values


    def calc_x(self, z, alpha, theta, K, T, P):
        """Composition of each component in each phases

        Parameters
        ----------
        z : list, numpy array
            Total composition of each component with size Nc
        alpha : list, numpy array
            Molar phase fractions with size Np
        theta : list, numpy array
            Stability of phases with size Np
        K : list, numpy array
            Partition coefficients for each component
            in each phase with size Nc x Np
        T : float
            Temperature in Kelvin
        P : float
            Pressure in bar

        Returns
        ----------
        x : numpy array
            Composition of each component in each phase
            at fixed alpha and theta with size Nc x Np
        """
        if type(z) != np.ndarray:
            z = np.asarray(z)
        if type(alpha) != np.ndarray:
            alpha = np.asarray(alpha)
        if type(theta) != np.ndarray:
            theta = np.asarray(theta)
        if type(K) != np.ndarray:
            K = np.asarray(K)
  
        x_numerator = z[:, np.newaxis]*K*np.exp(theta[np.newaxis, :])
        x_denominator = 1 + np.sum(
                alpha[np.newaxis, :]*(K*np.exp(theta[np.newaxis, :]) - 1),
                axis=1)
        x_mat = x_numerator/x_denominator[:, np.newaxis]

        #TODO : Figure out why this is unused!
        fug_mat = self.calc_fugacity(T, P, x_mat)
        for hyd_phase, ind in self.hyd_phases.items():
            x_mat[:, ind] = self.fug_list[ind].hyd_comp()
        x = np.minimum(1, np.abs(x_mat))
        x = x / np.sum(x, axis=0)[np.newaxis, :]
        return x

    def calc_K(self, T, P, x_mat):
        """Partition coefficients of each component in each phase

        Parameters
        ----------
        T : float
            Temperature in Kelvin
        P : float
            Pressure in bar
        x_mat : numpy array
            Composition of each component in each phase

        Returns
        ----------
        K : numpy array
            Partition coefficient matrix of each component
            in each phase at fixed alpha and theta with size Nc x Np
        """
        fug_mat = self.calc_fugacity(T, P, x_mat)
        K_mat = np.ones_like(x_mat)
        for ii, phase in enumerate(self.phases):
            if phase != self.ref_phase:
                K_mat[:, ii] = (fug_mat[:, self.ref_ind]/fug_mat[:, ii]
                                * x_mat[:, ii]/x_mat[:, self.ref_ind])
        K = np.real(np.abs(K_mat))
        return K


    def calc_fugacity(self, T, P, x_mat):
        """Fugacity of each component in each phase

        Parameters
        ----------
        T : float
            Temperature in Kelvin
        P : float
            Pressure in bar
        x_mat : numpy array
            Composition of each component in each phase

        Returns
        ----------
        fug_out : numpy array
            Fugacity matrix of each component in each phase
            at fixed alpha and theta with size Nc x Np
        """
        fug_out = np.zeros_like(x_mat)
        for ii, phase in enumerate(self.phases):
            if phase == 'aqueous':
                fug_out[:,ii] = self.fug_list[ii].calc(self.compobjs,
                                                       T,
                                                       P,
                                                       x_mat[:,ii])

            elif phase == 'vapor' or phase == 'lhc':
                fug_out[:,ii] = self.fug_list[ii].calc(self.compobjs,
                                                       T,
                                                       P,
                                                       x_mat[:, ii],
                                                       phase=phase)
        # Update the reference phase fugacity, which cannot be hydrate.
        self.ref_fug = fug_out[:, self.ref_ind]
        self.ref_comp = x_mat[:, self.ref_ind]

        # Do this separately because we need the reference phase fugacity.
        for hyd_phase, ind in self.hyd_phases.items():
            fug_out[:, ind] = self.fug_list[ind].calc(self.compobjs,
                                                      T,
                                                      P,
                                                      [],
                                                      self.ref_fug)
        return fug_out

    def find_alphatheta_min(self, z, alpha0, theta0, K, print_iter_info=False):
        """Algorithm for determining objective function minimization at fixed K

        Parameters
        ----------
        z : list, numpy array
            Molar fraction of each component with size Nc
        alpha0 : list, numpy array
            Initial molar phase fraction with size Np
        theta0 : list, numpy array
            Initial molar phase stability with size Np
        K : list, numpy array
            Partition coefficient matrix with size Nc x Np
        print_iter_info : bool
            Flag to print minimization progress

        Returns
        ----------
        new_values : list
            Result of gibbs energy minimization
            new_values[0] : numpy array
                Molar phase fractions at gibbs energy minimum
                at fixed x and K with size Np
            new_values[0] : numpy array
                Phase stabilities at gibbs energy minimum
                at fixed x and K with size Np
        """
        if not hasattr(self, 'ref_ind'):
            self.ref_ind = 0

        # Set iteration parameters
        nres = 1e6
        ndx = 1e6
        TOL = 1e-8
        kmax = 100
        k = 0
        dx = np.zeros([2*self.Np])
        
        if type(z) != np.ndarray:
            z = np.asarray(z)
        if type(alpha0) != np.ndarray:
            alpha0 = np.asarray(alpha0)
        if type(theta0) != np.ndarray:
            theta0 = np.asarray(theta0)
        if type(K) != np.ndarray:
            K = np.asarray(K)

        # Mask arrays to avoid the reference phase.
        alf_mask = np.ones([2*self.Np], dtype=bool)
        theta_mask = np.ones([2*self.Np], dtype=bool)
        arr_mask = np.ones([self.Np], dtype=bool)
        arrdbl_mask = np.ones([2*self.Np], dtype=bool)
        mat_mask = np.ones([2*self.Np, 2*self.Np], dtype=bool)

        # Populate masked arrays for 4 different types.
        # Mask reference phase in alpha array
        alf_mask[self.ref_ind] = 0
        alf_mask[self.Np:] = 0
        # Mask reference phase in theta array
        theta_mask[0:self.Np] = 0
        theta_mask[self.ref_ind + self.Np] = 0
        # Mask reference phase in x array
        arr_mask[self.ref_ind] = 0
        arrdbl_mask[self.ref_ind] = 0
        arrdbl_mask[self.ref_ind + self.Np] = 0
        # Mask reference phase rows and columns in jacobian matrix
        mat_mask[self.ref_ind, :] = 0
        mat_mask[self.ref_ind + self.Np, :] = 0
        mat_mask[:, self.ref_ind] = 0
        mat_mask[:, self.ref_ind + self.Np] = 0

        # Define 'x' as the concatenation of alpha and theta
#        x = np.concatenate((alpha0, theta0))

        alpha_old = alpha0.copy()
        theta_old = theta0.copy()
        alpha_new = alpha_old.copy()
        theta_new = theta_old.copy()
        
        # Iterate until converged
        while nres > TOL and ndx > TOL/100 and k < kmax:

            # Solve for change in variables using non-reference phases
            res = objective(z, alpha_old, theta_old, K)
            J = jacobian(z, alpha_old, theta_old, K)
            J_mod = J[mat_mask].reshape([2*(self.Np - 1), 2*(self.Np - 1)])
            res_mod = res[arrdbl_mask]
            # try:
            #     dx_tmp = -np.linalg.solve(J_mod, res_mod)
            # except:
            #     dx_tmp = -np.matmul(np.linalg.pinv(J_mod), res_mod)
            dx_tmp = -np.matmul(np.linalg.pinv(J_mod), res_mod)



            # Populate dx for non-reference phases
            dx[arrdbl_mask] = dx_tmp

            # Determine error
            nres = np.linalg.norm(res)
            ndx = np.linalg.norm(dx)

            # Adjust alpha using a maximum change of
            # the larger of 0.5*alpha_i or 0.01.
            alpha_new[arr_mask] = (alpha_old[arr_mask]
                           + np.sign(dx[alf_mask])
                             * np.minimum(np.maximum(1e-2, 0.5*alpha_new[arr_mask]),
                                          np.abs(dx[alf_mask])))
            
            # Limit alpha to exist between 0 and 1 and adjust alpha_{ref_ind}
            alpha_new[arr_mask] = np.minimum(1,
                                     np.maximum(0, alpha_new[arr_mask]))
            alpha_new[self.ref_ind] = np.minimum(1,
                                         np.maximum(0,
                                                    1 - np.sum(alpha_new[arr_mask])))
            alpha_new = alpha_new/np.sum(alpha_new)

            # Adjust theta and limit it to a positive value
            theta_new[arr_mask] = theta_old[arr_mask] + dx[theta_mask]
            theta_new[arr_mask] = np.maximum(0, theta_new[arr_mask])

            # Use technique of Gupta to enforce that theta_i*alpha_i = 0
            # or that theta_i = alpha_i = 1e-10, which will kick one of them
            # to zero on the next iteration.
            alpha_new[alpha_new < 1e-10] = 0
            theta_new[theta_new < 1e-10] = 0
            change_ind = (((alpha_new < 1e-10)
                           & (theta_new == 0))
                          | ((alpha_new == 0)
                             & (theta_new < 1e-10))
                          | ((alpha_new < 1e-10)
                              & (theta_new < 1e-10)))
            alpha_new[change_ind] = 1e-10
            theta_new[change_ind] = 1e-10

            k += 1
            
            alpha_old = alpha_new.copy()
            theta_old = theta_new.copy()

            if print_iter_info:
                print('k=', k)
                print('error=', nres)
                print('param change=', ndx)

        new_values = [alpha_new, theta_new]
        return new_values

    # Initialize the partition coefficient matrix based on P, T and components
    # Provide the option to specify the feed to predict the appropriate
    # reference phase or the option to specify the reference phase explicitly.
    def make_ideal_K_mat(self, compobjs, T, P, **kwargs):
        """Ideal partition coefficient initialization routine

        Parameters
        ----------
        compobjs : list
            List of components
        T : float
            Temperature in Kelvin
        P : float
            Pressure in bar

        Returns
        ----------
        K_mat_ref : numpy array
            Ideal partition coefficients for
            each component in each phase with size Nc x Np
        """
        if not hasattr(compobjs, '__iter__'):
            compobjs = [compobjs]

        if kwargs is not None:
            if ('z' or 'feed') in kwargs:
                try:
                    self.set_feed(kwargs['z'])
                except:
                    self.set_feed(kwargs['feed'])
            elif 'ref_phase' in kwargs:
                self.ref_phase = kwargs['ref_phase']
                if self.ref_phase not in self.phases:
                    self.phases.append(self.ref_phase)
                self.set_ref_index()

            if 'phases' in kwargs:
                phase_return = kwargs['phases']
            else:
                phase_return = self.phases
        else:
            phase_return = self.phases

        K_all_mat = make_ideal_K_allmat(compobjs, T, P)
        K_mat_ref = np.zeros([len(compobjs), len(phase_return)])

        K_refdict = K_transform[self.ref_phase]
        for ii, phase in enumerate(phase_return):
            trans_tuple = K_refdict[phase]
            if type(trans_tuple) is int:
                if trans_tuple == 9:
                    K_mat_ref[:, ii] = 1
                else:
                    K_mat_ref[:, ii] = K_all_mat[:, trans_tuple]
            else:
                if type(trans_tuple[0]) is int:
                    if type(trans_tuple[1]) is int:
                        if trans_tuple[0] == 9:
                            K_mat_ref[:, ii] = 1/K_all_mat[:, trans_tuple[1]]
                        else:
                            K_mat_ref[:, ii] = (K_all_mat[:, trans_tuple[0]]
                                                / K_all_mat[:, trans_tuple[1]])
                    else:
                        K_mat_ref[:, ii] = (
                                K_all_mat[:, trans_tuple[0]]
                                / (K_all_mat[:, trans_tuple[1][0]]
                                   * K_all_mat[:, trans_tuple[1][1]]))
                else:
                    K_mat_ref[:, ii] = (
                            (K_all_mat[:, trans_tuple[0][0]]
                             * K_all_mat[:, trans_tuple[0][1]])
                            / K_all_mat[:, trans_tuple[1]])
        return K_mat_ref


    # TODO fill in documentation here...
    def incipient_calc(self, T, P):
        if (self.h2oexists) and (len(self.feed) > 2):
            z_wf = []
            comp_wf = []
            wf_comp_map = {}
            for ii in range(len(self.feed)):
                if ii != self.h2oind:
                    z_wf.append(self.feed[ii])
                    comp_wf.append(self.compname[ii])
                    wf_comp_map.update({ii: len(z_wf) - 1})
            z_wf = np.asarray(z_wf) / sum(z_wf)
            vlhc_flash = FlashController(comp_wf, phases=['vapor', 'lhc'])
            vlhc_output = vlhc_flash.main_handler(vlhc_flash.compobjs, z=z_wf, T=T, P=P)

            z_aqv = []
            z_aqlhc = []
            for ii in range(self.Nc):
                if ii == self.h2oind:
                    z_aqv.append(self.feed[ii])
                    z_aqlhc.append(self.feed[ii])
                else:
                    z_aqv.append(vlhc_output[0][wf_comp_map[ii], 0] / (1 - self.feed[self.h2oind]))
                    z_aqlhc.append(vlhc_output[0][wf_comp_map[ii], 1] / (1 - self.feed[self.h2oind]))

            if 'vapor' in self.phases:
                aqv_flash = FlashController(self.compname, phases=['aqueous', 'vapor'])
                aqv_output = aqv_flash.main_handler(aqv_flash.compobjs, z=np.asarray(z_aqv), T=T, P=P)

            if 'lhc' in self.phases:
                aqlhc_flash = FlashController(self.compname, phases=['aqueous', 'lhc'])
                aqlhc_output = aqlhc_flash.main_handler(aqlhc_flash.compobjs, z=np.asarray(z_aqlhc), T=T, P=P)

            if 's1' in self.phases:
                aqs1_flash = FlashController(self.compname, phases=['aqueous', 's1'])
                aqs1_output = aqs1_flash.main_handler(aqs1_flash.compobjs, z=self.feed, T=T, P=P)

            if 's2' in self.phases:
                aqs2_flash = FlashController(self.compname, phases=['aqueous', 's2'])
                aqs2_output = aqs2_flash.main_handler(aqs2_flash.compobjs, z=self.feed, T=T, P=P)

            x_tmp = np.zeros([self.Nc, self.Np])
            for jj, phase in enumerate(self.phases):
                if phase == 'vapor':
                    x_tmp[:, jj] = aqv_output[0][:, 1]
                elif phase == 'lhc':
                    x_tmp[:, jj] = aqlhc_output[0][:, 1]
                elif phase == 'aqueous':
                    if vlhc_output[1][0] > vlhc_output[1][1]:
                        x_tmp[:, jj] = aqv_output[0][:, 0]
                    else:
                        x_tmp[:, jj] = aqlhc_output[0][:, 0]
                elif phase == 's1':
                    x_tmp[:, jj] = aqs1_output[0][:, 1]
                elif phase == 's2':
                    x_tmp[:, jj] = aqs2_output[0][:, 1]

            return x_tmp / x_tmp[:, self.ref_ind][:, np.newaxis]

        else:
            return self.make_ideal_K_mat(self.compobjs, T, P)



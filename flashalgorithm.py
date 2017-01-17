#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:58:38 2017

@author: kdarnell
"""

"""
Starting the general flash handler for iteration and calling
of EOS objects. I will prepare skeleton here first (01/11/17).
"""
import numpy as np
# Not sure if I want these classes in this file
import component_properties as cp
import AqHB_EOS as aq
import HvdWPM_EOS as h
import SRK_EOS as hc

class FlashController(object):
    # This the possible aliases for phases.
    # Keys are phases, values are aliases.
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

    # This is the ultimate hope, but right now the eos's are limited.
    # Keys are phases, values are different eos'.
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
                 P=1):

        self.T = T
        self.P = P
        self.ref_phase = None
        # Check that components exceed 1.
        if type(components) is str or len(components) == 1:
            raise ValueError("""
                    More than one component is necessary to run
                    flash algorithm.
                    """)
        elif type(components[0]) is str:
            # Use list or array of component names to populate a new list of
            # component objects
            self.compobjs = []
            for compname in components:
                self.compobjs.append(cp.Component(compname))
        elif isinstance(components[0], cp.Component):
            self.compobjs = components
        else:
            raise ValueError("""
                    Component are not properly defined. Pass a list of strings
                    or a list of component objects""")

        self.compname = []
        self.h2oexists = False
        self.h2oind = None
        for ii, comp in enumerate(self.compobjs):
            self.compname.append(comp.compname)
            if comp.compname == 'h2o':
                self.h2oexists = True
                self.h2oind = ii

        # Check that phases exceed 1
        if type(phases) is str or len(phases) == 1:
            raise ValueError("""
                    More than one phase is necessary to run
                    flash algorithm.""")
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
                    raise ValueError("""
                            One or more phases are repeated. Distinct phases
                            are necessary to run flash algorithm.""")

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
        fug_list = []
        hyd_phases = []
        for ii, phase in enumerate(self.phases):
            if phase in ('aqueous', 's1', 's2'):
                if self.h2oexists:
                    if phase == 'aqueous':
                        aq_obj = aq.HegBromEos(self.compobjs, self.T, self.P)
                        fug_list.append(aq_obj)
                        self.ref_phase = phase
                    else:
                        hyd_phases.append((phase, ii))
                        h_obj = h.HvdwpmEos(self.compobjs, self.T, self.P,
                                            structure=phase)
                        fug_list.append(h_obj)
                else:
                    self.phases.remove(phase)

            elif phase == 'ice':
                print('Not currently supported for an ice phase.')
                self.phases.remove(phase)
                if not hyd_phases:
                    for hyd_phase in hyd_phases:
                        hyd_phase[1] += -1
            else:
                hc_obj = hc.SrkEos(self.compobjs, self.T, self.P)
                fug_list.append(hc_obj)
                if self.ref_phase is None and phase == 'vapor':
                    self.ref_phase = 'vapor'
                
        self.fug_list = fug_list
        self.hyd_phases = hyd_phases
        
    def set_feed(self, z):
        if len(z) != len(self.compobjs):
            raise ValueError('Feed fraction has different dimension than'
                             + ' initial component list!')
        elif self.h2oexists:
            if z[self.h2oind] > 0.8:
                self.ref_phase = 'aqueous'
        else:
            # This is true for now. In practice, this sometims needs to be 
            # lhc, but we will handle that within a different method.
            self.ref_phase = 'vapor'
        
        self.feed = z 

            
    def calc_K(self, T, P, x_mat):
        fug_mat = self.calc_fugacity(T, P, x_mat)
        K_mat = np.ones_like(x_mat)
        for ii, phase in enumerate(self.phases):
            if phase != self.ref_phase:
                K_mat[:, ii] = (self.ref_fug/fug_mat[:, ii]
                                * x_mat[:, ii]/self.ref_comp)
                
        return K_mat
            
    
    # x_mat will be a matrix of the compositions in each phase.
    # It should be Nc x Np
    def calc_fugacity(self, T, P, x_mat):
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
            if self.ref_phase == phase:
                    self.ref_fug = fug_out[:,ii]
                    self.ref_comp = x_mat[:,ii]

        # Do this separetly because we need the reference phase fugacity.
        for hyd_phase in self.hyd_phases:
            phase = hyd_phase[0]
            ind = hyd_phase[1]
            fug_out[:,ind] = self.fug_list[ind].calc(self.compobjs, 
                                                     T, 
                                                     P,
                                                     [], 
                                                     self.ref_fug)
        return fug_out









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
# Not sure if I want these classes in this file
import component_properties as cp  

class FlashController(object):
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
                 phases=('aqueous', 'vapor', 's1'),
                 eos=eos_default):

        # Check that components exceed 1.
        if type(components) is str or len(components) == 1:
            raise ValueError('More than one component is necessary to run '
                               + 'flash algorithm.')
        else:
            # Use list or array of component names to populate a new list of 
            # component objects
            self.compobjs = []
            for compname in components:
                self.compobjs.append(cp.Component(compname))
             
        # Check that phases exceed 1
        if type(phases) is str or len(phases) == 1:
            raise ValueError('More than one phase is necessary to run '
                               + 'flash algorithm.')  
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
                        raise ValueError(phase + ' is not a supported phase!!'
                                         + '\nConsult "FlashController.phase_menu" '
                                         + 'attribute for \nvalid phases and '
                                         + 'associated call strings.')
                else:
                    # Perhaps, I should just print warning that eliminates 
                    # duplicate phases.
                    raise ValueError('One or more phases are repeated. ' 
                             + 'Distinct phases are necessary to run ' 
                             + 'flash algorithm.')
              
        # Allow option for changing the default eos for any given phase.
        # Check to make sure the eos is supported and that the phase being
        # modified is a valid phase.s
        if eos != self.eos_default:
            if type(eos) is not dict:
                raise TypeError('"eos" specifies equation of state to be used ' 
                                + 'with a specific phase. Pass a dictionary as ' 
                                + 'eos (e.g., eos={"vapor": "pr"}).')
            else:
                self.eos = self.eos_default.copy()
                for phase_tmp, eos_tmp in eos.items():
                    if phase_tmp in self.eos_default.keys():
                        if eos_tmp in self.eos_menu[phase_tmp]:
                            self.eos[phase_tmp] = eos_tmp
                        else:
                            raise ValueError(eos_tmp + ' is not a valid eos for ' 
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
                                            + valid_phase + ' phase.'
                                    )
                        if not phase_found:
                            raise ValueError(phase_tmp + ' is not a valid' 
                                                 + ' phase.')
        else:
            self.eos = eos
        
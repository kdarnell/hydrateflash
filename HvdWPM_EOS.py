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

# Parent class for a series of slightly different equations of state
# --however, they will all be based on van der Waals-Platteeuw
# The parent class is desigend to reduce redundancy. Some functions
# will modified in the child class.
class HydrateEos(object):
    def __init__(self, compobjs, T, P, structure='s1'):
        self.description = (
            'Object for calculating fugacity of water in hydrate' +
            ' with a mixture of gases.'
        )
        # Set up arrays and matrices for future calculation
        self.Nc = len(compobjs)
        self.compobjs = compobjs
        self.gwbeta_RT_cons = np.zeros(1)
        self.volume_int = np.zeros(1)
        self.volume = np.zeros(1)
        self.activity = np.zeros(1)
        self.delta_mu_RT = np.zeros(1)
        self.gw0_RT = np.zeros(1)
        self.compnames = [c.compname for c in compobjs]
        self.h2oind = [ii for ii, name in enumerate(self.compnames)
                       if name == 'h2o'][0]
        self.Hs = HydrateStructure(structure)
        self.make_constant_mats(compobjs, T, P)
        self.alt_fug_vec = np.zeros(self.Nc)
        
        # Transfer everything from the eos-specific dictionary to
        # the Hs object for use in the rest of the class
        # Note: 'eos_key' will be defined within sub-class
        for k,v in dict.items(self.Hs.eos[self.eos_key]):
            setattr(self.Hs, k, v)
        
    def make_constant_mats(self, compobjs, T, P):
        # Create all values that are invariant wrt T and P
        self.T = T
        self.P = P
        self.gw0_RT = compobjs[self.h2oind].gibbs_ideal(T, P)
        
        # Add additional stuff for specific EOS's here
        return self
        
    def langmuir_consts(self, compobjs, T, P):
        # Each child EOS will have a different way of calculating this.
        C_small = np.zeros(self.Nc)
        C_large = np.zeros(self.Nc)
        
        return C_small, C_large
        
    def calc_langmuir(self, C):
        Y = np.zeros(self.Nc)
        denominator = (1.0 + np.sum(C*self.alt_fug_vec))
        for ii, comp in enumerate(self.compobjs):
            if comp.compname != 'h2o':
                Y[ii] = C[ii]*self.alt_fug_vec[ii]/denominator
            else:
                Y[ii] = 0
        return Y

    def delta_mu_func(self, compobjs, T, P, x):
        C_small, C_large = self.langmuir_consts(compobjs, T, P)
        self.Y_small = self.calc_langmuir(C_small)
        self.Y_large = self.calc_langmuir(C_large)

        delta_mu = (
            self.Hs.Nm['small']*np.log(1-np.sum(self.Y_small)) 
            + self.Hs.Nm['large']*np.log(1-np.sum(self.Y_large))
        )/self.Hs.Num_h2o
        return delta_mu
        
    def fugacity_calc(self, compobjs, T, P, x, alt_fug):
        pass
        
    def calc(self, compobjs, T, P, x, alt_fug):
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
                
        
        self.delta_mu_func(compobjs, T, P, x)
        fug = self.fugacity_calc(compobjs, T, P, x, alt_fug)
        return fug
        

class HvdwpmEos(HydrateEos):
    cp = {'a0': 0.735409713*R,
          'a1': 1.4180551e-2*R,
          'a2': -1.72746e-5*R,
          'a3': 63.5104e-9*R}
    eos_key = 'hvdwpm'
      
    def __init__(self, compobjs, T, P, structure='s1'):
        # Inheret all prroperties from HydrateEos
        HydrateEos.__init__(self, compobjs, T, P, structure='s1')
        self.kappa = np.zeros(1)
        self.kappa0 = self.Hs.kappa
        self.a0_cubed = self.volume_func(self.Hs.a0_ast)
        
        for ii, comp in enumerate(compobjs):
            self.kappa_vec[ii] = comp.kappa[self.Hs.hydstruc]
            self.rep_sm_vec[ii] = comp.rep[self.Hs.hydstruc]
            self.rep_lg_ve[ii] = comp.rep[self.Hs.hydstruc]
            self.D_vec[ii] = comp.diam
    
    def volume_func(self, a_param):
        if self.Hs.hydstruc != 'sH':
            v = 6.0221413e23/self.Hs.Num_h2o/1e24*(a_param)**3
        else:
            v = self.Hs.v0
            
        return v

#        if self.Hs.hydstruc != 'sH':
#            self.v_H_0 = 6.0221413e23/self.Hs.Num_h2o/1e24*(a_param)**3
#        else:
#            self.v_H_0 = self.Hs.v0              
            
    
    def make_constant_mats(self, compobjs, T, P):
        # Do standard stuff
        HydrateEos.make_constant_mats(self, compobjs, T, P)
        
        # TODO: Work out what else goes here...?
        # Multiply the following by 'volume'
        self.vol_int = (lambda v,kappa:(
            v*np.exp(self.Hs.alf[1]*(T - T_0)+ self.Hs.alf[2]*(T - T_0)**2 
                     + self.Hs.alf[3]*(T - T_0)**3 - kappa*(P - P_0))
        )/(-kappa))
        
        self.gwbeta_RT = (
            (self.Hs.gw_0beta/(R*T_0) - (12*T*self.Hs.hw_0beta 
             - 12*T_0*self.Hs.hw_0beta + 12*T_0**2*self.cp['a0'] 
             + 6*T_0**3*self.cp['a1'] + 4*T_0**4*self.cp['a2'] 
             + 3*T_0**5*self.cp['a3'] - 12*T*T_0*self.cp['a0'] 
             - 12*T*T_0**2*self.cp['a1'] + 6*T**2*T_0*self.cp['a1'] 
             - 6*T*T_0**3*self.cp['a2'] + 2*T**3*T_0*self.cp['a2'] 
             - 4*T*T_0**4*self.cp['a3'] + T**4*T_0*self.cp['a3'] 
             + 12*T*T_0*self.cp['a0']*np.log(T)
             - 12*T*T_0*self.cp['a0']*np.log(T_0)))/(12*R*T*T_0)
            + (self.vol_int(P,self.a0_cubed,self.kappa0,T) 
               - self.vol_int(P_0,self.a0_cubed,self.kappa0,T))*1e-1/(R*T)
        )
        return self
        
    def unknown_func(self):
        small_const = (
            (1 + self.Hs.etam['small']/self.Hs.Num_h2o)*self.Y_small
            / (1 + (self.Hs.etam['small']/self.Hs.Num_h2o)*self.Y_small)
        )
        
        if self.Nc>2:
            self.repulsive_small = np.sum(small_const)
        else:
            self.repulsive_small = (small_const*np.exp(
                self.D_vec - np.sum(self.Y_small*self.D_vec)
            ))
        
        self.repulsive_large = (
            (1 + self.Hs.etam['large']/self.Hs.Num_h2o)*self.Y_large
            / (1 + (self.Hs.etam['large']/self.Hs.Num_h2o)*self.Y_large)
        )
      
        return self
        
    def kappa_func(self):
        if self.Nc>2:
            kappa = np.sum(self.kappa_vec)
        else:
            kappa = np.sum(self.kappa_vec*self.Y_large)
            
        return kappa
        
    def activity_func(self, v_H_0):
        kappa_wtavg = self.kappa_func(self)
        activity = (
            (v_H_0 - self.a0_cubed)/R*(self.Hs.a_fit/T_0 
                                       + self.Hs.b_fit*(1/T - 1/T_0))
            + ((self.vol_int(P,v_H_0,kappa_wtavg,T) 
                - self.vol_int(P,self.a0_cubed,self.kappa0,T)) 
              - (self.vol_int(P_0,v_H_0,kappa_wtavg,T) 
                 - self.vol_int(P_0,self.a0_cubed,self.kappa0,T)))*1e-1/(R*T)
        )

        return activity
        
        
    def convert_a_param(self):
        v = (self.Hs.a0_ast 
             + (self.Hs.Nm['small']
                * np.sum(self.repulsive_small*self.rep_sm_vec)) 
             + (self.Hs.Nm['large']
                *np.sum(self.repulsive_large*self.rep_lg_vec))
             )
        return v
        
    def fugacity_calc(self, compobjs, T, P, x, alt_fug):     
        self.alt_fug_vec = alt_fug
        # It may be wise to insert a while loop to monitor changes C
        delta_mu_RT = self.delta_mu_func(compobjs, T, P, x)
        v_H = self.convert_a_param()
        activity = self.activity_func(v_H)
        mu_H_RT = self.gwbeta_RT + activity + delta_mu_RT
        fug = self.exp(mu_H_RT - self.gw_0_RT)
        return fug
         
    
        
# Properties of each hydrate structure necessary for further calculation.         
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
            self.eos = {'hvdwpm': {'v0': 22.7712,
                                   'kappa': 3e-5,
                                   'a0_ast': 11.99245,
                                   'gw_0beta': -235537.85,
                                   'hw_0beta': -291758.77,
                                   'a_fit': 25.74,
                                   'b_fit': -481.32,
                                   'alf': {1: 3.38496e-4, 
                                           2: 5.40099e-7, 
                                           3: -4.76946e-11}}}
            self.Nm = {'small': 2, 'large': 6}
            self.etam = {'small': 20, 'large': 24}
            self.Num_h2o = 46
        elif self.hydstruc == 's2':
            self.eos = {'hvdwpm': {'v0': 22.9456,
                                   'kappa': 3e-6,
                                   'a0_ast': 17.10000,
                                   'gw_0beta': -235627.53,
                                   'hw_0beta': -292044.10,
                                   'a_fit': 260,
                                   'b_fit': -68.64,
                                   'alf': {1: 2.029776e-4, 
                                           2: 1.851168e-7, 
                                           3: -1.879455e-10}}}

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

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
from scipy.optimize import minimize, newton_krylov, fsolve

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
        self.Nc = len(self.compobjs)

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
        fug_list = list()
        hyd_phases = dict()
        for ii, phase in enumerate(self.phases):
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
                if self.ref_phase is None and phase == 'vapor':
                    self.ref_phase = 'vapor'

        self.fug_list = fug_list
        self.hyd_phases = hyd_phases
        self.nonhyd_phases = [ii for ii in range(len(self.phases))
                              if ii not in self.hyd_phases.values()]
        self.Np = len(self.phases)

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
        
    def set_ref_index(self):
        self.ref_ind = [self.ref_phase == phase for phase in self.phases]
        

    # TODO Check the next three functions against Matlab output
    def calc_x(self, z, alpha, theta, K):
        # z, alpha, and theta are vectors.
        # z is length Nc, alpha and theta are length Np
        # K is matrix of size Nc x Np
        x_mat = np.zeros([len(self.compobjs), len(self.phases)])
        x_numerator = np.zeros([len(self.compobjs), len(self.phases)])


        for ii, comp in enumerate(self.compobjs):
            x_denominator = 1.0
            for kk, phase in enumerate(self.phases):
                if phase not in ('s1', 's2'):
                    x_numerator[ii, kk] = z[ii]*K[ii, kk]*np.exp(theta[kk])

                x_denominator += alpha[kk]*(K[ii, kk]*theta[kk] - 1.0)
            x_mat[ii, self.nonhyd_phases] = (
                x_numerator[ii, self.nonhyd_phases]/x_denominator
            )

            for hyd_phase, ind in self.hyd_phases.items():
                x_mat[:, ind] = self.fug_list[ind].hyd_comp()

        return x_mat

    def Objective(self, z, alpha, theta, K):
        # z, alpha, and theta are vectors.
        # z is length Nc, alpha and theta are length Np
        # K is matrix of size Nc x Np

        # Making use of numpy's broadcasting capabilities to implicitly
        # reshape and 'tile' matrices.
        E_numerator = z[:, np.newaxis]*(K*np.exp(theta[np.newaxis, :]) - 1)
        E_denomintor = 1 + np.sum(
                alpha[np.newaxis, :]*(K*np.exp(theta[np.newaxis, :]) - 1), 
                axis=1)
        E_cost = np.sum(E_numerator/E_denomintor[:, np.newaxis], axis=0)
        Y_cost = alpha*theta/(alpha + theta)
        Cost = np.concatenate((E_cost, Y_cost))
        return Cost
    
    def Jacobian(self, z, alpha, theta, K):
        # z, alpha, and theta are vectors.
        # z is length Nc, alpha and theta are length Np
        # K is matrix of size Nc x Np
        
        
        # Making use of numpy's broadcasting capabilities to implicitly
        # reshape and 'tile' matrices.
        Stability_mat = (K*np.exp(theta[np.newaxis, :]) - 1.0)
        J_alphaNumerator = (z[:, np.newaxis, np.newaxis]
                     * Stability_mat[:, :, np.newaxis]
                     * Stability_mat[:, np.newaxis, :])

        J_thetaNumerator = (z[:, np.newaxis, np.newaxis]
                     * Stability_mat[:, :, np.newaxis]
                     * K[:, np.newaxis,:]
                     * alpha[np.newaxis, np.newaxis, :]
                     * np.exp(theta[np.newaxis, np.newaxis :]))
        
        Denomiator = (1.0 + (np.sum(alpha[np.newaxis, :]
                             * Stability_mat, axis=1)))**2
        
        Jac_alphaCost = -np.sum(J_alphaNumerator
                                / Denomiator[:, np.newaxis, np.newaxis], 
                                axis = 0)
        
        Jac_thetaCost = -np.sum(J_thetaNumerator
                                / Denomiator[:, np.newaxis, np.newaxis], 
                                axis = 0)
        
        Diag_denom = 1.0 + np.sum((K*np.exp(theta[np.newaxis, :]) - 1.0)
                                   * alpha[np.newaxis,:], axis=1)
        Diag = np.sum(z[:, np.newaxis]*K*np.exp(theta[np.newaxis, :]) 
                      / Diag_denom[:, np.newaxis], 
                      axis=0)        
        Jac_thetaCost += np.diag(Diag)

        Jacobian_Cost = np.concatenate((Jac_alphaCost, Jac_thetaCost), axis=1)
        
        
        Jac_alphaStability = (theta/(alpha + theta) 
                              - alpha*theta/(alpha + theta)**2)
        Jac_thetaStability = (alpha/(alpha + theta) 
                              - alpha*theta/(alpha + theta)**2)
        Jacobiaon_Stability = np.concatenate((np.diag(Jac_alphaStability),             
                                              np.diag(Jac_thetaStability)),
                                             axis=1)
        Jacobian = np.concatenate((Jacobian_Cost, Jacobiaon_Stability), axis=0)
        
        return Jacobian
        
    def Stability_func(self, alpha, theta):
        Y = alpha*theta/(alpha + theta)
        return Y 

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
        for hyd_phase, ind in self.hyd_phases.items():
            fug_out[:, ind] = self.fug_list[ind].calc(self.compobjs,
                                                      T,
                                                      P,
                                                      [],
                                                      self.ref_fug)
        return fug_out




    # TODO: This entire method is not working. I'm not sure how to implement 
    # scipy's algorithms to solve this appropriatley. It is probably a good
    # idea to use my original algorithm or at the least compare the results
    # because it may be apppropriately solving things, but not applying the 
    # constraints correctly.
#    def find_alphatheta_min(self, z, alpha0, theta0, K):
#        # z is always constant
#        # K will remain constant
#        # alpha0 is the starting alpha
#        # theta0 is the starting theta
#        # Constaints (1): sum(alpha[:]) - 1 =  0, (2): 0 <= alpha[i] <= 1
#        # xor(alpha[i] == 0, theta[i] == 0)
#        # Objective funciton: self.Objective
#        # Jacobian function: self.Jacobian
#        # We will convert x = np.concatenate(alpha, theta).
#        # Thus, alpha is x[0:self.Np] and theta is x[self.Np:]
#        beta = 0.01
#        
#        Objective = lambda x: np.abs(self.Objective(z, x[0:self.Np], x[self.Np:], K)) + beta*(1.0 - np.sum(x[0:]))
##        Jacobian = lambda x: self.Jacobian(z, x[0:self.Np], x[self.Np:], K)
#        
#        initial_guess = np.concatenate((alpha0, theta0))
#        result = fsolve(Objective, initial_guess)
#        return result


    def find_alphatheta_min(self, z, alpha0, theta0, K):
        
        Objective = lambda x: self.Objective(z, x[0:self.Np], x[self.Np:], K) 
        Jacobian = lambda x: self.Jacobian(z, x[0:self.Np], x[self.Np:], K)
        
        
        if ~hasattr(self, 'ref_ind'):
            self.ref_ind = 0
            
        nres = 1e6
        ndx = 1e6
        TOL = 1e-6
        kmax = 500
        k = 0
        
        
        alf_mask = np.ones([2*self.Np], dtype=bool)
        theta_mask = np.ones([2*self.Np], dtype=bool)
        arr_mask = np.ones([2*self.Np], dtype=bool)
        mat_mask = np.ones([2*self.Np,2*self.Np], dtype=bool)

        
        alf_mask[self.ref_ind] = 0
        alf_mask[self.Np:] = 0
        theta_mask[0:self.Np] = 0
        theta_mask[self.ref_ind + self.Np] = 0
        arr_mask[self.ref_ind] = 0
        arr_mask[self.ref_ind + self.Np] = 0
        mat_mask[self.ref_ind, :] = 0
        mat_mask[self.ref_ind + self.Np, :] = 0
        mat_mask[:, self.ref_ind] = 0
        mat_mask[:, self.ref_ind + self.Np] = 0

        x = np.concatenate((alpha0, theta0))
        
        while nres > TOL and ndx > TOL and k < kmax:
            dx = np.zeros([2*self.Np])
            res = Objective(x)
            J = Jacobian(x)
            dx_tmp = -np.matmul(np.linalg.pinv(
                                J[mat_mask].reshape([2*(self.Np - 1), 
                                                     2*(self.Np - 1)])), 
                            res[arr_mask])
            dx[arr_mask] = dx_tmp
            nres = np.linalg.norm(res)
            
            x[alf_mask] = (x[alf_mask] 
                           + np.sign(dx[alf_mask]) 
                           * np.minimum(np.maximum(1e-2,x[alf_mask]*0.5),
                                        np.abs(dx[alf_mask])))
            x[alf_mask] = np.minimum(1,np.maximum(0,x[alf_mask]))
            x[self.ref_ind] = np.minimum(1,np.maximum(1e-3,1 - np.sum(x[alf_mask])))

            
            x[theta_mask] += dx[theta_mask]
            x[theta_mask] = np.maximum(0,x[theta_mask])
            change_ind = (((x[0:self.Np] < 1e-10) & (x[self.Np:] == 0)) |
                    ((x[0:self.Np] < 1e-10) & (x[self.Np:] == 0)) |
                    ((x[0:self.Np] < 1e-10) & (x[self.Np:] < 1e-10)))
            adjust_ind = np.append(change_ind, change_ind)
            x[adjust_ind] = 1e-10 
            x[self.Np + self.ref_ind] = 0
            x[alf_mask] = x[alf_mask]*(x[theta_mask]<=1e-10)
            x[theta_mask] = x[theta_mask]*(x[alf_mask]<=1e-10)
            
            
            ndx = np.linalg.norm(dx)
            k += 1
            print('k=', k)
        
        return x

        



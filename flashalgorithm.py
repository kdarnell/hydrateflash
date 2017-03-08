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
import time

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
        self.set_ref_index()

    def set_feed(self, z, setref=True):
        self.feed = np.asarray(z)
        
        if setref:
            if len(z) != len(self.compobjs):
                raise ValueError("""Feed fraction has different dimension than
                                    initial component list!""")
            elif self.h2oexists:
                if z[self.h2oind] > 0.8:
                    self.ref_phase = 'aqueous'
            else:
                # This is true for now. In practice, this sometims needs to be
                # lhc, but we will handle that within a different method.
                self.ref_phase = 'vapor'
        
    def set_phases(self, phases):
        self.phases = phases

    def set_ref_index(self):
        if self.ref_phase not in self.phases:
            self.phases.append(self.ref_phase)
        self.ref_ind = [ii for ii, phase in enumerate(self.phases)
                        if phase == self.ref_phase].pop()
        
    def change_ref_phase(self):
        self.ref_phases_tried.append(self.ref_phase)
        self.ref_phase = [phase for phase in self.phases
                          if phase not in self.ref_phases_tried 
                          and phase not in ['s1', 's2']].pop(0)
        self.set_ref_index()
        
    def main_handler(self, compobjs, z, T, P, 
                     K_init='default', verbose=False, 
                     initialize=True, run_diagnostics=False,
                     **kwargs):
        
        z = np.asarray(z / sum(z))
        self.set_feed(z)
        self.set_ref_index()
        self.ref_phases_tried = []
        print(self.ref_phase)
        
        if verbose:
            tstart = time.time()
        
        if K_init != 'default':
            # Add more code to allow the specification of
            print('K is not the default')
            K_0 = K_init
        else:
            K_0 = self.make_ideal_K_mat(compobjs, T, P)
            alpha_0 = np.ones([self.Np])/self.Np
            theta_0 = np.zeros([self.Np])
            
        if type(z) != np.ndarray:
            z = np.asarray(z)
            
        if type(K_init)!= np.ndarray:
            K_init = np.asarray(K_init)
            
        if initialize or not hasattr(self, 'alpha_calc'):
#            newton_out = self.find_alphatheta_min(z, alpha_0, theta_0, K_0)
#            alpha_new = newton_out[0:self.Np]
#            theta_new = newton_out[self.Np:]
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
        iterlim = 50
        
        alpha_old = alpha_new.copy()
        theta_old = theta_new.copy()
        x_old = x_new.copy()
        K_old = K_new.copy()
        
        
        
        while error > TOL and itercount < iterlim:
            # Perform newton iteration to update alpha and theta at
            # a fixed x and K
#            newton_out = self.find_alphatheta_min(z, alpha_old, 
#                                                  theta_old, K_old)
#            alpha_new = newton_out[0:self.Np]
#            theta_new = newton_out[self.Np:]
            alpha_new, theta_new = self.find_alphatheta_min(z, alpha_old, 
                                                             theta_old, K_new)

            
            # Perform one iteration of successive substitution to update
            # x and K at the new alpha and theta.
#            x_update = self.calc_x(z, alpha_new, theta_new, K_old, T, P)
#            x_diff = x_update - x_new
#            x_new = x_new + np.sign(x_diff)*np.maximum(0.5*x_new, abs(x_diff))
            x_error = 1e6
            x_counter = 0
            while x_error > TOL*100 and x_counter < 1:
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
            Obj_error = np.sum(self.Objective(z, alpha_new, 
                                              theta_new, K_new))
            error = max(Obj_error, x_error)
            
            itercount += 1
            refphase_itercount += 1
            nan_occur = (np.isnan(x_new).any() or np.isnan(K_new).any() 
                         or np.isnan(alpha_new).any() or np.isnan(theta_new).any())
            
            if ((refphase_itercount > 10 
                and alpha_new[self.ref_ind] < 0.01) 
                or nan_occur) :
                self.change_ref_phase() 
                refphase_itercount = 0
                K_new = self.make_ideal_K_mat(compobjs, T, P)
                alpha_new = np.ones([self.Np])/self.Np
                theta_new = np.zeros([self.Np])
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
                print('Objective function error: ', Obj_error)

        if verbose:
            print('\nElapsed time =', time.time() - tstart, '\n')
        
        self.K_calc = K_new.copy()
        self.x_calc = x_new.copy()
        self.alpha_calc = alpha_new.copy()
        self.theta_calc = theta_new.copy()
            
        return [x_new, alpha_new, K_new, itercount, error]


    # TODO Check the next three functions against Matlab output
    # And change calc_x so that I make use of the np.newaxis utility!!
    def calc_x(self, z, alpha, theta, K, T, P):
        # z, alpha, and theta are vectors.
        # z is length Nc, alpha and theta are length Np
        # K is matrix of size Nc x Np
        
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
        
#        x_mat = np.zeros([self.Nc, self.Np])
#        x_numerator = np.zeros([self.Nc, self.Np])
#        for ii, comp in enumerate(self.compobjs):
#            x_denominator = 1.0
#            for kk, phase in enumerate(self.phases):
#                if phase not in ('s1', 's2'):
#                    x_numerator[ii, kk] = z[ii]*K[ii, kk]*np.exp(theta[kk])
#                
#                x_denominator += alpha[kk]*(K[ii, kk]*np.exp(theta[kk]) - 1.0)
#            x_mat[ii, self.nonhyd_phases] = (
#                x_numerator[ii, self.nonhyd_phases]/x_denominator)
    
        fug_mat = self.calc_fugacity(T, P, x_mat)
        for hyd_phase, ind in self.hyd_phases.items():
            x_mat[:, ind] = self.fug_list[ind].hyd_comp()
            
        
#        # Normalize x_mat so that each column adds to one
#        x_colsum = np.sum(x_mat, axis=0)
#        x_mat = x_mat / x_colsum[np.newaxis, :]

        return np.minimum(1, np.abs(x_mat))

    def Objective(self, z, alpha, theta, K):
        if type(z) != np.ndarray:
            z = np.asarray(z)
        if type(alpha) != np.ndarray:
            alpha = np.asarray(alpha)
        if type(theta) != np.ndarray:
            theta = np.asarray(theta)
        if type(K) != np.ndarray:
            K = np.asarray(K)
            
        # z, alpha, and theta are vectors.
        # z is length Nc, alpha and theta are length Np
        # K is matrix of size Nc x Np

        # Making use of np's broadcasting capabilities to implicitly
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
        
        if type(z) != np.ndarray:
            z = np.asarray(z)
        if type(alpha) != np.ndarray:
            alpha = np.asarray(alpha)
        if type(theta) != np.ndarray:
            theta = np.asarray(theta)
        if type(K) != np.ndarray:
            K = np.asarray(K)


        # Making use of np's broadcasting capabilities to implicitly
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
        Jacobion_Stability = np.concatenate((np.diag(Jac_alphaStability),
                                             np.diag(Jac_thetaStability)),
                                            axis=1)
        Jacobian = np.concatenate((Jacobian_Cost, Jacobion_Stability), axis=0)

        return Jacobian

    def Stability_func(self, alpha, theta):
        Y = alpha*theta/(alpha + theta)
        return Y

    def calc_K(self, T, P, x_mat):
        fug_mat = self.calc_fugacity(T, P, x_mat)
        K_mat = np.ones_like(x_mat)
        
        for ii, phase in enumerate(self.phases):
            if phase != self.ref_phase:
                K_mat[:, ii] = (fug_mat[:, self.ref_ind]/fug_mat[:, ii]
                                * x_mat[:, ii]/x_mat[:, self.ref_ind])
        return np.real(np.abs(K_mat))

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
        self.ref_fug = fug_out[:, self.ref_ind]
        self.ref_comp = x_mat[:, self.ref_ind]

        # Do this separetly because we need the reference phase fugacity.
        for hyd_phase, ind in self.hyd_phases.items():
            fug_out[:, ind] = self.fug_list[ind].calc(self.compobjs,
                                                      T,
                                                      P,
                                                      [],
                                                      self.ref_fug)

            
        return fug_out

    def find_alphatheta_min(self, z, alpha0, theta0, K, print_iter_info=False):

        # Use pre-defined Objective and Jacobian functions, but adjust for
        #  single input.
#        Objective = lambda x: self.Objective(z, x[0:self.Np], x[self.Np:], K)
#        Jacobian = lambda x: self.Jacobian(z, x[0:self.Np], x[self.Np:], K)

        # Set reference index if the controller hasn't already assigned it.
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
        # Mask reference phase rows and columns in Jacobian matrix
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
            res = self.Objective(z, alpha_old, theta_old, K)
            J = self.Jacobian(z, alpha_old, theta_old, K)
            J_mod = J[mat_mask].reshape([2*(self.Np - 1), 2*(self.Np - 1)])
            res_mod = res[arrdbl_mask]
#            try:
#                dx_tmp = -np.linalg.solve(J_mod, res_mod)
#            except:
#                dx_tmp = -np.matmul(np.linalg.pinv(J_mod), res_mod)
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

        return [alpha_new, theta_new]

    def ideal_LV(self, compobjs, T, P):
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

    def ideal_VAq(self, compobjs, T, P):
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

    def ideal_VHs1(self, compobjs, T, P):
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
                K[ii] = (self.ideal_VAq(comp, T, P)
                         / (0.88*self.ideal_IceAq(comp, T, P)))
        return np.abs(K)

    def ideal_VHs2(self, compobjs, T, P):
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
                K[ii] = (self.ideal_VAq(comp, T_Kelvin, P)
                         / (0.90*self.ideal_IceAq(comp, T_Kelvin, P)))
        return K

    def ideal_IceAq(self, compobjs, T, P):
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


    def make_ideal_K_allmat(self, compobjs, T, P):
        if not hasattr(compobjs, '__iter__'):
            compobjs = [compobjs]
        K_all_mat = np.zeros([len(compobjs), 5])
        K_all_mat[:, 0] = self.ideal_LV(compobjs, T, P)
        K_all_mat[:, 1] = self.ideal_VAq(compobjs, T, P)
        K_all_mat[:, 2] = self.ideal_VHs1(compobjs, T, P)
        K_all_mat[:, 3] = self.ideal_VHs2(compobjs, T, P)
        K_all_mat[:, 4] = self.ideal_IceAq(compobjs, T, P)
        return K_all_mat

    # Initialize the partition coefficient matrix based on P, T and components
    # Provide the option to specify the feed to predict the appropriate
    # reference phase or the option to speciy the reference phase explicitly.
    def make_ideal_K_mat(self, compobjs, T, P, **kwargs):
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

        K_all_mat = self.make_ideal_K_allmat(compobjs, T, P)
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
# Mapping from columns of K_all_mat to corresponding partition coefficient
# First phase is numerator, second phase is denominator
K_dict = {0: ('lhc', 'vapor'),
          1: ('vapor', 'aqueous'),
          2: ('vapor', 's1'),
          3: ('vapor', 's2'),
          4: ('ice', 'aqueous')}

# Tranformation from K_all_mat to partition coefficients of the form,
# K_{j, ref_phase} where keys in K_transform refer to reference phase
# and subsequent keys refer to phase j, and tuple describes algebraic
# manipulation of K_mat_all. 9 refers to the value 1 0-4 refers to a
# column
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

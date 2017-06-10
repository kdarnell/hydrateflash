#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 21:30:38 2017

@author: kdarnell
"""
import numpy as np
from scipy.integrate import quad

"""
    This is an equation of state (EOS) for water in the hydrate phase
    in the presence of multiple gases using the Ballard-modified
    van der Waals Platteeuw EOS.
"""
# Constants
R = 8.3144621  # universal gas constant
T_0 = 298.15
P_0 = 1.0
k = 1.3806488e-23


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
        self.compnames = [c.compname for c in compobjs]
        try:
            self.h2oind = [ii for ii, name in enumerate(self.compnames)
                           if name == 'h2o'][0]
        except IndexError:
            raise RuntimeError("""
                        Hydrate EOS requires water to be present!'\n
                        Please provide water in your component list.'""")

        self.Nc = len(compobjs)
        self.compobjs = compobjs
        self.gwbeta_RT_cons = np.zeros(1)
        self.volume_int = np.zeros(1)
        self.volume = np.zeros(1)
        self.activity = np.zeros(1)
        self.delta_mu_RT = np.zeros(1)
        self.gw0_RT = np.zeros(1)


        self.Hs = HydrateStructure(structure)
        self.alt_fug_vec = np.zeros(self.Nc)
        self.Y_small = np.zeros(self.Nc)
        self.Y_large = np.zeros(self.Nc)
        self.fug = 0

        # Transfer everything from the eos-specific dictionary to
        # the Hs object for use in the rest of the class
        # Note: 'eos_key' will be defined within sub-class
        for k, v in dict.items(self.Hs.eos[self.eos_key]):
            setattr(self.Hs, k, v)
        self.R_sm = np.zeros(len(self.Hs.R['sm']))
        self.z_sm = np.zeros_like(self.R_sm)
        for ii in range(len(self.z_sm)):
            self.z_sm[ii] = self.Hs.z['sm'][ii + 1]
        self.R_lg = np.zeros(len(self.Hs.R['lg']))
        self.z_lg = np.zeros_like(self.R_lg)
        for ii in range(len(self.z_lg)):
            self.z_lg[ii] = self.Hs.z['lg'][ii + 1]


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

    def calc_langmuir(self, C, eq_fug):
        Y = np.zeros(self.Nc)
        denominator = (1.0 + np.sum(C*eq_fug))
        for ii, comp in enumerate(self.compobjs):
            if comp.compname != 'h2o':
                Y[ii] = C[ii]*eq_fug[ii]/denominator
            else:
                Y[ii] = 0
        return Y

    def delta_mu_func(self, compobjs, T, P):
        delta_mu = (
            self.Hs.Nm['small']*np.log(1-np.sum(self.Y_small))
            + self.Hs.Nm['large']*np.log(1-np.sum(self.Y_large))
        )/self.Hs.Num_h2o
        return delta_mu

    def fugacity_calc(self, compobjs, T, P, x, alt_fug):
        pass

    def calc(self, compobjs, T, P, x, eq_fug):
        # x is a dummy variable in this EOS! Inserted here for parallism with
        # other EOS'

        # Raise flag if components change.
        if compobjs != self.compobjs:
            print("""Warning: Action not supported.\n
                     Components have changed.\n
                     Please create a new fugacity object.""")
            return None
        else:
            # Re-calculate constants if pressure or temperature changes.
            if self.T != T or self.P != P:
                self.make_constant_mats(compobjs, T, P)

        fug = self.fugacity_calc(compobjs, T, P, x, eq_fug)
        return fug


class HvdwpmEos(HydrateEos):
    cp = {'a0': 0.735409713*R,
          'a1': 1.4180551e-2*R,
          'a2': -1.72746e-5*R,
          'a3': 63.5104e-9*R}
    eos_key = 'hvdwpm'

    def __init__(self, compobjs, T, P, structure='s1'):
        # 'stdstate_fug' is the (partial?) fugacity at P_0, T_0

        # Inheret all prroperties from HydrateEos
        super().__init__(compobjs, T, P, structure)
        self.kappa = np.zeros(1)
        self.kappa0 = self.Hs.kappa
        self.a0_cubed = self.lattice_to_volume(self.Hs.a0_ast)
        self.kappa_vec = np.zeros(self.Nc)
        self.rep_sm_vec = np.zeros(self.Nc)
        self.rep_lg_vec = np.zeros(self.Nc)
        self.D_vec = np.zeros(self.Nc)
        self.a_new = self.Hs.a_norm
        self.Y_small_0 = np.zeros(self.Nc)
        self.Y_large_0 = np.zeros(self.Nc)
        self.eq_fug = np.zeros(self.Nc)
        self.stdstate_fug = np.zeros(self.Nc)

        # Retrieve information for components and populate within vectors
        for ii, comp in enumerate(compobjs):
            self.kappa_vec[ii] = comp.HvdWPM[self.Hs.hydstruc]['kappa']
            self.rep_sm_vec[ii] = comp.HvdWPM[self.Hs.hydstruc]['rep']['small']
            self.rep_lg_vec[ii] = comp.HvdWPM[self.Hs.hydstruc]['rep']['large']
            self.D_vec[ii] = comp.diam
            self.stdstate_fug[ii] = comp.stdst_fug

        # Set up parameters that do not change with the system, which
        # in the case means the fugacity of other phases.
        self.make_constant_mats(compobjs, T, P)

        # Set up parameters that depend on standard state (T_0, P_0)
        # Now, the lattice size won't change. However, the volume will
        # depend on P, T and kappa. And kappa depends on cage occupancy (Y)
        # and cage occupancy depends on langmuir constants, which in turn
        # depend on the volume. Thus, the langmuir constants will have to be
        # determined at each '.calc' call.
        self.find_stdstate_volume(compobjs, T, P)


    def make_constant_mats(self, compobjs, T, P):
        # Do standard stuff
        super().make_constant_mats(compobjs, T, P)

        self.gwbeta_RT = (
            self.Hs.gw_0beta/(R*T_0)
            - (12*T*self.Hs.hw_0beta - 12*T_0*self.Hs.hw_0beta
               + 12*T_0**2*self.cp['a0'] + 6*T_0**3*self.cp['a1']
               + 4*T_0**4*self.cp['a2'] + 3*T_0**5*self.cp['a3']
               - 12*T*T_0*self.cp['a0'] - 12*T*T_0**2*self.cp['a1']
               + 6*T**2*T_0*self.cp['a1'] - 6*T*T_0**3*self.cp['a2']
               + 2*T**3*T_0*self.cp['a2'] - 4*T*T_0**4*self.cp['a3']
               + T**4*T_0*self.cp['a3'] + 12*T*T_0*self.cp['a0']*np.log(T)
               - 12*T*T_0*self.cp['a0']*np.log(T_0))/(12*R*T*T_0)
            + (self.Hvol_Pint(T, P, self.a0_cubed, self.kappa0)
               - self.Hvol_Pint(T, P_0, self.a0_cubed, self.kappa0))*1e-1/(R*T)
        )

        # To be used later, there will be a similar *_Pfactor
        self.vol_Tfactor = self.Hydrate_size(T, P_0, 1.0, self.kappa0)
        self.lattice_Tfactor = self.Hydrate_size(T, P_0, 1.0,
                                                 self.kappa0, dim='linear')
        return self

    # Determine v0(x):
    def find_stdstate_volume(self, compobjs, T, P):
        error = 1e6
        TOL = 1e-3
        C_small = np.zeros(self.Nc)
        C_large = np.zeros(self.Nc)
        # 'Lattice sz' will originally be equal to self.Hs.a0_ast because we
        # set self.Y_*_0 equal to zero.
        # That will produce one set of C's and new Y's. We will then iterate
        # until convergence on 'C'.
        kappa = self.kappa0
        lattice_sz = self.Hs.a0_ast
        while error > TOL:
            out = self.iterate_function(compobjs, T_0, P_0,
                                        self.stdstate_fug,
                                        lattice_sz, kappa)
            C_small_new = out[0]
            C_large_new = out[1]
            self.Y_small_0 = out[2]  # Update these on every iteration
            self.Y_large_0 = out[3]
            lattice_sz = self.filled_lattice_size()
            kappa = self.kappa_func(self.Y_large_0)

            error = 0.0
            for ii, comp in enumerate(compobjs):
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
        return self

    # Determine v(x, T, P) and C, Y:
    def find_hydrate_properties(self, compobjs, T, P, eq_fug):
        error = 1e6
        TOL = 1e-3

        if 'C_small' in self.__dir__():
            C_small = self.C_small
            C_large = self.C_large
        else:
            C_small = np.zeros(self.Nc)
            C_large = np.zeros(self.Nc)

        # That will produce one set of C's and new Y's. We will then iterate
        # until convergence on 'C'. 'lattice_sz' is now fixed.
        kappa = self.kappa_tmp.copy()
        lattice_sz = self.a_0
        while error > TOL:
#            print('dictionary at find_props=', self.__dict__)
            out = self.iterate_function(compobjs, T, P, eq_fug,
                                        lattice_sz, kappa)
            C_small_new = out[0]
            C_large_new = out[1]
            self.Y_small = out[2]  # Update these on every iteration
            self.Y_large = out[3]
            kappa = self.kappa_func(self.Y_large)

            error = 0.0
            for ii, comp in enumerate(compobjs):
                if comp.compname != 'h2o':
                    error += (abs(C_small_new[ii]
                                  - C_small[ii])/C_small_new[ii]
                              + abs(C_large_new[ii]
                                    - C_large[ii])/C_large_new[ii])
            C_small = C_small_new
            C_large = C_large_new

        self.C_small = C_small
        self.C_large = C_large
        self.kappa_tmp = kappa.copy()
        self.v_H = self.Hydrate_size(T, P, self.v_H_0, kappa)
        return self

    def kappa_func(self, Y_large):
        # This is what I had in the Matlab prototype that seemed to agree
        # with CSMGem. However, it may not be accurate. The weighted sum
        # may also need to be evaluated.
        if self.Nc > 2:
            kappa = 3.0*np.sum(self.kappa_vec*Y_large)
        else:
            kappa = self.kappa0

        return kappa

    # Pass the volumetric versions of these!
    # Standard hydrate parameters are reported as volumeteric.
    # Componenet parameters for kappa are reported as linear.
    def Hydrate_size(self, T, P, v, kappa, dim='volumetric'):
        if dim == 'volumetric':
            factor = 1.0
        elif dim == 'linear':
            factor = 1.0/3.0

        H_size = (v*np.exp(factor*(self.Hs.alf[1]*(T - T_0)
                                   + self.Hs.alf[2]*(T - T_0)**2
                                   + self.Hs.alf[3]*(T - T_0)**3
                                   - kappa*(P - P_0))))
        return H_size

    # Pressure-integrated hydrate volume (for use in other equations)
    def Hvol_Pint(self, T, P, v, kappa):
        v = self.Hydrate_size(T, P, v, kappa)/(-kappa)
        return v

    # Conversion of 'linear' lattice parameter to a volume in cm^3/mol
    def lattice_to_volume(self, lattice_sz):
        if self.Hs.hydstruc != 'sH':
            v = 6.0221413e23/self.Hs.Num_h2o/1e24*(lattice_sz)**3
        else:
            v = self.Hs.v0
        return v

    # Distortion of cage from occupancy
    def cage_distortion(self):
        # Note that these are evaluated at self.Y_size0. This is because
        # this distortion function is evaluated at T_0, P_0
        small_const = (
            (1 + self.Hs.etam['small']/self.Hs.Num_h2o)*self.Y_small_0
            / (1 + (self.Hs.etam['small']/self.Hs.Num_h2o)*self.Y_small_0)
        )

        # Ballard did not differentiate between multiple or single guests
        # as we do here. In his version, the weighted exponential is always
        # used. However, we saw better agreement by separating single and
        # multiple guests.
        if self.Nc>2:
            self.repulsive_small = (small_const*np.exp(
                self.D_vec - np.sum(self.Y_small_0*self.D_vec)
            ))
        else:
            self.repulsive_small = small_const

        self.repulsive_large = (
            (1 + self.Hs.etam['large']/self.Hs.Num_h2o)*self.Y_large_0
            / (1 + (self.Hs.etam['large']/self.Hs.Num_h2o)*self.Y_large_0)
        )
        return self

    # Determine size of lattice due to the filling of cages at T_0, P_0
    def filled_lattice_size(self):
        self.cage_distortion()
        lattice_sz = (self.Hs.a0_ast
                      + (self.Hs.Nm['small']
                         * np.sum(self.repulsive_small*self.rep_sm_vec))
                      + (self.Hs.Nm['large']
                         *np.sum(self.repulsive_large*self.rep_lg_vec)))
        return lattice_sz

    def iterate_function(self, compobjs, T, P, fug, lattice_sz, kappa):
        C_small, C_large = self.langmuir_consts(compobjs, T, P,
                                                lattice_sz, kappa)
        Y_small = self.calc_langmuir(C_small, fug)
        Y_large = self.calc_langmuir(C_large, fug)
        return [C_small, C_large, Y_small, Y_large]


    def delta_func(self, N, Rn, aj, r):
        delta = ((1.0 - r/Rn - aj/Rn)**(-N) - (1.0 + r/Rn - aj/Rn)**(-N))/N
        return delta

    def w_func(self, zn, eps_k, r, Rn, sigma, aj):
        w = (2*zn*eps_k*(sigma**12/(Rn**11*r)
                         * (self.delta_func(10, Rn, aj, r)
                            + (aj/Rn)*self.delta_func(11, Rn, aj, r))
                         - sigma**6/(Rn**5*r)
                         * (self.delta_func(4, Rn, aj, r)
                            + (aj/Rn)*self.delta_func(5, Rn, aj, r))))
        return w
    
    def integrand(self, r, R, z, eps_k, sigma, aj, T):
        integrand_sum = 0
        for ii in range(len(R)):
            integrand_sum += self.w_func(z[ii], eps_k, r, R[ii], sigma, aj)
            
        output = r**2*np.exp((-1.0/T)*integrand_sum)
        return output

#    def integrand_sm(self, r, R1, R2, z1, z2, eps_k, sigma, aj, T):
#        output = r**2*np.exp((-1.0/T)
#                             * (self.w_func(z1, eps_k, r, R1, sigma, aj)
#                                + self.w_func(z2, eps_k, r, R2, sigma, aj)))
#        return output
#
#    def integrand_lg(self, r, R1, R2, R3, R4, z1, z2, z3, z4,
#                     eps_k, sigma, aj, T):
#        output = r**2*np.exp((-1.0/T)
#                             * (self.w_func(z1, eps_k, r, R1, sigma, aj)
#                                + self.w_func(z2, eps_k, r, R2, sigma, aj)
#                                + self.w_func(z3, eps_k, r, R3, sigma, aj)
#                                + self.w_func(z4, eps_k, r, R4, sigma, aj)))
#        return output

    def compute_integral_constants(self, T, P, lattice_sz, kappa):
        Pfactor = self.Hydrate_size(T_0, P, 1.0, kappa, dim='linear')
        a_factor = (lattice_sz/self.Hs.a_norm)*self.lattice_Tfactor*Pfactor
        for ii in range(len(self.Hs.R['sm'])):
            self.R_sm[ii] = self.Hs.R['sm'][ii + 1]*a_factor
        for ii in range(len(self.Hs.R['lg'])):
            self.R_lg[ii] = self.Hs.R['lg'][ii + 1]*a_factor
#        self.R1_sm = self.Hs.R['sm'][1]*a_factor
#        self.R2_sm = self.Hs.R['sm'][2]*a_factor
#        self.R1_lg = self.Hs.R['lg'][1]*a_factor
#        self.R2_lg = self.Hs.R['lg'][2]*a_factor
#        self.R3_lg = self.Hs.R['lg'][3]*a_factor
#        self.R4_lg = self.Hs.R['lg'][4]*a_factor
        return self

    def langmuir_consts(self, compobjs, T, P, lattice_sz, kappa):
        self.compute_integral_constants(T, P, lattice_sz, kappa)
        C_small = np.zeros(self.Nc)
        C_large = np.zeros(self.Nc)
        C_const = 1e-10**3*4*np.pi/(k*T)*1e5
        for ii, comp in enumerate(compobjs):
            if comp.compname != 'h2o':

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
        kappa_wtavg = self.kappa_func(self.Y_large)
        activity = (
            (v_H_0 - self.a0_cubed)/R*(self.Hs.a_fit/T_0
                                       + self.Hs.b_fit*(1/T - 1/T_0))
            + ((self.Hvol_Pint(T, P, v_H_0, kappa_wtavg)
                - self.Hvol_Pint(T, P, self.a0_cubed, self.kappa0))
              - (self.Hvol_Pint(T, P_0, v_H_0, kappa_wtavg)
                 - self.Hvol_Pint(T, P_0, self.a0_cubed, self.kappa0)))*1e-1/(R*T)
        )
        return activity


    def fugacity_calc(self, compobjs, T, P, x, eq_fug):

        if (eq_fug == self.eq_fug).all():
            fug = self.fug.copy()
        else:
            fug = np.zeros(self.Nc)
            fug[1:] = eq_fug[1:]

        # This will produce the correct C's and Y's, where Y is a
        # strong fucntion of 'eq_fug' and C is a weak function of 'eq_fug'
#        print(self.__dict__)
        self.find_hydrate_properties(compobjs, T, P, eq_fug)
#        print(self.__dict__)


        # This strongly depends on Y
        delta_mu_RT = self.delta_mu_func(compobjs, T, P)

        # This weakly depends on Y
        activity = self.activity_func(T, P, self.v_H_0)

        # Thus, this depends strongly on Y
        mu_H_RT = self.gwbeta_RT + activity + delta_mu_RT
        fug[0] = np.exp(mu_H_RT - self.gw0_RT)
        self.fug = fug.copy()

        return fug

    # Calcualte hydrate composition
    def hyd_comp(self):
        x_tmp = (self.Hs.Nm['small']*self.Y_small
                 + self.Hs.Nm['large']*self.Y_large)/self.Hs.Num_h2o
        x = x_tmp/(1.0 + np.sum(x_tmp))
        x[self.h2oind] = 1.0 - np.sum(x)

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
                                   'a_fit': 260,
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:40:41 2016

@author: kdarnell
"""
import numpy as np

"""
    This an equation of state (EOS) for vapor and liquid hydrocarbon phases
    using the Soave-Redlich-Koave (SRK). At present (30 Dec. 16), there is
    no correction for the liquid volume.
"""
R = 83.144621


class SRK_fugs(object):
    def __init__(self, compobjs, T, P):
        self.Nc = len(compobjs)
        self.S1_vec = np.zeros(self.Nc)
        self.kij_mat = np.zeros([self.Nc, self.Nc])
        self.a_vec = np.zeros(self.Nc)
        self.b_vec = np.zeros(self.Nc)
        self.a_mat = np.zeros([self.Nc, self.Nc])
        self.fug_mat = np.zeros(self.Nc)
        self.alf_vec = np.zeros(self.Nc)
        self.Tr_vec = np.zeros(self.Nc)
        self.Pr_vec = np.zeros(self.Nc)
        # Store these, but continue feeding them into subsequent functions
        # to make sure they don't change!!
        self.compobjs = compobjs
        self.T = T
        self.P = P

        # Assuming pressure and temperature won't change, compute terms that
        # are not functions of compostion.
        self.make_constant_mats(compobjs, T, P)

    def make_constant_mats(self, compobjs, T, P):
        for ii, comp in enumerate(compobjs):
            self.Tr_vec[ii] = comp.Tc/T
            self.Pr_vec[ii] = comp.Pc/P
            self.S1_vec[ii] = 0.48508 + 1.55171*comp.SRK['omega'] -\
                0.15613*comp.SRK['omega']**2
            self.alf_vec[ii] = (
                1.0 + self.S1_vec[ii]*(1.0 - np.sqrt(self.Tr_vec[ii])) +
                comp.SRK['S2']*(1 - np.sqrt(self.Tr_vec[ii])) /
                np.sqrt(self.Tr_vec[ii])
                                )**2
            self.a_vec[ii] = 0.42747*R**2*comp.Tc**2 / comp.Tc
            self.b_vec[ii] = 0.08644*R*comp.Tc / comp.Pc

        for ii, compouter in enumerate(compobjs):
            for jj, compinner in enumerate(compobjs):
                self.kij_mat[ii, jj] = compouter.SRK['kij'][compinner.compname]
                self.a_mat[ii, jj] = (
                        1 - self.kij_mat[ii, jj] *
                        np.sqrt(self.alf_vec[ii]*self.a_vec[ii] *
                                self.alf_vec[jj]*self.a_vec[jj])
                                      )

    def b_tot(self, x):
        b = 0.0
        for ii in range(self.Nc):
            b += x[ii]*self.b_vec[ii]
        return b

    def a_tot(self, x):
        a = 0.0
        for ii in range(self.Nc):
            for jj in range(self.Nc):
                a += x[ii]*x[jj]*self.a_mat[ii, jj]
        return a

    def calc(self, compobjs, T, P, x):
        if T != self.T or P != self.P or compobjs != self.compobjs:
            print('Warning: Action not supported. \n' +
                  'Key parameters (components, temperature, pressure)' +
                  ' have changed. \nPlease create a new fugacity object.')
            return None
        else:
            A = self.a_tot(x)*P / (R**2 * T**2)
            B = self.b_tot(x)*P / (R*T)
            Z = np.roots([1, -1, A - B - B**2, -(A*B)])
            return Z

#S1 = 0.48508 + 1.55171.*omega - 0.15613.*omega.^2;
#% S1(key.water) = 1.2440;%From pg. 378 of Ballard Thesis
#alf = (1 + S1.*(1 - sqrt(T./Tc)) + S2.*(1-sqrt(T./Tc))./sqrt(T./Tc)).^2;
#a = 0.42747.*R^2.*Tc.^2./Pc;
#a_mat = zeros(N,N);
#a_tot = 0;
#for i=1:N
#    for j=1:N
#        a_mat(i,j) = (1 - k(i,j))*sqrt(alf(i)*a(i)*alf(j)*a(j));
#        a_tot = a_tot + x(i)*x(j)*a_mat(i,j);
#    end
#end
#
#b = 0.08664*R*Tc./Pc;
#b_tot = sum(x.*b);
#
#A = a_tot*P/(R^2*T^2);
#B = b_tot*P/(R*T);
#if isnan(A) || isnan(B)
#    Z=NaN;
#else
#coeff(1) = 1;
#coeff(2) = -1;
#coeff(3) = A - B - B^2;
#coeff(4) = -(A*B);
#Z = cubic_roots(coeff);
#Z_v = max(Z);
#Z_l = min(Z);
#if isempty(Z)
#    Z_v=NaN;
#    Z_l=NaN;
#end
#end
#fug_l = x.*P.*exp((b./b_tot).*(Z_l - 1) - log(Z_l-B) - A./B.*(2.*(a_mat*x)./a_tot - b./b_tot).*log(1+B./Z_l));
#fug_v = x.*P.*exp((b./b_tot).*(Z_v - 1) - log(Z_v-B) - A./B.*(2.*(a_mat*x)./a_tot - b./b_tot).*log(1+B./Z_v));
#if max(fug_l(2:end)<fug_v(2:end))
#    fug=fug_l;
#else
#    fug=fug_v;
#end
#index = strcmp(comps,comprequest);
#out = fug(index);
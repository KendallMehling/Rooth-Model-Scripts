# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 14:06:13 2018

@author: kenda
"""

import numpy as np
import matplotlib.pyplot as plt

# We will now Fix K_d and Southern Precipitation and will vary the magnitude of the ratio of the Fresh Water Forcing.
# This model is for asymmetric temperture about the equator. We know change our polynomial expression and must add an additional
# Temperature offset to our stability matrices. 
K_dI = 0
K_dII = 1e-11
K_dIII = 1e-10
K_dIV = 4e-10

T1 = 4
T3 = 7

poly = []


s_prec = .9*10**-10

northern_prec_start = 0
northern_prec_end = 9e-9


northern_prec_values = np.linspace(northern_prec_start, northern_prec_end, 10000)

k = 1.5 * 10**-6
beta = 8.0 * 10**-4
alpha = 1.5e-4

# These polynomials not take into consideration the temperature gradient that was earlier ignored.
def polynomial_pos_coeff(kd,n_prec):
    poly = []
    for i in range(len(northern_prec_values)):
        coeff = [1, 2*kd + (k*alpha*(T1 - T3)), kd**2 - k*beta*s_prec + (2*k*kd*alpha*(T1 - T3)), alpha * (kd**2) *k *(T1 - T3) + (kd*k*beta*(n_prec[i] -s_prec))]
        poly.append(coeff)
    return poly


polyI_pos = polynomial_pos_coeff(K_dI, northern_prec_values)
polyII_pos = polynomial_pos_coeff(K_dII, northern_prec_values)
polyIII_pos = polynomial_pos_coeff(K_dIII, northern_prec_values)
polyIV_pos = polynomial_pos_coeff(K_dIV, northern_prec_values)

def polynomial_neg_coeff(kd,n_prec):
    poly = []
    for i in range(len(northern_prec_values)):
        coeff = [1, (k*alpha*(T1-T3)) - 2*kd, kd**2 - k*beta*n_prec[i] - (2*k*kd*alpha*(T1 - T3)), alpha*(kd**2)*k*(T1 - T3) + (kd*k*beta*(n_prec[i] - s_prec))]
        poly.append(coeff)
    return poly

polyI_neg = polynomial_neg_coeff(K_dI, northern_prec_values)
polyII_neg = polynomial_neg_coeff(K_dII, northern_prec_values)
polyIII_neg = polynomial_neg_coeff(K_dIII, northern_prec_values)
polyIV_neg = polynomial_neg_coeff(K_dIV, northern_prec_values)

qI_pos = []
qI_neg = []
qII_pos = []
qII_neg = []
qIII_pos = []
qIII_neg = []
qIV_pos = []
qIV_neg = []

def qsolve(q,poly,nonneg):
    for i in range(len(northern_prec_values)):
        sol = np.roots(poly[i])
        if nonneg:
            sol = [max(s, 0) for s in sol]
        else:
            sol = [min(s, 0) for s in sol]
        q.append(sol)
    return q

qI_pos = (qsolve(qI_pos, polyI_pos, True)) 
qI_neg = (qsolve(qI_neg,polyI_neg, False))
qII_pos = (qsolve(qII_pos, polyII_pos, True))
qII_neg = (qsolve(qII_neg, polyII_neg, False))
qIII_pos = (qsolve(qIII_pos, polyIII_pos, True))
qIII_neg = (qsolve(qIII_neg, polyIII_neg, False))
qIV_pos = (qsolve(qIV_pos, polyIV_pos, True))
qIV_neg = (qsolve(qIV_neg, polyIV_neg, False))

   
   
qI_pos = np.asarray(qI_pos)
qI_neg = np.asarray(qI_neg)
qII_pos = np.asarray(qII_pos)
qII_neg = np.asarray(qII_neg)
qIII_pos = np.asarray(qIII_pos)
qIII_neg = np.asarray(qIII_neg)
qIV_pos = np.asarray(qIV_pos)
qIV_neg = np.asarray(qIV_neg)    
    
    
def del_imaginary(q_list):
    for q in q_list:
        q.sort()
    nrow = q_list.shape[0]
    ncol = q_list.shape[1]
    for i in range(nrow):
        for j in range(ncol):
            if np.imag(q_list[i, j]) != 0:
                q_list[i, j] = np.nan
            elif q_list[i,j] == 0:
                q_list[i,j] = np.nan

del_imaginary(qI_pos)
del_imaginary(qI_neg)
del_imaginary(qII_pos)
del_imaginary(qII_neg)
del_imaginary(qIII_pos)
del_imaginary(qIII_neg)
del_imaginary(qIV_pos)
del_imaginary(qIV_neg)

temp_offset = (T3 - T1)*k*alpha

def pos_eigen(kd, q, n_prec):
    eigenvalues = []
    index = []
    for i in range(len(n_prec)):
        if np.isnan(q[i]):
            pass
        else:
            A11 = -(3*(q[i]+temp_offset) + 3*kd)
            A12 = (((q[i]+temp_offset) + kd)/(k*beta)) + (n_prec[i]/((q[i]+temp_offset) + kd))
            A21 = -(3*(q[i]+temp_offset)*k*beta)
            A22 = ((k*beta)*(((q[i]+temp_offset) + kd)*(n_prec[i] - s_prec) + (kd*n_prec[i]))/(((q[i]+temp_offset) + kd)**2)) - kd
            value = np.linalg.eigvals([[A11,A12],[A21,A22]])
            eigenvalues.append(value)
            index.append(i)
            
    return np.asarray(eigenvalues), np.asarray(index)

north_prec = northern_prec_values

values_qI_pos = qI_pos[:,2]
values_qI_neg = qI_neg[:,0]
values_qII_pos1 = qII_pos[:,1]
values_qII_pos2 = qII_pos[:,2]
values_qII_neg1 = qII_neg[:,0]
values_qII_neg2 = qII_neg[:,1]
values_qIII_pos1 = qIII_pos[:,1]
values_qIII_pos2 = qIII_pos[:,2]
values_qIII_neg1 = qIII_neg[:,0]
values_qIII_neg2 = qIII_neg[:,1]
values_qIV_pos = qIV_pos[:,2]
values_qIV_neg = qIV_neg[:,0]


pos_eigenvalue_qI_pos, pos_index_qI_pos = pos_eigen(0,values_qI_pos, north_prec)
pos_eigenvalue_qII_pos1, pos_index_qII_pos1 = pos_eigen(1e-11, values_qII_pos1,  north_prec)
pos_eigenvalue_qII_pos2, pos_index_qII_pos2 = pos_eigen(1e-11, values_qII_pos2, north_prec)
pos_eigenvalue_qIII_pos1, pos_index_qIII_pos1 = pos_eigen(1e-10, values_qIII_pos1, north_prec)
pos_eigenvalue_qIII_pos2, pos_index_qIII_pos2 = pos_eigen(1e-10, values_qIII_pos2, north_prec)
pos_eigenvalue_qIV_pos, pos_index_qIV_pos = pos_eigen(4e-10, values_qIV_pos, north_prec)


def pos_stability(eig_list, plot_index, q_list, line_color):
    n_prec_stable = []
    stable_q = []
    n_prec_unstable = []
    unstable_q = []
    nrow = eig_list.shape[0]
    
    for i in range(nrow):
       if eig_list[i,0] and eig_list[i,1] <= 0:
           stable_q.append(q_list[plot_index[i]])
           n_prec_stable.append(plot_index[i])
       else:
           unstable_q.append(q_list[plot_index[i]])
           n_prec_unstable.append(plot_index[i])
    plt.plot(north_prec[n_prec_stable], stable_q, color = line_color)
    plt.plot(north_prec[n_prec_unstable], unstable_q, "--", color = line_color)
            
pos_stability(pos_eigenvalue_qI_pos, pos_index_qI_pos, values_qI_pos, "black")
pos_stability(pos_eigenvalue_qII_pos1, pos_index_qII_pos1, values_qII_pos1, "red")
pos_stability(pos_eigenvalue_qII_pos2, pos_index_qII_pos2, values_qII_pos2, "red")
pos_stability(pos_eigenvalue_qIII_pos1, pos_index_qIII_pos1, values_qIII_pos1, "blue")
pos_stability(pos_eigenvalue_qIII_pos2, pos_index_qIII_pos2, values_qIII_pos2, "blue")
pos_stability(pos_eigenvalue_qIV_pos, pos_index_qIV_pos, values_qIV_pos, "green")

def neg_eigen(kd, q, n_prec):
    eigenvalues = []
    index = []
    for i in range(len(n_prec)):
        if np.isnan(q[i]):
            pass
        else:
            A11 = 3*((q[i]+temp_offset) - kd)
            A12 =  (-s_prec /(kd - (q[i]+temp_offset))) + (((q[i]+temp_offset) - kd)/(k*beta))
            A21 = -3*(q[i]+temp_offset)*k*beta
            A22 = -kd - k*beta*(((kd - (q[i]+temp_offset))*(n_prec[i] - s_prec) - s_prec*kd)/((kd - (q[i]+temp_offset))**2))
            value = np.linalg.eigvals([[A11,A12],[A21, A22]])
            eigenvalues.append(value)
            index.append(i)

    return np.asarray(eigenvalues), np.asarray(index)


north_prec = northern_prec_values

values_qI_neg = qI_neg[:,0]
values_qII_neg1 = qII_neg[:,0]
values_qII_neg2 = qII_neg[:,1]
values_qIII_neg1 = qIII_neg[:,0]
values_qIII_neg2 = qIII_neg[:,1]
values_qIV_neg = qIV_neg[:,0]

neg_eigenvalue_qI_neg, neg_index_qI_neg = neg_eigen(0, values_qI_neg, north_prec)
neg_eigenvalue_qII_neg1, neg_index_qII_neg1 = neg_eigen(1e-11, values_qII_neg1, north_prec)
neg_eigenvalue_qII_neg2, neg_index_qII_neg2 = neg_eigen(1e-11, values_qII_neg2, north_prec)
neg_eigenvalue_qIII_neg1, neg_index_qIII_neg1 = neg_eigen(1e-10, values_qIII_neg1, north_prec)
neg_eigenvalue_qIII_neg2, neg_index_qIII_neg2 = neg_eigen(1e-10, values_qIII_neg2, north_prec)
neg_eigenvalue_qIV_neg, neg_index_qIV_neg = neg_eigen(4e-10, values_qIV_neg, north_prec)



def neg_stability(eig_list, plot_index, q_list, line_color):
    n_prec_stable = []
    stable_q = []
    n_prec_unstable = []
    unstable_q = []
    nrow = eig_list.shape[0]
    
    for i in range(nrow):
            if eig_list[i,0] and eig_list[i,1] <= 0:
                stable_q.append(q_list[plot_index[i]])
                n_prec_stable.append(plot_index[i])
            else:
               unstable_q.append(q_list[plot_index[i]])
               n_prec_unstable.append(plot_index[i])
    plt.plot(north_prec[n_prec_stable], stable_q, color = line_color)
    plt.plot(north_prec[n_prec_unstable], unstable_q, "--", color = line_color)
                

neg_stability(neg_eigenvalue_qI_neg, neg_index_qI_neg, values_qI_neg, "black")
neg_stability(neg_eigenvalue_qII_neg1, neg_index_qII_neg1, values_qII_neg1, "red")
neg_stability(neg_eigenvalue_qII_neg2, neg_index_qII_neg2, values_qII_neg2, "red")
neg_stability(neg_eigenvalue_qIII_neg1, neg_index_qIII_neg1, values_qIII_neg1, "blue")
neg_stability(neg_eigenvalue_qIII_neg2, neg_index_qIII_neg2, values_qIII_neg2, "blue")
neg_stability(neg_eigenvalue_qIV_neg, neg_index_qIV_neg, values_qIV_neg, "green")



plt.xlim(0, northern_prec_end)
plt.ylim(-20e-10, 10e-10)
plt.locator_params(axis = "y", nbins = 6)
locs, labels = plt.yticks()
plt.yticks(locs, [-20,-15,-10,-5,0,5,10] )
locs, labels = plt.xticks()
plt.xticks(locs, [0,1,2,3,4,5,6,7,8,9])


plt.xlabel("$\Phi_N (10^{-9} \  psu \  s^{-1})$")
plt.ylabel("$q(10^{-10} \ s^{-1})$")
plt.grid(True)

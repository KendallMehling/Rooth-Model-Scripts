# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 16:55:34 2018

@author: kenda
"""

import numpy as np
import matplotlib.pyplot as plt

# We will now Fix K_d and will vary the magnitude of the ratio of the Fresh Water Forcing. We also have the 
# asymmetric temperature gradient to see how this effects the system.

K_d = 1e-10
T1 = 7
T3 = 7

steps = 10000

q_start = -4 * 10**-10
q_end = 4 * 10**-10

### This is the Freshwater Forcing that will be amplified to see the effect upon 
### Circulation of the Ocean. 

southern_prec_start = .9 * 10 ** -11
southern_prec_end = .9 * 10 ** -9

northern_prec_start = .9 * 10 ** -11
northern_prec_end = .9 * 10 ** -9

southern_prec_values = np.linspace(southern_prec_start, southern_prec_end, steps)
northern_prec_values = np.linspace(northern_prec_start, northern_prec_end, steps)

k = 1.5e-6
beta = 8e-4
alpha = 1.5e-4

### Will use the following Functions to produce the Coefficients
### of the Cubic equation for both positive and negative flow to 
### Then find the roots of the Cubic Expression. Will plot similarily
### to the normal T.I Rooth Simulation Created before.

def polynomial_pos_coeff(kd,s_prec,n_prec):
    poly = []
    for i in range(steps):
        coeff = [1, 2*kd + (k*alpha * (T1 - T3)), kd**2 - k*beta*s_prec[i] + (2*k*kd*alpha*(T1 - T3)), alpha*(kd**2)*k*(T1 -T3) + (kd*k*beta*(n_prec[i] - s_prec[i]))]
        poly.append(coeff)
    return poly

### This is for the Case outlined in Longworth Et al.

polyI_pos = polynomial_pos_coeff(K_d, southern_prec_values, northern_prec_values)
polyII_pos = polynomial_pos_coeff(K_d, southern_prec_values, .5 * northern_prec_values)
polyIII_pos = polynomial_pos_coeff(K_d, southern_prec_values, 1.5 * northern_prec_values)

def polynomial_neg_coeff(kd,s_prec,n_prec):
    poly = []
    for i in range(steps):
        coeff = [1, (k*alpha * (T1 - T3)) - 2*kd, kd**2 - k*beta*n_prec[i] - (2*k*kd*alpha*(T1 - T3)), alpha*(kd**2)*k*(T1 -T3) + (kd*k*beta*(n_prec[i] - s_prec[i]))]
        poly.append(coeff)
    return poly

polyI_neg = polynomial_neg_coeff(K_d, southern_prec_values, northern_prec_values)
polyII_neg = polynomial_neg_coeff(K_d, southern_prec_values, .5 * northern_prec_values)
polyIII_neg = polynomial_neg_coeff(K_d, southern_prec_values, 1.5 * northern_prec_values)

qI_pos = []
qI_neg = []

qII_pos = []
qII_neg = []

qIII_pos = []
qIII_neg = []

def qsolve(q,poly,nonneg):
    for i in range(steps):
        sol =  np.roots(poly[i]) 
        if nonneg:
            sol = [max(s, 0) for s in sol]
        else:
            sol = [min(s, 0) for s in sol]
        q.append(sol)
    return q

qI_pos = np.asarray(qsolve(qI_pos, polyI_pos, True))
qI_neg = np.asarray(qsolve(qI_neg,polyI_neg, False))

qII_pos = np.asarray(qsolve(qII_pos, polyII_pos, True))
qII_neg = np.asarray(qsolve(qII_neg, polyII_neg, False))

qIII_pos = np.asarray(qsolve(qIII_pos, polyIII_pos, True))
qIII_neg = np.asarray(qsolve(qIII_neg, polyIII_neg, False))



def del_imaginary(q_list):
    for q in q_list:
        q.sort()
    nrow = q_list.shape[0]
    ncol = q_list.shape[1]
    for i in range(nrow):
        for j in range(ncol):
            if np.imag(q_list[i, j]) != 0:
                q_list[i, j] = np.NaN
            elif q_list[i,j] == 0:
                q_list[i,j] = np.NaN
                
del_imaginary(qI_pos)
del_imaginary(qI_neg)
del_imaginary(qII_pos)
del_imaginary(qII_neg)
del_imaginary(qIII_pos)
del_imaginary(qIII_neg)

temp_offset =0  #(T3 - T1)*k*alpha

def pos_eigen(kd, q, s_prec, n_prec):
    eigenvalues = []
    index = []
    for i in range(len(n_prec)):
        if np.isnan(q[i]):
            pass
        else:
            A11 = -(3*(q[i]+temp_offset) + 3*kd)
            A12 = (((q[i]+temp_offset) + kd)/(k*beta)) + (n_prec[i]/((q[i]+temp_offset) + kd))
            A21 = -(3*(q[i]+temp_offset)*k*beta)
            A22 = ((k*beta)*(((q[i]+temp_offset) + kd)*(n_prec[i] - s_prec[i]) + (kd*n_prec[i]))/(((q[i]+temp_offset) + kd)**2)) - kd
            value = np.linalg.eigvals([[A11,A12],[A21,A22]])
            eigenvalues.append(value)
            index.append(i)
            
    return np.asarray(eigenvalues), np.asarray(index)

north_prec = northern_prec_values
south_prec = southern_prec_values

values_qI_pos = qI_pos[:,2]
values_qII_pos2 = qII_pos[:,2]
values_qIII_pos1 = qIII_pos[:,1]
values_qIII_pos2 = qIII_pos[:,2]



pos_eigenvalue_qI_pos, pos_index_qI_pos = pos_eigen(K_d,values_qI_pos, south_prec, north_prec)
pos_eigenvalue_qII_pos2, pos_index_qII_pos2 = pos_eigen(K_d, values_qII_pos2, south_prec, .5*north_prec)
pos_eigenvalue_qIII_pos1, pos_index_qIII_pos1 = pos_eigen(K_d, values_qIII_pos1, south_prec, 1.5*north_prec)
pos_eigenvalue_qIII_pos2, pos_index_qIII_pos2 = pos_eigen(K_d, values_qIII_pos2, south_prec, 1.5*north_prec)



def pos_stability(eig_list, plot_index, q_list, line_color):
    n_prec_stable = []
    stable_q = []
    n_prec_unstable = []
    unstable_q = []
    nrow = eig_list.shape[0]
    for i in range(nrow):
       if eig_list[i,0] and eig_list[i,1] <= 0:
           stable_q.append(10e9 * q_list[plot_index[i]])
           n_prec_stable.append(plot_index[i])
       else:
           unstable_q.append(10e9 * q_list[plot_index[i]])
           n_prec_unstable.append(plot_index[i])
    plt.plot(north_prec[n_prec_stable], stable_q, color = line_color)
    plt.plot(north_prec[n_prec_unstable], unstable_q, "--", color = line_color)
            
pos_stability(pos_eigenvalue_qI_pos, pos_index_qI_pos, values_qI_pos, "black")
pos_stability(pos_eigenvalue_qII_pos2, pos_index_qII_pos2, values_qII_pos2, "red")
pos_stability(pos_eigenvalue_qIII_pos1, pos_index_qIII_pos1, values_qIII_pos1, "blue")
pos_stability(pos_eigenvalue_qIII_pos2, pos_index_qIII_pos2, values_qIII_pos2, "blue")


def neg_eigen(kd, q, s_prec, n_prec):
    eigenvalues = []
    index = []
    for i in range(len(n_prec)):
        if np.isnan(q[i]):
            pass
        else:
            A11 = 3*((q[i]+temp_offset) - kd)
            A12 =  (-s_prec[i] /(kd - (q[i]+temp_offset))) + (((q[i]+temp_offset) - kd)/(k*beta))
            A21 = -3*(q[i]+temp_offset)*k*beta
            A22 = -kd - k*beta*(((kd - (q[i]+temp_offset))*(n_prec[i] - s_prec[i]) - s_prec[i]*kd)/((kd - (q[i]+temp_offset))**2))
            value = np.linalg.eigvals([[A11,A12],[A21, A22]])
            eigenvalues.append(value)
            index.append(i)

    return np.asarray(eigenvalues), np.asarray(index)


north_prec = northern_prec_values
south_prec = southern_prec_values

values_qI_neg = qI_neg[:,0]
values_qII_neg1 = qII_neg[:,0]
values_qII_neg2 = qII_neg[:,1]
values_qIII_neg1 = qIII_neg[:,0]

neg_eigenvalue_qI_neg, neg_index_qI_neg = neg_eigen(K_d, values_qI_neg, south_prec, north_prec)
neg_eigenvalue_qII_neg1, neg_index_qII_neg1 = neg_eigen(K_d, values_qII_neg1, south_prec, .5*north_prec)
neg_eigenvalue_qII_neg2, neg_index_qII_neg2 = neg_eigen(K_d, values_qII_neg2, south_prec, .5*north_prec)
neg_eigenvalue_qIII_neg1, neg_index_qIII_neg1 = neg_eigen(K_d, values_qIII_neg1, south_prec, 1.5*north_prec)



def neg_stability(eig_list, plot_index, q_list, line_color):
    n_prec_stable = []
    stable_q = []
    n_prec_unstable = []
    unstable_q = []
    nrow = eig_list.shape[0]
    for i in range(nrow):
        if eig_list[i,0] and eig_list[i,1] <= 0:
            stable_q.append(10e9 * q_list[plot_index[i]])
            n_prec_stable.append(plot_index[i])
        else:
           unstable_q.append(10e9 * q_list[plot_index[i]])
           n_prec_unstable.append(plot_index[i])
    plt.plot(north_prec[n_prec_stable], stable_q, color = line_color)
    plt.plot(north_prec[n_prec_unstable], unstable_q, "--", color = line_color)
                

neg_stability(neg_eigenvalue_qI_neg, neg_index_qI_neg, values_qI_neg, "black")
neg_stability(neg_eigenvalue_qII_neg1, neg_index_qII_neg1, values_qII_neg1, "red")
neg_stability(neg_eigenvalue_qII_neg2, neg_index_qII_neg2, values_qII_neg2, "red")
neg_stability(neg_eigenvalue_qIII_neg1, neg_index_qIII_neg1, values_qIII_neg1, "blue")



plt.plot(3*10**-10, 3*10**-10, "k", label = "$ \Phi_N = \Phi_S $")
plt.plot(3*10**-10, 3*10**-10, "r", label = "$ \Phi_N = .5\Phi_S $")
plt.plot(3*10**-10, 3*10**-10, "b", label = "$ \Phi_N = 1.5\Phi_S $")


plt.xlabel("Southern Precipitation $\Phi_S \  (10^{-10} \ psu/s)$")
plt.ylabel("Q-Flow $ \ 10^{-10} \ s^{-1}$")


plt.xlim(0,1e-9)
plt.ylim(-15,15)

locs, labels = plt.xticks()
plt.xticks(locs,[0,2,4,6,8,10])

plt.legend()
plt.legend(frameon = False)

plt.grid(True)
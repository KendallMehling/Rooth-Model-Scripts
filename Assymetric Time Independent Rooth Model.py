# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 09:05:46 2018

@author: kenda
"""
#This model is similar to the Time independent modle except this does not assume symmetric temperature about the equator. Therfore the coefficients to 
# the polynomials are changed and the stability matrices must have an additional temperature offset. 

import numpy as np
import matplotlib.pyplot as plt

steps = 10000

k_start = 0
k_end = 4 * 10**-10
k_values = np.linspace(k_start, k_end, steps)

T1 = 0
T3 = 4

q_start = -4 * 10**-10
q_end = 4 * 10**-10

k = 1.5 * 10**-6
beta = 8.0 * 10**-4
alpha = 1.5e-4

def polynomial_pos(kd,s_prec,n_prec):
    poly = []
    for i in range(steps):
        coeff = [1, (k * alpha * (T1 - T3)) + 2*kd[i], kd[i]**2 - k*beta*s_prec + (2*kd[i]*k*alpha*(T1 - T3)), alpha*(kd[i]**2)*k*(T1 - T3) + (kd[i]*k*beta*(n_prec - s_prec))]
        poly.append(coeff)
    return poly

polyI_pos = polynomial_pos(k_values,.9*10**-10,.9*10**-10)
polyII_pos = polynomial_pos(k_values, .9*10**-10, .45*10**-10)
polyIII_pos = polynomial_pos(k_values, .9*10**-10, 1.35*10**-10)

def polynomial_neg(kd,s_prec,n_prec):
    poly = []
    for i in range(steps):
        coeff = [1, (k * alpha * (T1 - T3)) -2*kd[i], kd[i]**2 - k*beta*n_prec - (2*kd[i]*k*alpha*(T1 - T3)), alpha*(kd[i]**2)*k*(T1 - T3) + (kd[i]*k*beta*(n_prec - s_prec))]
        poly.append(coeff)
    return poly

polyI_neg = polynomial_neg(k_values,.9*10**-10,.9*10**-10)
polyII_neg = polynomial_neg(k_values, .9*10**-10, .45*10**-10)
polyIII_neg = polynomial_neg(k_values, .9*10**-10, 1.35*10**-10)

qI_pos = []
qI_neg = []

qII_pos = []
qII_neg = []

qIII_pos = []
qIII_neg = []

def qsolve(q,poly,nonneg):
    for i in range(steps):
        sol = np.roots(poly[i])
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
        q = q 
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


temp_offset = (T3 - T1)*k*alpha

def pos_eigen(kd, q,s_prec, n_prec):
    eigenvalues = []
    index = []
    for i in range(len(k_values)):
        if np.isnan(q[i]):
            pass
        else:
            A11 = -(3*(q[i] + temp_offset) + 3*kd[i])
            A12 = (((q[i] + temp_offset) + kd[i])/(k*beta)) + (n_prec/((q[i]+ temp_offset) + kd[i]))
            A21 = -(3*(q[i]+ temp_offset)*k*beta)
            A22 = ((k*beta)*(((q[i] + temp_offset) + kd[i])*(n_prec - s_prec) + (kd[i]*n_prec))/(((q[i] + temp_offset) + kd[i])**2)) - kd[i]
            value = np.linalg.eigvals([[A11,A12],[A21,A22]])
            eigenvalues.append(value)
            index.append(i)
            
    return np.asarray(eigenvalues), np.asarray(index)

south_prec = .9e-10
north_precI = south_prec
north_precII = .5* south_prec
north_precIII = 1.5* south_prec

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


pos_eigenvalue_qI_pos, pos_index_qI_pos = pos_eigen(k_values,values_qI_pos, south_prec, north_precI)
pos_eigenvalue_qII_pos1, pos_index_qII_pos1 = pos_eigen(k_values, values_qII_pos1, south_prec, north_precII)
pos_eigenvalue_qII_pos2, pos_index_qII_pos2 = pos_eigen(k_values, values_qII_pos2, south_prec, north_precII)
pos_eigenvalue_qIII_pos1, pos_index_qIII_pos1 = pos_eigen(k_values, values_qIII_pos1, south_prec, north_precIII)
pos_eigenvalue_qIII_pos2, pos_index_qIII_pos2 = pos_eigen(k_values, values_qIII_pos2, south_prec, north_precIII)

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
    plt.plot(k_values[n_prec_stable], stable_q, color = line_color)
    plt.plot(k_values[n_prec_unstable], unstable_q, "--", color = line_color)
            
pos_stability(pos_eigenvalue_qI_pos, pos_index_qI_pos, values_qI_pos, "black")
pos_stability(pos_eigenvalue_qII_pos1, pos_index_qII_pos1, values_qII_pos1, "red")
pos_stability(pos_eigenvalue_qII_pos2, pos_index_qII_pos2, values_qII_pos2, "red")
pos_stability(pos_eigenvalue_qIII_pos1, pos_index_qIII_pos1, values_qIII_pos1, "blue")
pos_stability(pos_eigenvalue_qIII_pos2, pos_index_qIII_pos2, values_qIII_pos2, "blue")

def neg_eigen(kd, q,s_prec, n_prec):
    eigenvalues = []
    index = []
    for i in range(steps):
        if np.isnan(q[i]):
            pass
        else:
            A11 = 3*((q[i]+ temp_offset) - kd[i])
            A12 =  (-s_prec /(kd[i] - (q[i] + temp_offset)) + ((q[i] + temp_offset) - kd[i])/(k*beta))
            A21 = -3*(q[i]+temp_offset)*k*beta
            A22 = -kd[i]  -k*beta*(((kd[i] - (q[i]+temp_offset))*(n_prec - s_prec) - s_prec*kd[i])/((kd[i] - (q[i]+ temp_offset))**2))
            value = np.linalg.eigvals([[A11,A12],[A21, A22]])
            eigenvalues.append(value)
            index.append(i)

    return np.asarray(eigenvalues), np.asarray(index)


values_qI_neg = qI_neg[:,0]
values_qII_neg1 = qII_neg[:,0]
values_qII_neg2 = qII_neg[:,1]
values_qIII_neg1 = qIII_neg[:,0]
values_qIII_neg2 = qIII_neg[:,1]

neg_eigenvalue_qI_neg, neg_index_qI_neg = neg_eigen(k_values, values_qI_neg,south_prec, north_precI)
neg_eigenvalue_qII_neg1, neg_index_qII_neg1 = neg_eigen(k_values, values_qII_neg1,south_prec, north_precII)
neg_eigenvalue_qII_neg2, neg_index_qII_neg2 = neg_eigen(k_values, values_qII_neg2, south_prec, north_precII)
neg_eigenvalue_qIII_neg1, neg_index_qIII_neg1 = neg_eigen(k_values, values_qIII_neg1, south_prec, north_precIII)
neg_eigenvalue_qIII_neg2, neg_index_qIII_neg2 = neg_eigen(k_values, values_qIII_neg2, south_prec, north_precIII)



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
    plt.plot(k_values[n_prec_stable], stable_q, color = line_color)
    plt.plot(k_values[n_prec_unstable], unstable_q, "--", color = line_color)
                

neg_stability(neg_eigenvalue_qI_neg, neg_index_qI_neg, values_qI_neg, "black")
neg_stability(neg_eigenvalue_qII_neg1, neg_index_qII_neg1, values_qII_neg1, "red")
neg_stability(neg_eigenvalue_qII_neg2, neg_index_qII_neg2, values_qII_neg2, "red")
neg_stability(neg_eigenvalue_qIII_neg1, neg_index_qIII_neg1, values_qIII_neg1, "blue")
neg_stability(neg_eigenvalue_qIII_neg2, neg_index_qIII_neg2, values_qIII_neg2, "blue")


plt.plot(0,0, 'k', label = "$ \Phi_N = \Phi_S$" )
plt.plot(0,0, 'r', label = "$ \Phi_N = .5\Phi_S$")
plt.plot(0,0, 'b', label = "$ \Phi_N = 1.5\Phi_S$")
plt.plot(0,0, 'none', label = "T1 = " + str(T1) + "C" )
plt.plot(0,0, "none", label = "T3 = " + str(T3) + "C" )

plt.legend()
plt.legend(frameon = False)

plt.xlim(k_start,k_end)
plt.ylim(-.4e-9, 1.2e-9)

#locs, labels = plt.xticks()
#plt.xticks(locs, [0,.5,1,1.5,2,2.5,3,3.5,4] )
locs, labels = plt.yticks()
plt.yticks(locs, [-4,-2,0,2,4,6,8,10,12])


plt.xlabel("$K_{d} (10^{-10} \  s^{-1}$)")
plt.ylabel("q $(10^{-10} \ s^{-1})$")
plt.grid(True)

plt.show()
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 12:00:22 2018

@author: kenda
"""


import numpy as np
import matplotlib.pyplot as plt

# The purpose of this Script is to reproduce Fig 6. From Longworth (2004) for steady state solutions
# for the Rooth model. We will vary the Diffusion parameter and see how it affects the flow strength.


#Below we establish the parameter region that we will calculate circulation from.
steps = 5000
k_start = 0
k_end = 4 * 10**-10
k_values = np.linspace(k_start, k_end, steps)

poly = []


q_start = -4 * 10**-10
q_end = 4 * 10**-10

k = 1.5 * 10**-6
beta = 8.0 * 10**-4
south_prec = .9e-10
# We Will use the analytical expression Given in Longworth to solve for q. We make a list of all the 
# Coefficients and will then find the Real Roots of these coefficients to find obtainable solutions.


def polynomial_pos(kd,s_prec,n_prec):
    poly = []
    for i in range(steps):
        coeff = [1, 2*kd[i], kd[i]**2 - k*beta*s_prec,(kd[i]*k*beta*(n_prec - s_prec))]
        poly.append(coeff)
    return poly

### We do this for the three cases of Freshwater Forcing Concerned. 

polyI_pos = polynomial_pos(k_values,south_prec, south_prec)
polyII_pos = polynomial_pos(k_values, south_prec, 0.5* south_prec)
polyIII_pos = polynomial_pos(k_values, south_prec, 1.5* south_prec)

# We do the same process for negative Flow using its analytical expression.

def polynomial_neg(kd,s_prec,n_prec):
    poly = []
    for i in range(steps):
        coeff = [1, -2*kd[i], kd[i]**2 - k*beta*n_prec,(kd[i]*k*beta*(n_prec - s_prec))]
        poly.append(coeff)
    return poly

polyI_neg = polynomial_neg(k_values, south_prec, south_prec)
polyII_neg = polynomial_neg(k_values, south_prec, .5*south_prec)
polyIII_neg = polynomial_neg(k_values, south_prec, 1.5*south_prec)

qI_pos = []
qI_neg = []

qII_pos = []
qII_neg = []

qIII_pos = []
qIII_neg = []

# We Solve for the roots for each coefficient list and only take the positive values for the positive coefficient list
# and negative values for negative coefficient list. We then turn these solutions into arrays so we may edit them more easily.

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

# We sort our solutions and filter out imaginary solutions since they are not physically obtainable.
def del_imaginary(q_list):
    for q in q_list:
        q.sort()
    nrow = q_list.shape[0]
    ncol = q_list.shape[1]
    for i in range(nrow):
        for j in range(ncol):
            if np.imag(q_list[i, j]) != 0:
                q_list[i, j] = np.nan
                
# We also filter out the zeros appended above since only Case I can achieve q = 0.
def del_imaginary_zero(q_list):
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
del_imaginary_zero(qII_pos)
del_imaginary_zero(qII_neg)
del_imaginary_zero(qIII_pos)
del_imaginary_zero(qIII_neg)

# The following section is a stability analysis using the coefficients of the matrices produced in Section 3d. These have been implemented
# in many of the scripts to determine the stability of solutions for differeing parameters. 


s_prec = 0.9e-10
n_precI = south_prec
n_precII = .5*south_prec
n_precIII = 1.5*south_prec


#Once we have calculated the coefficient matrix for feedback analysis, we find the eigenvalues for the solution sets.
def pos_eigen(kd, q, n_prec):
    eigenvalues = []
    index = []
    for i in range(len(k_values)):
        if np.isnan(q[i]):
            pass
        else:
            A11 = -(3*q[i] + 3*kd[i])
            A12 = ((q[i] + kd[i])/(k*beta)) + (n_prec/(q[i] + kd[i]))
            A21 = -(3*q[i]*k*beta)
            A22 = ((k*beta)*((q[i] + kd[i])*(n_prec - s_prec) + (kd[i]*n_prec))/((q[i] + kd[i])**2)) - kd[i]
            value = np.linalg.eigvals([[A11,A12],[A21,A22]])
            eigenvalues.append(value)
            index.append(i)
            
    return np.asarray(eigenvalues), np.asarray(index)


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

# If the real parts of both eigenvalues are negative then the solutions are stable and unstable otherwise.

pos_eigenvalue_qI_pos, pos_index_qI_pos = pos_eigen(k_values,values_qI_pos, n_precI)
pos_eigenvalue_qII_pos1, pos_index_qII_pos1 = pos_eigen(k_values, values_qII_pos1, n_precII)
pos_eigenvalue_qII_pos2, pos_index_qII_pos2 = pos_eigen(k_values, values_qII_pos2, n_precII)
pos_eigenvalue_qIII_pos1, pos_index_qIII_pos1 = pos_eigen(k_values, values_qIII_pos1, n_precIII)
pos_eigenvalue_qIII_pos2, pos_index_qIII_pos2 = pos_eigen(k_values, values_qIII_pos2, n_precIII)

# We plot the stable solutions and unstable solutions below.

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
#We plot the solution for Case I that q = 0 for all values of diffusion.
plt.plot(k_values, qI_pos[:,1], "black")
# We then follow the same process for negative flow.

def neg_eigen(kd, q, n_prec):
    eigenvalues = []
    index = []
    for i in range(len(k_values)):
        if np.isnan(q[i]):
            pass
        else:
            A11 = 3*(q[i] - kd[i])
            A12 =  (-s_prec /(kd[i] - q[i])) + ((q[i] - kd[i])/(k*beta))
            A21 = -3*q[i]*k*beta
            A22 = -kd[i]  -k*beta*(((kd[i] - q[i])*(n_prec - s_prec) - s_prec*kd[i])/((kd[i] - q[i])**2))
            value = np.linalg.eigvals([[A11,A12],[A21, A22]])
            eigenvalues.append(value)
            index.append(i)

    return np.asarray(eigenvalues), np.asarray(index)



values_qI_neg = qI_neg[:,0]
values_qII_neg1 = qII_neg[:,0]
values_qII_neg2 = qII_neg[:,1]
values_qIII_neg1 = qIII_neg[:,0]
values_qIII_neg2 = qIII_neg[:,1]

neg_eigenvalue_qI_neg, neg_index_qI_neg = neg_eigen(k_values, values_qI_neg, n_precI)
neg_eigenvalue_qII_neg1, neg_index_qII_neg1 = neg_eigen(k_values, values_qII_neg1, n_precII)
neg_eigenvalue_qII_neg2, neg_index_qII_neg2 = neg_eigen(k_values, values_qII_neg2, n_precII)
neg_eigenvalue_qIII_neg1, neg_index_qIII_neg1 = neg_eigen(k_values, values_qIII_neg1, n_precIII)
neg_eigenvalue_qIII_neg2, neg_index_qIII_neg2 = neg_eigen(k_values, values_qIII_neg2, n_precIII)



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


ax = plt.subplot(1,1,1)


# Below are various solutions found in the Time dependent model that we may test this steady state model against.
# They are for varying cases and use different values of diffusion. The expected q values are below.
# If they overlap then our model is in agreement.
Kd_I = [.5*10**-10, 1*10**-10, 2*10**-10, 3*10**-10]
Kd_II = [.5*10**-10, 1*10**-10, 2*10**-10, 3*10**-10]
Kd_III = [.25*10**-10, .5*10**-10, .75*10**-10, 1*10**-10, 1.15*10**-10]

init1_QI = [2.79*10**-10, 2.29*10**-10, 1.29*10**-10, .29*10**-10 ]
init1_QII = [2.92*10**-10, 2.59*10**-10, 2.02*10**-10, 1.58*10**-10]
init1_QIII = [2.97*10**-10, 2.63*10**-10, 2.25*10**-10, 1.79*10**-10, 1.32*10**-10]

ax.plot(Kd_I, init1_QI, "ks") 
ax.plot(Kd_II, init1_QII, "ro")
ax.plot(Kd_III, init1_QIII, "bs")


init2_QI = [-2.79*10**-10, -2.29*10**-10, -1.29*10**-10, -.29*10**-10]
init2_QII = [2.92*10**-10, 2.59*10**-10, 2.02*10**-10, 1.58*10**-10]
init2_QIII = [-3.6*10**-10, -3.22*10**-10, -2.53*10**-10, -1.95*10**-10]

### These solutions below had to be solved by hand to find the differences in Salinity
### They were more difficult to find and I had to compute the equations by hand.

unstable_Kd_II = [0, .15*10**-10, .25*10**-10, .35*10**-10, .45*10**-10]
unstable_QII = [-2.32*10**-10, -2.08*10**-10, -1.917*10**-10, -1.725*10**-10, -1.491*10**-10]

ax.plot(unstable_Kd_II, unstable_QII, "rH")


plt.legend()
plt.legend(frameon = False)

plt.xlim(k_start,k_end)
plt.ylim(q_start,q_end)


locs, labels = plt.xticks()
plt.xticks(locs, [0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]) 
locs, labels = plt.yticks()
plt.yticks(locs, [-4,-3,-2,-1,0,1,2,3,4])

plt.xlabel("$K_{d} (10^{-10} \  s^{-1}$)")
plt.ylabel("q $(10^{-10} \ s^{-1})$")
plt.grid(True)

plt.show()





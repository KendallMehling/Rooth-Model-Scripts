# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 12:00:22 2018

@author: kenda
"""

# This is a continuation of a newer conceptual model That allows 
# For multiple Gyre diffusions in the North and South to better 
# Simulate the Overturning Pattern. All equations are now divergent from Longworth (2004) as the Northern and Southern
# diffusion terms have been placed accordingly rather than the single diffusion term k_d.
# We hold the Southern gyre strength constant and vary the Northern gyre strength. 


import numpy as np
import matplotlib.pyplot as plt

e = np.e

k_II = 0

k_start = 0
k_end = 1e-9
k_values = np.linspace(k_start, k_end, 8000)

poly = []

south_prec = .9e-10

q_start = -4 * 10**-10
q_end = 4 * 10**-10

k = 1.5 * 10**-6
beta = 8.0 * 10**-4

def polynomial_pos(kd,s_prec,n_prec):
    poly = []
    for i in range(len(k_values)):
        coeff = [1,kd[i] + k_II, (kd[i] * k_II) - k*beta*s_prec,-(k*beta*(kd[i]*s_prec - k_II*n_prec))]
        poly.append(coeff)
    return poly


polyI_pos = polynomial_pos(k_values,south_prec, south_prec)
polyII_pos = polynomial_pos(k_values, south_prec, .5*south_prec)
polyIII_pos = polynomial_pos(k_values, south_prec, 1.5*south_prec)

def polynomial_neg(kd,s_prec,n_prec):
    poly = []
    for i in range(len(k_values)):
        coeff = [1, -(kd[i] + k_II), (kd[i] * k_II) - k*beta*n_prec,(k*beta*(k_II*n_prec - kd[i]*s_prec))]
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

def qsolve(q,poly,nonneg):
    for i in range(len(k_values)):
        sol = np.roots(poly[i])
        if nonneg:
            sol = [max(s, 0) for s in sol]
        else:
            sol = [min(s, 0) for s in sol]
        q.append(sol)
    return q

qI_pos = qsolve(qI_pos, polyI_pos, True) 
qI_neg = qsolve(qI_neg,polyI_neg, False)

qII_pos = qsolve(qII_pos, polyII_pos, True)
qII_neg = qsolve(qII_neg, polyII_neg, False)

qIII_pos = qsolve(qIII_pos, polyIII_pos, True)
qIII_neg = qsolve(qIII_neg, polyIII_neg, False)


qI_pos = np.asarray(qI_pos)
qI_neg = np.asarray(qI_neg)
qII_pos = np.asarray(qII_pos)
qII_neg = np.asarray(qII_neg)
qIII_pos = np.asarray(qIII_pos)
qIII_neg = np.asarray(qIII_neg)
 
    
    
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
    



ax = plt.subplot(1,1,1)
ax.plot(k_values,qI_pos,"k-")
ax.plot(k_values,qI_neg,"k-")
ax.plot(k_values,qII_pos,"r-")
ax.plot(k_values,qII_neg,"r-")
ax.plot(k_values,qIII_pos,"b-")
ax.plot(k_values,qIII_neg,"b-")

ax.plot(0, 0, "k-", label = "$\Phi_N = \Phi_S$")
ax.plot(0,0, "r-", label = "$\Phi_N = .5 \Phi_S$")
ax.plot(0,0, "b-", label = "$\Phi_N = 1.5 \Phi_S$")
ax.plot(0,0, "None", label = "Kd_II = {}".format(k_II))



plt.legend()
plt.legend(frameon = False)

plt.xlim(k_start,k_end)
plt.ylim(q_start,q_end)

locs, labels = plt.yticks()
plt.yticks(locs, [-4,-3,-2,-1,0,1,2,3,4])
locs, labesl = plt.xticks()
plt.xticks(locs, [0,2,4,6,8,10])


### This Plot shows the effect of having varying effects of differing diffusions in
### The North and the South. Dependent on Southern Gyre Strength



plt.xlabel("$K_{d}\_I \  (10^{-10} \  s^{-1}$)")
plt.ylabel("q $(10^{-10} \ s^{-1})$")
plt.grid(True)

plt.show()


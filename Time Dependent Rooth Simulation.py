# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:09:40 2018

@author: kenda
"""

import numpy as np
import matplotlib.pyplot as plt


### This Script and those to follow are designed to reproduce the results of 
# Longworht (2004) concerning Ocean Dynamics and the Rooth Model. These scripts will
# all be posted on Github as a resource for the future. I hope the commentary is helpful
# For all readers.


# This script models the behavior of the Time dependent characteristics of the boxes for 
# Temperature, salinity, and overturning circulation to give insight into how long it takes
# the model to reach equilibrium and what these final results will be. 
# We use an Euler Forward Step model to write up the equations in Section 3. Gyres in the Rooth Model.


time0 = 0 # 0 years in seconds
timeTempf = 9467280000 # 300 years in seconds
timeSalf =  31557600000 # 1,000 years in seconds
steps = 1001
Tdeltat = (timeTempf - time0)/(steps -1) #increment is equal to 3/10 year in seconds
Sdeltat = (timeSalf - time0)/(steps-1) #increment is equal to one year in seconds
print("Temperature simulation: start (yrs):", time0, "-- end (yrs):", timeTempf/(60*60*24*365.25), "-- increment(yrs):", Tdeltat/(60*60*24*365.25))
print("Salinity simulation: start(yrs):", time0, "-- end (yrs):", timeSalf/(60*60*24*365.25), "-- increment (yrs):", Sdeltat/(60*60*24*365.25))



def euler_forward(initT1, initT2, initT3, initS1, initS2, initS3, kd):
    lam = 12.9*10**-10
    Te =  30
    Tp = 0
    n_prec = .45*10**-10
    s_prec = .9*10**-10
    k = 1.5e-6
    alpha = 1.5e-4
    beta = 8e-4
    
    T1 = np.zeros([steps])
    T2 = np.zeros([steps])
    T3 = np.zeros([steps])
    
    S1 = np.zeros([steps])
    S2 = np.zeros([steps])
    S3 = np.zeros([steps])
    
    q = np.zeros([steps])
    
    #Initial Conditions are established below.
    
    T1[0] = initT1 
    T2[0] = initT2 
    T3[0] = initT3 
    
    S1[0] = initS1
    S2[0] = initS2
    S3[0] = initS3
    
    for step in range(steps-1):
        q[step] = k*(alpha*(T3[step] - T1[step]) - beta*(S3[step] - S1[step]))
        if q[step] >= 0:   
            T1[step+1] = Tdeltat*(lam*(Tp - T1[step]) + kd *(T2[step] - T1[step]) + q[step] *(T2[step] - T1[step])) + T1[step]
            T2[step+1] = Tdeltat*(lam*(Te - T2[step]) + kd *(T1[step] + T3[step] - 2*T2[step]) + q[step]*(T3[step] -T2[step])) + T2[step]
            T3[step+1] = Tdeltat*(lam*(Tp - T3[step]) + kd *(T2[step] - T3[step]) + q[step] *(T1[step] - T3[step])) + T3[step]
            S1[step+1] = Sdeltat*(-n_prec + kd*(S2[step] - S1[step]) + q[step]*(S2[step] - S1[step])) + S1[step]
            S2[step+1] = Sdeltat*(n_prec + s_prec + kd*(S1[step] + S3[step] - 2*S2[step]) + q[step]*(S3[step] - S2[step])) + S2[step]
            S3[step+1] = Sdeltat*(-s_prec + kd*(S2[step] - S3[step]) + q[step]*(S1[step] - S3[step])) + S3[step]
        else: #negative flow solutions
            T1[step+1] = Tdeltat*(lam*(Tp - T1[step]) + kd *(T2[step] - T1[step]) + abs(q[step]) *(T3[step] - T1[step])) + T1[step]
            T2[step+1] = Tdeltat*(lam*(Te - T2[step]) + kd *(T1[step] + T3[step] - 2*T2[step]) + abs(q[step])*(T1[step] -T3[step])) + T2[step]
            T3[step+1] = Tdeltat*(lam*(Tp - T3[step]) + kd *(T2[step] - T3[step]) + abs(q[step]) *(T2[step] - T3[step])) + T3[step]
            S1[step+1] = Sdeltat*(-n_prec + kd*(S2[step] - S1[step]) + abs(q[step])*(S3[step] - S1[step])) + S1[step]
            S2[step+1] = Sdeltat*(n_prec + s_prec + kd*(S1[step] + S3[step] - 2*S2[step]) + abs(q[step])*(S1[step] - S2[step])) + S2[step]
            S3[step+1] = Sdeltat*(-s_prec + kd*(S2[step] - S3[step]) + abs(q[step])*(S2[step] - S3[step])) + S3[step]
    return (T1, T2, T3,) , (S1, S2, S3)


### typical example: kd = 1*10**-10 We use this model to see how the temperatures and salinities 
# adjust to initial conditions that determine the overturning circulation.

kdsim = 1*10**-10
timeTemp = np.linspace(time0, timeTempf/(1*10**10), steps)
timeSal = np.linspace(time0, timeSalf/(1*10**10), steps)

#The values for the inital conditions but can be given conditions around equilibrium using the equations
# found in Section 3c.

Temp, Sal = euler_forward(32,10,19, 31.7,32,33, kdsim)
Temp = np.transpose(Temp)
Sal = np.transpose(Sal)

# We acess the information and then plot the temperature adjustments based on the temperature time scale
# and the salinity is plotted against a salinity time scale. 


fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(timeTemp, Temp[:,0],"r--", label = "Northern high latitude")
plt.plot(timeTemp, Temp[:,1], "k-", label = "Tropical")
plt.plot(timeTemp, Temp[:,2], "g-.", label = "Southern high lattitude")
plt.xlabel("Time "+ "(1^10s)", size = 12, weight = 'bold')
plt.ylabel("Temperature (C)", size = 12, weight = 'bold')
plt.legend()
plt.subplot(2,1,2)
plt.plot(timeSal, Sal[:,0], "r--", timeSal, Sal[:,1], "k-", timeSal, Sal[:,2], "g-.")
plt.xlabel("Time" + "(1^10s)", size = 12, weight = 'bold')
plt.ylabel("Salinity (psu)", size = 12, weight = 'bold')

plt.show()

#Shows how long the system takes to equilibrate and the final values for given initial conditions.
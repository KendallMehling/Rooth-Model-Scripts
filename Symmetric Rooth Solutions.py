# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 13:36:01 2018

@author: kenda
"""
#Using the same treatment for flow characteristics I will set T1 = T3 to 
#Represent the Symmetric Temperature Disparity as Donce in Section C
#To reproduce Fig 6 to find various steady state flows as a function 
#Of K_d. This Script is helpful in finding steady state q solutions using the outcome
# of the time dependent model. This may be used to gather reference points to plot onto
# the steady state graph to see if they are in agreement. 


import numpy as np
import matplotlib.pyplot as plt

def my_timer(func):
    import time
    
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = round(time.time() - t1,3)
        print("{} ran in :{} sec ".format(func.__name__, t2))
        return result
    
    return wrapper



time0 = 0  # 0 years in seconds
timeTempf =  9467280000  # 300 years in seconds
timeSalf = 2 * 31557600000  # 2,000 years in seconds
steps = 100001
Tdeltat = (timeTempf - time0) / (steps - 1)  # increment is equal to 3/1000 year in seconds
Sdeltat = (timeSalf - time0) / (steps - 1)  # increment is equal to 1/100 year in seconds
print("Temperature simulation: start (yrs):", time0, "-- end (yrs):", timeTempf / (60 * 60 * 24 * 365.25),
      "-- increment(yrs):", Tdeltat / (60 * 60 * 24 * 365.25))
print("\nSalinity simulation: start(yrs):", time0, "-- end (yrs):", timeSalf / (60 * 60 * 24 * 365.25),
      "-- increment (yrs):", Sdeltat / (60 * 60 * 24 * 365.25))
print("\n")


@my_timer
def euler_forward(initT1, initT2, initT3, initS1, initS2, initS3, kd):
    lam = 12.9 * 10 ** -10
    Te = 30
    Tp = 0
    n_prec = .9 * 10 ** -10 # Case I: Change this case to find various steady-states.
    s_prec = .9 * 10 ** -10
    k = 1.5 * 10 ** -6
    alpha = 1.5 * 10 ** -4
    beta = 8.0 * 10 ** -4
    
    t1 = []
    t2 = []
    t3 = []
    s1 = []
    s2 = []
    s3 = []
    q = []


    t1.append(initT1)
    t2.append(initT2)
    t3.append(initT1)

    s1.append(initS1)
    s2.append(initS2)
    s3.append(initS3)
    

    q.append(k * (alpha * (t3[0] - t1[0]) - beta * (s3[0] - s1[0])))
    
    
    for step in range(0,steps - 1):
        if q[step] >= 0:  # nonnegative flow solutions
            
            t1.append(Tdeltat * (lam * (Tp - t1[step]) + kd * (t2[step] - t1[step]) + q[step] * (t2[step] - t1[step])) + t1[step])
    
            t2.append(Tdeltat * (lam * (Te - t2[step]) + kd * (t1[step] + t3[step] - 2 * t2[step]) + q[step] * (t3[step] - t2[step])) + t2[step])
    
            t3.append(Tdeltat * (lam * (Tp - t1[step]) + kd * (t2[step] - t1[step]) + q[step] * (t2[step] - t1[step])) + t1[step]) ### Symmetric Flow T1 = T3
    
            s1.append(Sdeltat * (-n_prec + kd * (s2[step] - s1[step]) + q[step] * (s2[step] - s1[step])) + s1[step])
    
            s2.append(Sdeltat * (n_prec + s_prec + kd * (s1[step] + s3[step] - 2 * s2[step]) + q[step] * (s3[step] - s2[step])) + s2[step])
    
            s3.append(Sdeltat * (-s_prec + kd * (s2[step] - s3[step]) + q[step] * (s1[step] - s3[step])) + s3[step])
    
            q.append(k * (alpha * (t3[step] - t1[step] ) - beta * (s3[step] - s1[step])))
            
        
        elif q[step ] < 0: # negative flow solutions
 
            t1.append(Tdeltat * (lam * (Tp - t1[step]) + kd * (t2[step] - t1[step]) + abs(q[step]) * (t3[step] - t1[step])) + t1[step])
            
            t2.append(Tdeltat * (lam * (Te - t2[step]) + kd * (t1[step] + t3[step] - 2 * t2[step]) + abs(q[step]) * (t1[step] - t2[step])) + t2[step])
            
            t3.append(Tdeltat * (lam * (Tp - t1[step]) + kd * (t2[step] - t1[step]) + abs(q[step]) * (t3[step] - t1[step])) + t1[step]) ### Symmetric Flow T1 = T3
            
            s1.append(Sdeltat * (-n_prec + kd * (s2[step] - s1[step]) + abs(q[step]) * (s3[step] - s1[step])) + s1[step])
            
            s2.append(Sdeltat * (n_prec + s_prec + kd * (s1[step] + s3[step] - 2 * s2[step]) + abs(q[step]) * (s1[step] - s2[step])) + s2[step])
            
            s3.append(Sdeltat * (-s_prec + kd * (s2[step] - s3[step]) + abs(q[step]) * (s2[step] - s3[step])) + s3[step])
            
            q.append(k * (alpha * (t3[step] - t1[step] ) - beta * (s3[step] - s1[step])))

    return (t1, t2, t3), (s1, s2, s3), q



### This program is helpful in finding the steady state flow solutions for various K_d
### Hopefully our outputs for various initial conditions will align with values found 
###in Figure 6 of the Longworth Paper. 


kdsim =  1e-11

timeTemp = np.linspace(time0, timeTempf / (1 * 10 ** 10), steps)
timeSal = np.linspace(time0, timeSalf / (1 * 10 ** 10), steps)


Temp, Sal, Q = euler_forward(.1,10,1, 35, 35.7, 34.3 , kdsim)


Temp = np.transpose(Temp)
Sal = np.transpose(Sal)


print(Q[100000])

##One may check if the flow has reached equilibrium but generally it has. The initial conditions are helpful as thye may 
## plot the steady state onto different solution branches.
            
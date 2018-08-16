import numpy as np
import matplotlib.pyplot as plt
import random

# This model will determine the type of instability that is predicted in Fig 8. in the 
# paper by Longworth. et al. Using the Forward step model, we may see how the q reacts in the regions of stability
# and instability. We will perturb the salinity to get the instability to arise in the forward model.

def my_timer(func):
    import time
    
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = round(time.time() - t1,3)
        print("{} ran in :{} sec ".format(func.__name__, t2))
        return result
    
    return wrapper



years = 2000000
steps = 2000001
time_initial = 0 
time_final = years * (60*60*24*365.25) # years in seconds
Tdeltat = ((time_final - time_initial) / steps) #change in time in seconds


n_prec_initial = 0 # Initial Salinity Flux
n_prec_final = 1.5e-9 # Final Salinity Flux Concerned

n_prec_values = np.linspace(n_prec_initial,n_prec_final, steps)


@my_timer
def euler_forward(initT1, initT2, initT3, initS1, initS2, initS3, kd):
    lam = 12.9e-10
    Te =  30
    Tp = 0
    alpha = 1.5e-4
    k = 1.5e-6
    beta = 8e-4

    s_prec = 0.9e-10
    n_prec = n_prec_values
    
    T1 = np.zeros([steps])
    T2 = np.zeros([steps])
    T3 = np.zeros([steps])
    
    S1 = np.zeros([steps])
    S2 = np.zeros([steps])
    S3 = np.zeros([steps])
    
    q = np.zeros([steps])
    
    T1[0] = initT1 
    T2[0] = initT2 
    T3[0] = initT1
    
    S1[0] = initS1
    S2[0] = initS2
    S3[0] = initS3
    
    for step in range(steps-1):
        q[step] = k*((alpha * (T3[step] - T1[step])) - (beta * (S3[step] - S1[step])))
        if q[step] > 0:
            
            T1[step+1] = Tdeltat*(lam*(Tp - T1[step]) + kd *(T2[step] - T1[step]) + q[step] *(T2[step] - T1[step])) + T1[step]
            T2[step+1] = Tdeltat*(lam*(Te - T2[step]) + kd *(T1[step] + T3[step] - 2*T2[step]) + q[step]*(T3[step] -T2[step])) + T2[step]
            T3[step+1] = Tdeltat*(lam*(Tp - T1[step]) + kd *(T2[step] - T1[step]) + q[step] *(T2[step] - T1[step])) + T1[step] #T1 = T3 Symmetry
            S1[step+1] = Tdeltat*(-n_prec[step] + kd*(S2[step] - S1[step]) + q[step]*(S2[step] - S1[step])) + S1[step]
            S2[step+1] = Tdeltat*(n_prec[step] + s_prec + kd*(S1[step] + S3[step] - 2*S2[step]) + q[step]*(S3[step] - S2[step])) + S2[step]
            S3[step+1] = Tdeltat*(-s_prec + kd*(S2[step] - S3[step]) + q[step]*(S1[step] - S3[step])) + S3[step] + (random.randint(-1000, 1000) * 1e-12)
            
        else: #negative flow solutions
        
            T1[step+1] = Tdeltat*(lam*(Tp - T1[step]) + kd *(T2[step] - T1[step]) + abs(q[step]) *(T3[step] - T1[step])) + T1[step]
            T2[step+1] = Tdeltat*(lam*(Te - T2[step]) + kd *(T1[step] + T3[step] - 2*T2[step]) + abs(q[step])*(T1[step] -T3[step])) + T2[step]
            T3[step+1] = Tdeltat*(lam*(Tp - T1[step]) + kd *(T2[step] - T1[step]) + abs(q[step]) *(T3[step] - T1[step])) + T1[step] # T1 = T3 Symmetry
            S1[step+1] = Tdeltat*(-n_prec[step] + kd*(S2[step] - S1[step]) + abs(q[step])*(S3[step] - S1[step])) + S1[step]
            S2[step+1] = Tdeltat*(n_prec[step] + s_prec + kd*(S1[step] + S3[step] - 2*S2[step]) + abs(q[step])*(S1[step] - S2[step])) + S2[step]
            S3[step+1] = Tdeltat*(-s_prec + kd*(S2[step] - S3[step]) + abs(q[step])*(S2[step] - S3[step])) + S3[step]

    return q
# These initial conditions with the above equations will plot the flow as time progresses and we see that it is in agreement with 
# the findings of Longworth et. al (2004) Figure 8. Our stability model has been further coroborated. This only works for temperature
# symmetry about the equator. That is how much of Longworth's work is conducted but we are not confined to this. 

kdsim_black = 0
kdsim_red = 1e-11
kdsim_blue = 1e-10
kdsim_green = 4e-10

q_values_black = euler_forward(5.06, 24.94,5.06,35.09,35.06, 34.817, kdsim_black) 

q_values_red = euler_forward(5.08,24.75,5.08,35.08,35.89,34.82, kdsim_red)

q_values_blue = euler_forward(7,10,7,35,35.7,2, kdsim_blue)

q_values_green = euler_forward(7,10,7,35,35.7,2, kdsim_green)


plt.ylim(-15e-10,6e-10 )
plt.xlim(0, 1.5e-9)

locs, labels = plt.xticks()
plt.xticks(locs, [0,2,4,6,8,10,12,14])
locs, labels = plt.yticks()
plt.yticks(locs, [-15,-12.5,-10,-7.5,-5,-2.5,0,2.5,5,7.5])

plt.plot(n_prec_values, q_values_black, "black")
plt.plot(n_prec_values, q_values_red, "red")
plt.plot(n_prec_values, q_values_blue, "blue")
plt.plot(n_prec_values, q_values_green, "green")


plt.xlabel("$\Phi_N (10^{-9} psu \ s^{-1} $)")
plt.ylabel("q $(10^{-10} \ s^{-1})$")
plt.grid(True)



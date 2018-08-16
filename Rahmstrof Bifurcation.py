import numpy as np
import matplotlib.pyplot as plt

# This script is used to reproduce the bifurcation model (Fig 2) of Rahmstorf (1996). Arbitrary values are used in the equation
# To achieve the bifurcation limit he discusses in his paper.

e = np.e
S0 = 35

k = 1.5e-6
alpha = 1.5e-4
beta = 8.0e-4


forcing_1 = np.linspace(-1e-10,1.26e-10, 10000)
T2 = 0
T1 = 20

m = []
for i in range(len(forcing_1)):
    m.append((-.5 * k * alpha * (T2 - T1)) + np.sqrt((k*alpha*(T2 -T1))**2 - 4*k * beta * S0 * forcing_1[i]))


forcing_2 = np.linspace(-1e-10,0,10000)
T2 = 20
T1 = 0

m2 = []
for i in range(len(forcing_2)):
    m2.append((-.5 * k * alpha * (T2 - T1)) + np.sqrt(.25*(k*alpha*(T2 -T1))**2 - k * beta * S0 * forcing_2[i]))


T2 = 0
T1 = 20

m3 = []
forcing_3 = np.linspace(0,1.2e-10,10000)

for i in range(len(forcing_3)):
    m3.append((-.5 * k * alpha * (T2 - T1)) - np.sqrt(.25*(k*alpha*(T2 -T1))**2 - k * beta * S0 * forcing_3[i]))

plt.plot(forcing_1, m, "r")
plt.plot(forcing_2,m2,"g")
plt.plot(forcing_3,m3,"c--")
plt.ylim(-.02e-8,1e-8)
plt.xlim(-1.25e-10,1.5e-10,0)
plt.grid(True)

criticalx = [0,0,k*alpha**2*(T2-T1)**2/(4*beta*S0)]
criticaly = [0,.675e-8, k*alpha*(T1-T2)/2]

plt.plot(criticalx,criticaly, "ko")
plt.ylim(0,1e-8)
plt.xlim(-1.5e-10,1.5e-10)
locs, labels = plt.yticks()
plt.yticks(locs, [0,2,4,6,8,1.0])

locs, labels = plt.xticks()
plt.xticks(locs, [-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])


plt.ylabel("NADW Flow")
plt.xlabel("Freshwater Forcing ")

plt.text(-.85e-10,.08e-8,"Haline", rotation = -10, weight = "bold" )
plt.text(-.85e-10,.83e-8,"Thermohaline", rotation = -16, weight = "bold" )
plt.text(.8e-10,.49e-8,"Thermal", rotation = -32, weight = "bold" )
plt.text(.65e-10,.065e-8,"(Unstable)", rotation = 12, weight = "bold" )
plt.text(-1.25e-10, .84e-8, " T2 < T1", size = 8, weight = "bold")
plt.text(-1.25e-10, .09e-8, " T2 > T1", size = 8, weight = "bold")
plt.text(1.11e-10, .95e-8, "Bifurcation Limit", rotation = 90, weight = "bold", size = 10)

point1 = [(k*alpha**2*(T2-T1)**2)/(4*beta*S0),(k*alpha**2*(T2-T1)**2)/(4*beta*S0)]
point2 = [-1,1]

present1 = [.58e-10,.65e-10]
present2 = [-1,1]

plt.plot(.615e-10, .54e-8, "yo")
plt.plot(point1, point2, "k--", marker = "o")
plt.plot(present1, present2, "y--")

plt.text(.52e-10, .50e-8,"Approx. Present Climate", rotation = 90, size = 8, weight = "bold")







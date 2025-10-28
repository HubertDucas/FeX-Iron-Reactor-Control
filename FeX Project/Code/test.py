import numpy as np
import matplotlib.pyplot as plt

# Constants
cp_bed = 450
Ci = 450
alpha = cp_bed / Ci

dt = 1e-1
t = np.arange(0, 50, dt)

# Equilibrium
Qdot_gen_i = 100000
Tin = 50
Tout_bar = 700
mdot_bar = Qdot_gen_i / (cp_bed * (Tout_bar - Tin))

# Control parameters
kp = 0.0001    # smaller gain helps stability

# Storage
Tout_hist = np.zeros_like(t)
mdot_hist = np.zeros_like(t)

# Initial conditions
Tout_hist[0] = Tin
mdot_hist[0] = mdot_bar
deltaTout = Tout_hist[0] - Tout_bar

for k in range(t.shape[0] - 1):
    # feedback controller
    error = Tout_bar - Tout_hist[k] + np.random.normal(0, 10)
    mdot = mdot_bar + kp * error
    deltamdot = mdot - mdot_bar

    # one step update (Euler)
    deltaTout = deltaTout + dt * ( -alpha*mdot_bar*deltaTout + alpha*(Tin - Tout_bar)*deltamdot )
    Tout_next = Tout_bar + deltaTout

    # store
    Tout_hist[k+1] = Tout_next
    mdot_hist[k+1] = mdot


# plot the linearized Tout(t) as a function of t 
plt.plot(t, Tout_hist)
plt.xlabel("Time(s) Label")
plt.ylabel("Tout (C) Label")
plt.title("Tout vs time for Toutbar=700")
plt.show()

# plot the linearized mdot(t) as a function of t 
plt.plot(t, mdot_hist)
plt.xlabel("Time(s) Label")
plt.ylabel("mdot (kg/s) Label")
plt.title("mdot vs time for Toutbar=700")
plt.show()


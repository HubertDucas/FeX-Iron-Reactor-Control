import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp 

# Constants 
cp_bed = 450 # J/kgK
Ci = 450 # J/kgK

# Time
dt = 1e-1
t = np.arange(0, 50, dt)
steps = t.shape

# Equilibrium points
Qdot_gen_i = 100e3 # J (100 kJ/s)
Tin = 50 + 275 # K
Tout_bar = 700 + 275 # K
mdot_bar = Qdot_gen_i/(cp_bed * (Tout_bar - Tin)) # kg/s
print(mdot_bar)
# print(mdot_bar)

# Example mdot(t) input 
mdot = np.sin(t)
# print(mdot - mdot_bar)


###################################

def linearized_step(Tout_k, mdot_k):
    deltaTout_k = Tout_k - Tout_bar
    deltamdot_k = mdot_k - mdot_bar

    deltaTout_kp1 = deltaTout_k + dt * (- (cp_bed/Ci) * mdot_bar * deltaTout_k + 
                               (cp_bed/Ci) * (Tin - Tout_bar) * deltamdot_k)
    
    Tout_kp1 = deltaTout_kp1 + Tout_bar
    return Tout_kp1

###################################

###################################

def PID(kp, ki, kd, sp, Tout, integral, previous_error):
    """
    Just a PID controller, need previous error for de/dt and integral 
    (sum of prevoius integrals) for int(e)dt

    Inputs: kp, ki, kd, sp (sampling period), Tout, integral, previous_error (all scalars since we want)
    Outputs: integral, derivative, mdot
    """

    noise = np.random.normal(0, 10) # 1C std 
    # error = Tout_bar - (Tout + noise) # error neg feedback
    error = (Tout + noise) - Tout_bar # error pos feedfwd
    integral += error * sp
    derivative = (error - previous_error) / sp
    mdot_error = kp * error + ki * integral + kd * derivative # increase mdot when T < Tout_bar

    mdot = mdot_bar + mdot_error
    return mdot, error, integral

# we have a problme here: Tin is much smaller than Tout_bar so when solving the 
# ODE via backward Euler, increasing mdot (or deltamdot) will DECREASE the temperature 
# but the proportional part of the controller is made so that increasing the error, 
# increases the mdot to reach the temperature.. but...but... it doesn't work since the 
# ODE solver says the opposite: ODE says increasing mdot decreases the temperature, controller 
# says increaseing mdot increasess the temperature... WRONG: need feedback and feedfwd at 
# same time

###################################

# Gains
kp = 0.01 # 0.0001
ki = 0.5 # 0.01
kd = 0 # 0.000001 

# Sampling Frequency
sp = 0.001

# Initial values 
initial_integral = 0
previous_error = 0 

# Solver for linearized model with step definition
Tout_solution = np.zeros((t.shape[0]))
mdot_solution = np.zeros((t.shape[0]))
Tout_solution[0] = Tin
mdot_solution[0] = PID(kp, ki, kd, sp, Tin, initial_integral, previous_error)[0]

for k in range(t.shape[0] - 1):
    Tout_k = Tout_solution[k]
    mdot_k = mdot_solution[k]

    Tout_kp1 = linearized_step(Tout_k=Tout_k, mdot_k=mdot_k)
    Tout_solution[k+1] = Tout_kp1

    mdot_kp1 = PID(kp, ki, kd, sp, Tout_kp1, initial_integral, previous_error)[0]
    mdot_solution[k+1] = mdot_kp1

###################################

# plot the linearized Tout(t) as a function of t 
plt.plot(t, Tout_solution)
plt.xlabel(r"$Time(s)$")
plt.ylabel(r"$T_{out} (^{\circ}C)$")
# plt.title(r"$T_{out}(t)$ for $\bar{T}_{out}=700^{\circ}C$")
plt.show()

# plot the linearized mdot(t) as a function of t 
plt.plot(t, mdot_solution)
plt.xlabel(r"$Time(s)$")
plt.ylabel(r"$\dot{m} (kg/s)$")
# plt.title(r"$\dot{m}(t)$ for $\bar{T}_{out}=700^{\circ}C$")
plt.show()




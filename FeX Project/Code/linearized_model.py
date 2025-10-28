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

#  Physics model - Linearized ODE (backward Euler)

# SINE IN SHOULD SETTLE TO 0 
def linearized_model(mdot):
    """
    This linearized model is a simple implementation
    of equation 25. 

    Input: whatever mdot function you want (e.g.: mdot(t) = 10sin(t))
        important note: when solving the controlled system with the feedback loop 
        we'll want to input mdot to this function as a scalar at a given timestep
        which can also be done. mdot can be an array defined by mdot(t) or simply 
        a scalar 
    Output: Tout(t), deltaTout(t) + plots (Tout vs t, Tout vs mdot, deltaTout vs deltamdot) 
    """
    # linearized input
    deltamdot = mdot - mdot_bar 

    # initialize arrays 
    deltaTout = np.zeros((t.shape[0]))
    # first temp is Tin 
    deltaTout[0] = Tin - Tout_bar

    # solve for deltaTout(t) given deltamdot(t) 
    # solve eq. 25
    for k in range(t.shape[0] - 1):
        deltaTout[k+1] = deltaTout[k] + dt * (- (cp_bed/Ci) * mdot_bar * deltaTout[k] + 
                               (cp_bed/Ci) * (Tin - Tout_bar) * deltamdot[k])
        
    # now have deltaTout(t) as a column matrix, can get Tout(t) 
    Tout = deltaTout + Tout_bar

    return Tout, deltaTout, deltamdot

###################################

# Linearized system solution
Tout, deltaTout, deltamdot = linearized_model(mdot)

###### PLOT ######

# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# plot the linearized Tout(t) as a function of t 
plt.plot(t, Tout)
plt.xlabel("Time(s) Label")
plt.ylabel("Tout (C) Label")
plt.title("Tout vs time for Toutbar=700")
plt.show()

# plot the linearized Tout(t) as a function of mdot(t) 
plt.plot(mdot, Tout)
plt.xlabel("Time(s) Label")
plt.ylabel("Tout (C) Label")
plt.title("Tout vs mdot for Toutbar=700")
plt.show()

# plot small changes of Tout as a function of small changes of mdot 
plt.plot(deltamdot, deltaTout)
plt.xlabel("Time(s) Label")
plt.ylabel("Tout (C) Label")
plt.title("deltaTout vs deltamdot for Toutbar=700")
plt.show()


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

# Control loop
def feedback_loop(Tout):
    """
    The feedback loop is an implementation of Fig. 5. It takes in 
    Tout computed at timestep k by the function linearized_model(mdot)
    and gives the mdot that should be used at the next iteration

    Input: Tout (scalar)
    Output: mdot (scalar)
    """

    # compute the error
    noise = np.random.normal(0, 10) # 1C std 
    error = Tout_bar - (Tout + noise)

    # compute the reference mass flow rate 
    # (reference or error scaled by propoortoinal controller gain)
    kp = 0.0001
    error_mdot = error * kp 

    mdot = mdot_bar + error_mdot
    return mdot

###################################




###################################

# # Solver for linearized model function 

# # Combine the physical model and the feedback loop
# # Initialize the controlled Tout solution array  
# # Tout_controlled = np.zeros((t.shape[0]))

# # initialize the mdot array 
# mdot_solution = np.zeros((t.shape[0]))
# mdot_initial = feedback_loop(Tin)
# mdot_solution[0] = mdot_initial
# # print(mdot_controlled)
# mdot_controlled = np.ones((t.shape[0])) * mdot_initial


# for k in range(0, t.shape[0] - 1):
#     Tout_solution, _, _ = linearized_model(mdot_controlled) # this gives the array for the whole time range
#                                                     # but here at the first timestep, we feed an mdot_controlled
#                                                     # that only has info for the first timestep... so the solution is 
#                                                     # only valid for the next timestep, it gets updated 
#                                                     # after an new mdot has been computed and so on 

#     # compute mdot @ k+1
#     mdot_controlled[k+1] = feedback_loop(Tout_solution[k+1])
#     # print("The Tout_solution array at timestep", k, "is", Tout_solution)
#     # print("The mdot_controlled array at timestep", k, "is", mdot_controlled)

###################################



# ###################################

# Solver for linearized model with step definition
Tout_solution = np.zeros((t.shape[0]))
mdot_solution = np.zeros((t.shape[0]))
Tout_solution[0] = Tin
mdot_solution[0] = feedback_loop(Tin)

for k in range(t.shape[0] - 1):
    Tout_k = Tout_solution[k]
    mdot_k = mdot_solution[k]

    Tout_kp1 = linearized_step(Tout_k=Tout_k, mdot_k=mdot_k)
    Tout_solution[k+1] = Tout_kp1

    mdot_kp1 = feedback_loop(Tout_kp1)
    mdot_solution[k+1] = mdot_kp1

# ###################################






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


# Code
# figure out what's going on with the crazy sine amplitude change 
# investigate on PI, PID 
# Normalize things to not have to use kp = 0.0001


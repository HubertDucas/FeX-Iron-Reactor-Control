import numpy as np 
import scipy 
from matplotlib import pyplot as plt 

##########################
# PHYSICAL MODEL 

# Define constants 
L = 1 # (m) arbitrary length of the tube 
dl = 0.05 # (m) arbitrary step size for the tube

T = 50 # (s) time of the simulation (physical model and control model should run on different timeframes)
dt = 0.01 # (s) time step 

d = 0.0254 # (m) arbitrary 1 inch diameter 

# Start wit general C_i and Qdot_gen_i
Ci = 450 # arbitrary value
Qdot_gen_i = 10 #arbitrary value 

rho_iron = 7874 # (kg/m^3) change later 
rho_bed = rho_iron * 0.5 # arbitrary 50% infill, change later 
rho_ss = 8000 # (kg/m^3) change later 

cp_bed = 450 # J/kgK
cp_ss = 500 # J/kgK


# Define vector spaces 
x = np.arange(0, L, dl) # x-axis (along the length of the reactor)
t_physics = np.arange(0, T, dt) # physical model  
t_controller = np.arange(0, T, dt) # control model 

# # Define variables (do that later)
# l_i = L - l_r # define what l_r is later (it's just the height of the iron oxide)
# V_bed = l_i * math.pi * (d/2)**2 
# C_i = V_bed
# Define transfer function in time domain (first need to transform the TF in and ODE)


# Equilibrium points 
Tout_bar = 700 # (deg C) arbitrary target output temp
mdot_bar = 1 # arbitrary target input mass flow rate (keep constant)

# Define Tin for reference 
Tin = 600

# initial_Tout = 100
# initial_mdot = 1

ICs = np.array(([Tout_bar, mdot_bar]))

# Thermocouples temperatures (1:bottom, 4:top)
T1 = 175 
T2 = 350 
T3 = 525 
T4 = 700 

thermocouples = np.array([T1, T2, T3, T4])

# Solve physical model
# This first solver uses a constant C_i and Qdot_gen_i

# Compute mdot
def compute_mdot(mdot_bar, deltamdot):
    mdot = mdot_bar + deltamdot
    return mdot

# # Linearized enthalpy model (eq. 25)
# def linarized_enthalpy_model(Tout, mdot, Tout_bar, mdot_bar, t, Tin): # for now, don't include x-axis
#     # initialize 
#     Tout_dot = np.array((t.shape[0], 1))

#     for ti in range(t):
#         Tout_dot[ti] = -(cp_bed/Ci) * mdot_bar * Tout[ti] + (cp_bed/Ci) * (Tin - Tout_bar) * mdot

#     return Tout_dot

# solve the differential equation using an Euler discretization
def forward_euler(Tout_k, mdot_k, Tout_bar, mdot_bar, Tin, dt, cp, Ci):
    Tout_kp1 = Tout_k + dt * (-((cp/Ci) * mdot_bar * Tout_k) + ((cp/Ci) * (Tin - Tout_bar) * mdot_k))
    return Tout_kp1 
    

##########################
# CONTROL MODEL 
# Define the gains
# tweak later
proportional_gains = np.array([1, 1, 1, 1]) 

# Add Gaussian noise to Tout from plant, and compute deltaTout 
def compute_deltaTout(Tout):
    # add Gaussian noise to the array of Tout (at each timestep t)
    noise = np.random.normal(0, 1) # start with an arbitrary 1 deg C std
    Tn = Tout + noise
    deltaTout = Tout_bar - Tn
    return deltaTout

# Compute deltamdot using a proportional controller 
def compute_deltamdot(proportional_gains, deltaTout, Tout):
    # choose the appropriate gain 
    # depending on what Tout is, take gain of thermocouple whose temperature is closest to Tout
    index = np.argmin(np.abs(Tout - thermocouples))
    k_p = -proportional_gains[index]
    deltamdot = k_p * deltaTout 
    return deltamdot

##########################
# SOLVE 
# initialize matrces 
# matrix shape of interest
mat_shape = (int(T/dt), 1)
Tout = np.zeros(mat_shape)
Tout[0] = Tin

# Column matrices
Tn = np.zeros(mat_shape)
deltaTout = np.zeros(mat_shape)
deltamdot = np.zeros(mat_shape)
mdot = np.zeros(mat_shape)


# not ethat mdot is defined at each timestep by functions 
# so no need to initialize it 

for tk in range(int(T/dt) - 1):
    # compute deltaTout 
    deltaTout[tk] = compute_deltaTout(Tout[tk]) # get the column matrix of deltaTout updated at each timestep 
    # compute deltamdot
    deltamdot[tk] = compute_deltamdot(proportional_gains, deltaTout[tk], Tout[tk]) # get deltamdot (scalar)
    # compute mdot
    mdot[tk] = compute_mdot(mdot_bar, deltamdot[tk])
    # mdot = np.clip(mdot, 0.1, 1)

    # get Tout !
    # we're feeding the function with just the column matri x of Tout, the current mdot
    # and the other relevant info
    # get out Tout at the next step Tout[k+1] (you dont actually get it out, 
    # it just changes the array)
    Tout[tk+1] = forward_euler(Tout[tk], mdot[tk], Tout_bar, mdot_bar, Tin, dt, cp_bed, Ci)


# Alternative Euler computation 
# Simulation loop (Euler integration)
# for k in range(int(T/dt) - 1):
#     # Sensor noise
#     noise = np.random.normal(0, 1.0)

#     # Measure output with noise
#     measured_Tout = Tout[k] + noise

#     # Compute control input (Proportional only)
#     error = Tout_bar - measured_Tout
#     delta_mdot = -1 * error
#     mdot[k] = mdot_bar + delta_mdot

#     # Euler step
#     Tout[k+1] = Tout[k] + dt * (-((cp_bed/Ci) * mdot_bar * Tout[k]) + ((cp_bed/Ci) * (Tin - Tout_bar) * mdot[k]))


print("Tout is", Tout)
# print("deltaTout is", deltaTout)
# print("deltamdot is", deltamdot)
##########################
# PLOT 
# the goal is to plot Tout as a function of time for different Tout (say 10 in a range 690-710C to see 
# what are the deviations from the equilibrium values that lead to a catastrophic outcome)

# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# Plot Tout vs time
plt.plot(t_physics, Tout)
plt.xlabel("time(s) Label")
plt.ylabel("Tout (C) Label")
plt.title("Tout vs time for Toutbar=700")
plt.show()

# # Plot detaTout vs time
# plt.plot(t_physics, deltaTout)
# plt.xlabel("time(s) Label")
# plt.ylabel("deltaTout (C) Label")
# plt.title("deltaTout vs time for Toutbar=700")
# plt.show()

# Plot mdot vs time
plt.plot(t_physics, mdot)
plt.xlabel("time(s) Label")
plt.ylabel("mdot (C) Label")
plt.title("mdot vs time for Toutbar=700")
plt.show()
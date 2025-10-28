import numpy as np 
import scipy 
from matplotlib import pyplot as plt 

##########################
# PHYSICAL MODEL 

# Define constants 
L = 1 # (m) arbitrary length of the tube 
dl = 0.05 # (m) arbitrary step size for the tube

T = 20 # (s) time of the simulation (physical model and control model should run on different timeframes)
dt = 0.1 # (s) time step 

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

# Define Tin for reference 
Tin = 600

# Equilibrium points 
Tout_bar = 700 # (deg C) arbitrary target output temp
mdot_bar = Qdot_gen_i/(cp_bed * (Tout_bar - Tin)) # arbitrary target input mass flow rate (keep constant)


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
def forward_euler(Tout, mdot, Tout_bar, mdot_bar, Tin, T, dt, tk, cp, Ci):
    Tout[tk+1] = Tout[tk] + dt * (-((cp/Ci) * mdot_bar * Tout[tk]) + ((cp/Ci) * (Tin - Tout_bar) * mdot))
    return Tout 
    

##########################
# CONTROL MODEL 
# Define the gains
# tweak later
proportional_gains = np.array([1, 1, 1, 1]) 

# Add Gaussian noise to Tout from plant, and compute deltaTout 
def compute_deltaTout(Tout, tk):
    # add Gaussian noise to the array of Tout (at each timestep t)
    noise = np.random.normal(0, 1, (int(T/dt), 1)) # start with an arbitrary 1 deg C std
    Tn[tk] = Tout[tk] + noise[tk]
    deltaTout[tk] = Tout_bar - Tn[tk]
    return deltaTout

# Compute deltamdot using a proportional controller 
def compute_deltamdot(proportional_gains, deltaTout, Tout, tk):
    # choose the appropriate gain 
    # depending on what Tout is, take gain of thermocouple whose temperature is closest to Tout
    index = np.argmin(np.abs(Tout[tk] - thermocouples))
    k_p = proportional_gains[index]
    deltamdot = k_p * deltaTout[tk] 
    return deltamdot

##########################
# SOLVE 
# initialize matrces 
Tout = np.zeros((int(T/dt), 1))
Tout[0] = Tin

# Column matrices
Tn = np.zeros((int(T/dt), 1))
deltaTout = np.zeros((int(T/dt), 1))
deltamdot_list = np.zeros((int(T/dt), 1))
mdot_list = np.zeros((int(T/dt), 1))


# not ethat mdot is defined at each timestep by functions 
# so no need to initialize it 

for tk in range(int(T/dt) - 1):
    # compute deltaTout 
    deltaTout = compute_deltaTout(Tout, tk) # get the column matrix of deltaTout updated at each timestep 
    # compute deltamdot
    deltamdot = compute_deltamdot(proportional_gains, deltaTout, Tout, tk) # get deltamdot (scalar)
    deltamdot_list[tk] = deltamdot # add it to a list for visualization purposes 
    # compute mdot
    mdot = compute_mdot(mdot_bar, deltamdot)
    # mdot = np.clip(mdot, 0.1, 1)
    mdot_list[tk] = mdot # add it to a list for visualization purposes

    # get Tout !
    # we're feeding the function with just the column matri x of Tout, the current mdot
    # and the other relevant info
    # get out Tout at the next step Tout[k+1] (you dont actually get it out, 
    # it just changes the array)
    Tout = forward_euler(Tout, mdot, Tout_bar, mdot_bar, Tin, T, dt, tk, cp_bed, Ci)


print("Tout is", Tout)
print("deltaTout is", deltaTout)
print("deltamdot is", deltamdot)
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

# Plot detaTout vs time
plt.plot(t_physics, deltaTout)
plt.xlabel("time(s) Label")
plt.ylabel("deltaTout (C) Label")
plt.title("deltaTout vs time for Toutbar=700")
plt.show()

# Plot mdot vs time
plt.plot(t_physics, mdot_list)
plt.xlabel("time(s) Label")
plt.ylabel("mdot (C) Label")
plt.title("mdot vs time for Toutbar=700")
plt.show()


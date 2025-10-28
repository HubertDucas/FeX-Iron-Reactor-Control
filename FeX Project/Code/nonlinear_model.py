import numpy as np 
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp 

# Constants 
cp_bed = 450 
Ci = 450 

# Time
dt = 1e-1
t = np.arange(0, 50, dt)
steps = t.shape

# Equilibrium points
Qdot_gen_i = 100000 #arbitrary value 
Tin = 50
Tout_bar = 700
mdot_bar = Qdot_gen_i/(cp_bed * (Tout_bar - Tin))
# print(mdot_bar)

# Example mdot(t) input 
mdot = 10 * np.sin(t)
# print(mdot - mdot_bar)

# Non-linear ODE (RK4)
def RK4(mdot_scalar):
    """
    This linearized model is a simple implementation
    of equation 22. 

    Input: whatever mdot function you want (e.g.: mdot(t) = 10sin(t))
    Output: Tout(t), deltaTout(t) + plots (Tout vs t, Tout vs mdot, deltaTout vs deltamdot) 
    """
    Tout = np.array((t.shape[0], ))
    for tk in range(t.shape[0] - 1):
        K1 = (-cp_bed/Ci) * mdot[tk] * Tout[tk] + (
            cp_bed/Ci) * mdot[tk] * Tin + Qdot_gen_i/Ci
        
        K2 = (-cp_bed/Ci) * mdot[tk + (dt/2)] * (Tout[tk + (dt/2)] + (dt/2) * K1) + (
            cp_bed/Ci) * mdot[tk + (dt/2)] * Tin + Qdot_gen_i/Ci
        
        K3 = (-cp_bed/Ci) * mdot[tk + (dt/2)] * (Tout[tk + (dt/2)] + (dt/2) * K2) + (
            cp_bed/Ci) * mdot[tk + (dt/2)] * Tin + Qdot_gen_i/Ci
        
        K4 = (-cp_bed/Ci) * mdot[tk+1] * (Tout[tk+1] + dt * K3) + (
            cp_bed/Ci) * mdot[tk+1] * Tin + Qdot_gen_i/Ci
        
        Tout[tk+1] = Tout[tk] + (dt/6) * (K1 + 2 * K2 + 2 * K3 + K4)

    return Tout 

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


# Control loop
def feedback_loop(Tout):
    """
    The feedback loop is an implementation of Fig. 5. It takes in 
    Tout computed at the previous timestep by the function linearized_model(mdot)
    and gives the mdot that should be used at the next iteration

    Input: Tout (scalar)
    Output: mdot (scalar)
    """

    # compute the error
    noise = np.random.normal(0, 1) # 1C std 
    error = Tout_bar - (Tout + noise)

    # compute the reference mass flow rate 
    # (reference or error scaled by propoortoinal controller gain)
    kp = 1
    error_mdot = error * kp 

    mdot = mdot_bar + error_mdot
    return mdot



# ###################################

# Solver for linearized model with step definition
Tout_solution = np.zeros((t.shape[0]))
mdot_solution = np.zeros((t.shape[0]))
Tout_solution[0] = Tin
mdot_solution[0] = feedback_loop(Tin)

for k in range(t.shape[0] - 1):
    Tout_k = Tout_solution[k]
    mdot_k = mdot_solution[k]

    Tout_kp1 = RK4(mdot_scalar=mdot_k)
    Tout_solution[k+1] = Tout_kp1

    mdot_kp1 = feedback_loop(Tout_kp1)
    mdot_solution[k+1] = mdot_kp1

# ###################################

# plot the linearized Tout(t) as a function of t 
plt.plot(t, Tout_solution)
plt.xlabel("Time(s) Label")
plt.ylabel("Tout (C) Label")
plt.title("Tout vs time for Toutbar=700")
plt.show()

# plot the linearized mdot(t) as a function of t 
plt.plot(t, mdot_solution)
plt.xlabel("Time(s) Label")
plt.ylabel("Tout (C) Label")
plt.title("mdot vs time for Toutbar=700")
plt.show()
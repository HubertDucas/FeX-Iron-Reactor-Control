"""
In this code, the gain scheduling approach is implemented. 5 effective lengths 
are defined corresponding to which of the four thermocouples the reactive bed 
is closest to. The goal is to do loopshaping on each of these 

Hubert Ducas
Last changed: 12/28/2025
"""

# Packages
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import control
import unc_bound
import siso_rob_perf as srp 

# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# Golden ratio
gr = (1 + np.sqrt(5)) / 2

# Figure height
height = 4.25

##### Constants #####
cp_bed = 450 # J/kgK
cp_ss = 500 # J/kgK
Ci_nom = 10000 #75000 #450 # J/K
rho_bed = 6000 # kg/m^3 
rho_ss = 8000 # kg/m^3 

# dimensions 
# initial bed length
L = 0.3 # m 
# l*n segment length 
seg = L / 5
# bed radius 
r_bed = 0.05
# wall thickness 
t_wall = 0.01 # m 


# Time
dt = 1e-1
t_start = 0 
t_end = 50
t = np.arange(t_start, t_end, dt)
steps = t.shape

# Equilibrium points
Qdot_gen_i_init = 100e3 # J (100 kJ/s) (works for 10kJ to 100kJ)
Tin = 25 # deg C
Tout_bar = 700 # deg C
mdot_bar = Qdot_gen_i_init/(cp_bed * (Tout_bar - Tin)) # kg/s
Tout_init = 800

# Noise
np.random.seed(123321) 
noise = np.random.normal(0, 1, t.shape[0]) * 1


##### Functions of effective-length #####

# time-varying effective-length 
# (remember that the goal is to have a data-driven model, 
# so we don't really care about the actual heat diffusion PDE
# of the bed shrinkage)
def lindex_func(t): 
    """
    li is the segment index l1 is the longest segment (initial 
    reactive bed length), l5 is the shortest segment (final
    reactive bed length)
    """
    # length of the reactive bed as a function of time
    l = L - 0.005 * t # linear shrinkage rate of 1mm/s
    # segments in decreasing length order 
    if (l <= 5 * seg) and (l > 4 * seg):
        lindex = 1 
    elif (l <= 4 * seg) and (l > 3 * seg): 
        lindex = 2 
    elif (l <= 3 * seg) and (l > 2 * seg): 
        lindex = 3
    elif (l <= 2 * seg) and (l > 1 * seg): 
        lindex = 4
    else: 
        lindex = 5
    
    return lindex


# effective-length-varying thermal mass
def Ci_func(li):
    """
    Ci(li) = V_bed(li) * rho_bed * cp_bed + V_tube * rho_ss * cp_ss
    then compute Pi at each time step inside the closed-loop
    """
    li = li * seg
    Ci = (np.pi * (r_bed**2) * li * rho_bed * cp_bed) + (
                        np.pi * ((r_bed + t_wall)**2 - r_bed**2) * L * rho_ss * cp_ss)
    
    return Ci

# effective-length-varying plant
def Pi_func(li):
    """
    Pi(Ci) = (cp_bed * (Tin - Tout_bar)) / (Ci * s + cp_bed * mdot_bar)
    not super useful since Pi is computed directly inside the closed-loop
    """
    Ci = Ci_func(li)
    Pi = ((cp_bed * (Tin - Tout_bar)) / (Ci * s + cp_bed * mdot_bar)) #* (22/0.5) # normalizaiton constant
    return Pi

# effective-length-varying heat generation
def Qi_func(li):
    """
    Heat generation assumed constant for each bed length 
    """
    Qi = li * (Qdot_gen_i_init / 5)
    return Qi

# effective-length-varying equilibrium mass-flow rate
def mdot_bar_func(li): 
    Qdot_gen_i = Qi_func(li)
    mdot_bar = Qdot_gen_i/(cp_bed * (Tout_bar - Tin)) # kg/s
    return mdot_bar


##### Gain scheduling #####

# low-pass filter 
tau_d = 30
s = control.tf('s')
a_d = 1/tau_d

# Gain scheduling function
def gain_scheduling(li):
    """
    From the effective length of the reactive bed, 
    we know which set of gains to use. The gains satisfy the required 
    inequalities
    """
    # 
    mdot_bar_i = mdot_bar_func(li)
    Ci = Ci_func(li)
    # gains
    ki = -0.00001 # ki < 0 
    kp = (mdot_bar_i / (Tout_bar - Tin) - ki * tau_d) - 0.01 #-0.1 to satisfy the inequality
    kd = ((Ci + cp_bed * mdot_bar_i * tau_d) / (cp_bed * (Tout_bar - Tin)) - kp * tau_d) - 0.2 # -0.2 to satisfy the inequality

    return ki, kp, kd

# controller function
def C_func(li, disable=False): # disable= no controller 
    ki, kp, kd = gain_scheduling(li)
    C = kp + (ki / s) + kd * (s / (tau_d * s + 1)) 
    C = C #* 0.2 # added gain to satisfy design specs 

    # turn off controller (need this because sim is expecting 
    # a controller with a specific number of states)
    if disable: 
        C = 1e-12 + (1e-12 / s) + 1e-12 * (s / (tau_d * s + 1))

    # loopshaped controller
    # C = (1.07e4 * s + 362.1) / (-1.336e7 * s)
    return C



# plant and controller
#(right now the plant is FIXED, need to change that at some point to plant func of Ci)
# nevermind, actually the plant moves wit Ci, which changes with time so all is good!
P_init = (cp_bed * (Tin - Tout_bar)) / (Ci_nom * s + cp_bed * mdot_bar) # this is just for the dimensions actually 
C_init = C_func(5)


# loopshaping plots for each length
for i in range(1, 6):
    li = (6 - i) * seg

    Ci = Ci_func(li)
    Pi = Pi_func(li)
    C  = C_func(li)

    # Nominal plant (if needed)
    P_nom = Pi
    print("P_nom = ", P_nom)


##### Simulation #####
# simulation function
def simulate(t, Tout_bar, Tout_init, noise):
    """
    Plant and controller are linearized, put in big xdot=Ax+Bu problem
    Solve using RK45. Need this function because we want to pass noise at each time-step. 
    Note that P_nom does not even need to be used. 
    """

    # time needs to be redefined as a new variable for solve_ivp
    time = t 

    # P_nom and controller in state-space form, just to get dimensions (initial)
    P_init_ss = control.tf2ss(P_init) 
    C_init_ss = control.tf2ss(C_init)
    n_x_P = P_init_ss.A.shape[0]
    n_x_C = C_init_ss.A.shape[0]
    # n_x_C = np.shape(C_ss.A)[0]

    # ICs for plant and controller
    x_P_IC = np.array([Tout_init - Tout_bar])
    # x_P_IC = np.zeros((n_x_P, 1))
    x_C_IC = np.zeros((n_x_C, 1)) # integral and derivative "states" are 0, initially 

    # Set up closed-loop ICs.
    x_cl_IC = np.block([[x_P_IC], [x_C_IC]]).ravel()
    # print(x_cl_IC)

    def closed_loop(t, x):

        # li-variying parameters
        lindex = lindex_func(t) 
        li = 6 - lindex
        # print(li)
        mdot_bar_i = mdot_bar_func(li)
        mdot_bar_i = np.array([mdot_bar_i])
        Ci = Ci_func(li)
        C = C_func(li)
        C_ss = control.tf2ss(C)   
        Qi = Qi_func(li)
        # print(Qi)  

        # plant matrices at current time, manually
        # A_p = np.array([[-mdot_bar_i * cp_bed / Ci]])
        # B_p = np.array([[ cp_bed * (Tin - Tout_bar) / Ci]])
        # C_p = np.array([[1]])
        # D_p = np.array([[0]])

        # P_ss = control.ss(A_p, B_p, C_p, D_p)
        # P_tf = control.ss2tf(P_ss)
        Pi = Pi_func(li)
        P_ss = control.tf2ss(Pi)
        A_p, B_p, C_p, D_p = P_ss.A, P_ss.B, P_ss.C, P_ss.D
        # print(P_tf)

        # split states 
        x_P = x[:n_x_P].reshape((-1, 1))
        x_C = x[n_x_P:].reshape((-1, 1))

        # reference & noise
        r_now = Tout_bar
        n_now = np.interp(t, time, noise).reshape((1, 1))

        # plant output
        delta_z = C_p @ x_P
        y = Tout_bar + delta_z + n_now

        # error
        error = (r_now - y)

        # controller
        delta_u = (C_ss.C @ x_C + C_ss.D @ error)

        # plant dynamics # THIS COULD BE THE LINE WHERE THERE IS AN ERROR
        dot_x_sys = A_p @ x_P + B_p @ delta_u #+ B_p @ mdot_bar_i

        # controller dynamics
        dot_x_ctrl = C_ss.A @ x_C + C_ss.B @ error

        # combine
        x_dot = np.vstack((dot_x_sys, dot_x_ctrl)).ravel()
        return x_dot

    # find time-domain response by integrating the ODE 
    sol = integrate.solve_ivp(
        closed_loop,
        (t_start, t_end),
        x_cl_IC,
        t_eval=t,
        rtol=1e-8,
        atol=1e-6,
        method='RK45',
    )

    t_sol = sol.t
    N = t_sol.size

    # extract states 
    sol_x = sol.y 
    x_P = sol_x[:n_x_P, :]
    # print(x_P)
    x_C = sol_x[n_x_P:, :]

    # compute plant output, control signal, and ideal error
    # y = np.zeros(t.shape[0],)
    # u = np.zeros(t.shape[0],)
    # e = np.zeros(t.shape[0],)

    y = np.zeros(N)
    u = np.zeros(N)
    e = np.zeros(N)

    for i in range(time.size):
        # current time
        ti = t[i]

        # effective-length-dependant parameters
        li = lindex_func(ti)
        C = C_func(li)
        C_ss = control.tf2ss(C)
        
        mdot_bar_i = mdot_bar_func(li)

        # reference at current time 
        r_now = np.array([Tout_bar])

        # noise at current time
        n_now = np.interp(t[i], time, noise).reshape((1, 1))

        # current reference points (change with Ci(l))
        z_bar = Tout_bar 
        u_bar = mdot_bar_i # changed that and it made the mdot behave as it should (didn't upload to github, at the right time)

        # Plant output, with noise
        # delta_z = P_ss.C @ x_P[:, [i]]
        delta_z = x_P[:, [i]] # P_ss.C = 1 here
        # print("C", P_ss.C)
        # print("x_P", x_P[:, [i]])
        y[i] = (z_bar + delta_z + n_now).ravel()[0]
        # print("Tout right now", y[i])

        # Compute error
        error = (r_now - (z_bar + delta_z + n_now))
        # e[i] = error.ravel()[0]
        e[i] = error
        # print("error", e[i])

        # Compute control
        delta_u = (C_ss.C @ x_C[:, [i]] + C_ss.D @ error)
        u[i] = (u_bar + delta_u).ravel()[0]

    return sol.t, y, u, e 

# Run simulation
t_sol, y, u, e = simulate(t, Tout_bar, Tout_init, noise)

Tout = y 
mdot = u 


# mdot_bar_t = np.array([mdot_bar_func(ti) for ti in t_sol])


# plot styles
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# plots
fig, ax = plt.subplots(2, 1, figsize=(height * gr, height))

# Temperature plot
ax[0].plot(t_sol, Tout, label=r'$T_{out}(t)$', linewidth=2)
ax[0].axhline(y=Tout_bar, linestyle='--', label=r'$\overline{T_{out}}$', linewidth=1.5, color='r')
ax[0].set_xlabel(r"$Time\ (s)$")
ax[0].set_ylabel(r"$T_{out}\ (^\circ C)$")
ax[0].set_title('Closed-loop temp response')
ax[0].legend()

# Mass flow rate plot
ax[1].plot(t_sol, mdot, label=r'$\dot{m}(t)$', linewidth=2)
ax[1].axhline(y=mdot_bar, linestyle='--', label=r'$\overline{\dot{m}}$', linewidth=1.5, color='r')
# ax[1].plot(t_sol, mdot_bar_t, linestyle='--', linewidth=1.5, color='r',label=r'$\overline{\dot{m}}(t)$')
ax[1].set_xlabel(r"$Time\ (s)$")
ax[1].set_ylabel(r"$\dot{m}\ (kg/s)$")
ax[1].set_title('Controlled mass-flow rate')
ax[1].legend()

fig.tight_layout()
plt.show()





        



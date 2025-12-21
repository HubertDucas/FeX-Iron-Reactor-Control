"""
In this code, the full matrix implementation of the plant and the 
controller is done. ss_form_PID_plant.py did not have a varying Ci(l), 
this code does. Loopshaping is done to design the best controller.

Hubert Ducas
Last changed: 11/22/2025
"""

# Packages
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import control

# Golden ratio
gr = (1 + np.sqrt(5)) / 2

# Figure height
height = 4.25

# Constants 
cp_bed = 450 # J/kgK
cp_ss = 500 # J/kgK
Ci = 75000 #450 # J/K
rho_bed = 6000 # kg/m^3 
rho_ss = 8000 # kg/m^3 

# redefine changing Ci(l)
# dimensions 
# initial bed length
L = 0.3 # m 
# bed radius 
r_bed = 0.05
# wall thickness 
t_wall = 0.01 # m 

# changing thermal mass
def Ci(t):
    """
    Ci(li) = V_bed(li) * rho_bed * cp_bed + V_tube * rho_ss * cp_ss
    then compute Pi at each time step and get the Pi array
    """
    # l = 0.3 - 0.01 * t (1 cm/s)
    li = L - 0.01 * t
    Ci = (np.pi * (r_bed**2) * li * rho_bed * cp_bed) + (
                        np.pi * ((r_bed + t_wall)**2 - r_bed**2) * L * rho_ss * cp_ss)
    
    return Ci


# Time
dt = 1e-1
t_start = 0 
t_end = 50 
t = np.arange(t_start, t_end, dt)
steps = t.shape

# Equilibrium points
Qdot_gen_i = 100e3 # J (100 kJ/s)
Tin = 25 # deg C
Tout_bar = 700 # deg C
mdot_bar = Qdot_gen_i/(cp_bed * (Tout_bar - Tin)) # kg/s
Tout_init = 750

# Noise
np.random.seed(123321) 
noise = np.random.normal(0, 1, t.shape[0]) * 1


# low-pass filter 
tau_d = 10
s = control.tf('s')
a_d = 1/tau_d

# Gains
# kp = 0.01
# ki = -0.001 #0.08
# kd = 0.0001 # 0.000001 

# desinged gains (added 1 just to satisfy the inequality)
kp = 1 - (mdot_bar / (Tin - Tout_bar))
ki = 0.01
kd = 1 - ((Ci + cp_bed * mdot_bar * tau_d) / (cp_bed * (Tin - Tout_bar))) - kp * tau_d

print("k_p:", kp)
print("k_d:", kd)

# plant and controller
P = (cp_bed * (Tin - Tout_bar)) / (Ci * s + cp_bed * mdot_bar) # input and output in deviations
C = kp + (ki / s) + kd * (s / (tau_d * s + 1)) # controller controls deviations

# print(cp_bed * (Tin - Tout_bar))
# print(np.array(P.num))
# print(np.array(P.den))

# P_ss = control.tf(P.num, P.den)
# print(P_ss)
P_ss = control.tf2ss(P)
C_ss = control.tf2ss(C)
print(P)
print(P_ss)
print(C_ss)

# C_mat = control.tf2ss(P)
# print("C_mat", C_mat.C)

# print("The open-loop zeros are", control.poles(P))


# simulation function
def simulate(P, C, t, Tout_bar, mdot_bar, Tout_init, noise):
    """
    Plant and controller are linearized, put in big xdot=Ax+Bu problem
    Solve using RK45. Need this function because we want to pass noise at each time-step
    """

    # time needs to be redefined as a new variable for solve_ivp
    time = t 
    
    # plant in state-space form 
    # P_ss = control.tf2ss(P)
    A_p = np.array([[-mdot_bar * cp_bed / Ci]])
    B_p = np.array([[cp_bed * (Tin - Tout_bar) / Ci]])
    C_p = np.array([[1]])
    D_p = np.array([[0]])

    P_ss = control.ss(A_p, B_p, C_p, D_p)
    n_x_P = np.shape(P_ss.A)[0]
    # print(P_ss.A)
    # print(P_ss.C)

    # controller in state-space form 
    C_ss = control.tf2ss(C)
    n_x_C = np.shape(C_ss.A)[0]

    #Plant state-space form.
    # P_ss = control.tf2ss(np.array(P.num).ravel(), np.array(P.den).ravel())
    n_x_P = np.shape(P_ss.A)[0]
    
    # # Control state-space form.
    # C_ss = control.tf2ss(np.array(C.num).ravel(), np.array(C.den).ravel())
    # n_x_C = np.shape(C_ss.A)[0]

    """
    manual state-space realization
    A_p = np.array([[-mdot_bar * cp_bed / Ci]])
    B_p = np.array([[cp_bed * (Tin - Tout_bar) / Ci]])
    C_p = np.array([[1]])
    D_p = np.array([[0]])


    # controller 
    A_c = np.array([[0, 0],
                    [0, -a_d]])
    B_c = np.array([[1],
                    [a_d]])
    C_c = np.array([[ki, kd]])
    D_c = np.array([[kp]])


    P_ss = control.ss(A_p, B_p, C_p, D_p)
    C_ss = control.ss(A_c, B_c, C_c, D_c)
    n_x_P = np.shape(P_ss.A)[0]
    n_x_C = np.shape(C_ss.A)[0]
    """
    
    # print matrices
    # print("Ac:", C_ss.A)
    # print("Bc:", C_ss.B)
    # print("Cc:", C_ss.C)
    # print("Dc:", C_ss.D)

    # ICs for plant and controller
    x_P_IC = np.array([-(Tout_bar - Tout_init)])
    # x_P_IC = np.zeros((n_x_P, 1))
    x_C_IC = np.zeros((n_x_C, 1)) # integral and derivative "states" are 0, initially 

    # Set up closed-loop ICs.
    x_cl_IC = np.block([[x_P_IC], [x_C_IC]]).ravel()
    # print(x_cl_IC)

    # Define closed-loop system. This will be passed to solve_ivp
    def closed_loop(t, x):
        """
        Closed-loop system
        t: present time
        x: present state 
        """

        # reference at current time (constant)
        r_now = np.array([Tout_bar])

        # noise at current time 
        n_now = np.interp(t, time, noise).reshape((1, 1))
        
        # split state
        x_P = (x[:n_x_P]).reshape((-1, 1)) # plant output states (deviation)
        x_C = (x[n_x_P:]).reshape((-1, 1)) # controller states 

        # print(x_P)

        # equilibrium values (will change eventually as Qdotgen_i changes)
        z_bar = np.array([Tout_bar])
        u_bar = np.array([mdot_bar])

        # plant output with noise
        delta_z = P_ss.C @ x_P
        y = z_bar + delta_z + n_now
        # print(P_ss.D)

        # compute error 
        error = (r_now - y)

        # compute controller output (control signal)
        delta_u = -(C_ss.C @ x_C + C_ss.D @ error) # only thing that works is when controller output is neg
        u = u_bar + delta_u

        # advance system state 
        dot_x_sys = P_ss.A @ x_P + P_ss.B @ delta_u #- P_ss.B @ u_bar 

        # advance controller state
        dot_x_ctrl = C_ss.A @ x_C + C_ss.B @ error 

        # # Controller output
        # delta_u = C_ss.C @ x_C + C_ss.D @ x_P  
        # u = u_bar + delta_u 

        # # advance the system state
        # dot_x_sys = P_ss.A @ x_P + P_ss.B @ C_ss.C @ x_C + P_ss.B @ C_ss.D @ x_P

        # # advance the controller state
        # dot_x_ctrl = C_ss.B @ x_P + C_ss.A @ x_C

        # concatenate state derivatives 
        x_dot = np.block([[dot_x_sys], 
                          [dot_x_ctrl]]).ravel()
        
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

    # extract states 
    sol_x = sol.y 
    x_P = sol_x[:n_x_P, :]
    # print(x_P)
    x_C = sol_x[n_x_P:, :]

    # compute plant output, control signal, and ideal error
    y = np.zeros(t.shape[0],)
    u = np.zeros(t.shape[0],)
    e = np.zeros(t.shape[0],)

    # set initial output temp
    # y[0] = Tout_init # 900 deg C

    for i in range(0, time.size):

        # reference at current time 
        r_now = np.array([Tout_bar])

        # noise at current time
        n_now = np.interp(t[i], time, noise).reshape((1, 1))

        # current reference points (change with Ci(li))
        z_bar = Tout_bar 
        u_bar = mdot_bar

        # Plant output, with noise
        delta_z = P_ss.C @ x_P[:, [i]]
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
        delta_u = -(C_ss.C @ x_C[:, [i]] + C_ss.D @ error)
        u[i] = (u_bar + delta_u).ravel()[0]

    return y, u, e 

# Run simulation
y, u, e = simulate(P, C, t, Tout_bar, mdot_bar, Tout_init, noise)

Tout = y 
mdot = u 





# plot styles
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# plots
fig, ax = plt.subplots(2, 1, figsize=(height * gr, height))

# Temperature plot
ax[0].plot(t, Tout, label=r'$T_{out}(t)$', linewidth=2)
ax[0].axhline(y=Tout_bar, linestyle='--', label=r'$\overline{T_{out}}$', linewidth=1.5, color='r')
ax[0].set_xlabel(r"$Time\ (s)$")
ax[0].set_ylabel(r"$T_{out}\ (^\circ C)$")
ax[0].set_title('Closed-loop temp response')
ax[0].legend()

# Mass flow rate plot
ax[1].plot(t, mdot, label=r'$m_{dot}(t)$', linewidth=2)
ax[1].axhline(y=mdot_bar, linestyle='--', label=r'$\overline{\dot{m}}$', linewidth=1.5, color='r')
ax[1].set_xlabel(r"$Time\ (s)$")
ax[1].set_ylabel(r"$\dot{m}\ (kg/s)$")
ax[1].set_title('Controlled mass-flow rate')
ax[1].legend()

fig.tight_layout()
plt.show()




        



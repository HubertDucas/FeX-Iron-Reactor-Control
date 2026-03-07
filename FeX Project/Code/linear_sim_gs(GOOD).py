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
from scipy import signal

# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

s = control.tf('s')

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
L = 0.15 # 0.3 # m 
# l*n segment length 
seg = L / 5
# bed radius 
r_bed = 0.008525# 0.0005
# wall thickness 
t_wall = 0.001 # 0.0001# m 

# Time
dt = 1e-1
t_start = 0 
t_end = 135
t = np.arange(t_start, t_end, dt)
steps = t.shape

# Equilibrium points
Qdot_gen_i_init = 100 # 100 W
Tin = 25 # deg C
Tout_bar = 700 # deg C
mdot_bar_init = Qdot_gen_i_init/(cp_bed * (Tout_bar - Tin)) # kg/s
# print(mdot_bar_init)
Tout_init = 800

kgs2slpm = 50000 # conversion kg/s to slpm

# Normalization constants
e_nor_r = 0.05  # 5% allowable error for command following
r_nor = 100  # degC, largest expected change in reference
n_nor = 0.5  # degC, largest expected change in noise
u_nor_r = 10  # slpm, max control signal for command following

# normalization 
deltaTout_max  =50 #degC
deltamdot_max = 25 /kgs2slpm 
# Noise
np.random.seed(123321) 
noise = np.random.normal(0, 0.005, t.shape[0]) * 1
# noise = noise * (n_nor / e_nor_r) # normalize



##### Functions of effective-length #####

# time-varying effective-length 
# (remember that the goal is to have a data-driven model, 
# so we don't really care about the actual heat diffusion PDE
# of the bed shrinkage)
def lindex_func(t): 
    """
    lindex is the segment index. l1 is the longest segment (initial 
    reactive bed length), l5 is the shortest segment (final
    reactive bed length)
    """
    if t >=75:
        # length of the reactive bed as a function of time
        l = L - 0.005 * (t - 75) # linear shrinkage rate of 1mm/s
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
    else:
        lindex = 1
    
    return lindex

# effective-length-varying thermal mass
def Ci_func(lindex):
    """
    li is the effective length for configuration i
    Ci(li) = V_bed(li) * rho_bed * cp_bed + V_tube * rho_ss * cp_ss
    then compute Pi at each time step inside the closed-loop
    """
    li = (6 - lindex) * seg
    Ci = (np.pi * (r_bed**2) * li * rho_bed * cp_bed) + (
                        np.pi * ((r_bed + t_wall)**2 - r_bed**2) * L * rho_ss * cp_ss)
    
    return Ci

# effective-length-varying heat generation
def Qi_func(lindex):
    """
    Heat generation assumed constant for each bed length 
    Here use lindex instead of li since we assume the heat gen 
    is const. for a given config. 
    """
    Qi = (6 - lindex) * (Qdot_gen_i_init / 5)
    return Qi

# effective-length-varying equilibrium mass-flow rate
def mdot_bar_func(lindex): 
    """
    Use lindex here again
    """
    Qdot_gen_i = Qi_func(lindex)
    mdot_bar = Qdot_gen_i/(cp_bed * (Tout_bar - Tin)) # kg/s
    return mdot_bar

# effective-length-varying plant
def Pi_func(lindex):
    """
    li is the effective length for configuration i
    Pi(Ci) = (cp_bed * (Tin - Tout_bar)) / (Ci * s + cp_bed * mdot_bar)
    not super useful since Pi is computed directly inside the closed-loop
    """
    # li = (6 - lindex) * seg
    Ci = Ci_func(lindex)
    mdot_bar_i = mdot_bar_func(lindex)
    Pi = ((cp_bed * (Tin - Tout_bar)) / (Ci * s + cp_bed * mdot_bar_i)) #* (22/0.5) # normalizaiton constant
    # Normalize the plant: P(s) = e_nor_r^(-1) * P_tilde(s) * u_nor_r
    # P_tilde(s) = e_nor_r * P_dev / u_nor_r (convert slpm to kg/s)
    # Pi = Pi * ((u_nor_r / kgs2slpm) / e_nor_r) # normalization
    return Pi

# print("Pi_init", Pi_func(1))

##### Gain scheduling #####

# low-pass filter 
tau_d = 30
s = control.tf('s')
a_d = 1/tau_d

# Gain scheduling function
def gain_scheduling(lindex):
    """
    Take lindex again
    From the effective length of the reactive bed, 
    we know which set of gains to use. The gains satisfy the required 
    inequalities
    """
    # 
    mdot_bar_i = mdot_bar_func(lindex)
    # li = (6 - lindex) * seg
    Ci = Ci_func(lindex)
    # gains
    ki = -0.0001 #+0.00001 # ki < 0 
    kp = (mdot_bar_i / (Tout_bar - Tin) - ki * tau_d)  - 0.01 #+ 0.0001 # to satisfy the inequality
    kd = ((Ci + cp_bed * mdot_bar_i * tau_d) / (cp_bed * (Tout_bar - Tin)) - kp * tau_d) - 0.05 # -0.2 to satisfy the inequality


    return ki, kp, kd


# def gain_scheduling(lindex):
#     """
#     Gain scheduling with Routh-Hurwitz constraint c2*c1 > c3*c0
#     """

#     mdot_bar_i = mdot_bar_func(lindex)
#     Ci = Ci_func(lindex)

#     DeltaT = Tin - Tout_bar   # negative
#     cp = cp_bed

#     ki_tmp = -1e-6

#     kp = (mdot_bar_i / (Tout_bar - Tin) - ki_tmp * tau_d) - 0.01
#     kd = ((Ci + cp_bed * mdot_bar_i * tau_d) / (cp_bed * (Tout_bar - Tin)) - kp * tau_d) - 0.2 # -0.2 to satisfy the inequality

#     # RH upper bound on ki 
#     A = (
#         Ci
#         + cp * mdot_bar_i * tau_d
#         + cp * DeltaT * (kp * tau_d + kd)
#     )

#     B = (
#         cp * mdot_bar_i
#         + cp * DeltaT * kp
#     )

#     D = cp * DeltaT * tau_d
#     E = Ci * tau_d * cp * DeltaT

#     ki_max = (A * B) / (E - A * D)   # this is negative

#     ki = 0.1*ki_max  - 0.005 

#     return ki, kp, kd


for i in range(1, 6):
    lindex = i
    # print(lindex)
    Ci = Ci_func(lindex)
    mdot_bar_i = mdot_bar_func(lindex)
    Qdot_gen_i = Qi_func(lindex)
    # print(Qdot_gen_i)
    # print(mdot_bar_i)
    # print(Ci)

    ki, kp, kd = gain_scheduling(lindex)

    # inequality related to c2c1 > c3c0
    if (Ci + cp_bed * mdot_bar_i * tau_d + 
        cp_bed * (Tin - Tout_bar) * (kp * tau_d + kd)) * (cp_bed * mdot_bar_i + 
                                                       cp_bed * (Tin - Tout_bar) * (kp + ki * tau_d)) > (Ci * tau_d) * (cp_bed * (Tin - Tout_bar) * ki): 
        print("yes")
    else: 
        print("no")

# controller function
def C_func(lindex, disable=False): # disable= no controller 
    ki, kp, kd = gain_scheduling(lindex)
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
# P_init = (cp_bed * (Tin - Tout_bar)) / (Ci_nom * s + cp_bed * mdot_bar_init) # this is just for the dimensions actually 
# C_init = C_func(5)












####### Get plants and controller for gain schedulign #######
# Here, get the plants, controllers and mdot_bar_i for each configuration, 
# This is for after 75s
Pi_list = []
C_list = []
mdot_bar_i_list = []
mdot_bar_dev = []

for i in range(6):
    lindex = i 
    Qdot_gen_i = Qi_func(lindex)
    Ci = Ci_func(lindex)
    Pi = Pi_func(lindex)
    C = C_func(lindex)
    mdot_bar_i = mdot_bar_func(lindex)

    Pi_list.append(Pi)
    C_list.append(C)
    mdot_bar_i_list.append(mdot_bar_i)

    if i > 1:
        mdot_bar_dev.append(mdot_bar_i - mdot_bar_i_list[i-1])

# print(mdot_bar_i_list)
# print(mdot_bar_dev)
# so the deviation (jump) from config to next is 
jump = mdot_bar_dev[0] # see next section to see where we're putting it in u_bar(t)
# print("jump", jump)
######### Get first 75 s of linear simulation #########
def reference():
    """
    Tout ref in deviation
    """
    n_t = t.shape[0]
    delta_Tout_ref = np.zeros(n_t)

    # Phase 1: ramp (0–30 s)
    t1_end = int(30 / dt) # end time of ramp
    delta_Tout_ref[:t1_end] = np.linspace(0, 20, t1_end)

    # Phase 2: hold (30–60 s)
    t2_end = int(60 / dt) # end time of hold
    delta_Tout_ref[t1_end:t2_end] = 20

    # Phase 3: return to zero (>=60 s)
    delta_Tout_ref[t2_end:] = 0

    # Low-pass filter (smooth)
    tau_ref = 2
    _, delta_Tout_ref = control.forced_response(1 / (tau_ref * s + 1), T=t, U=delta_Tout_ref)
    return delta_Tout_ref

ref = reference()
a = 3
_, r_tilde = control.forced_response(1 / (1 / a * s + 1), t, ref, 0)
r = r_tilde
# r = r * (r_nor / e_nor_r) # normalize 
# print("ref", ref)

P_init = Pi_list[0]
C_init = C_list[0]

# print(P_init)
# print(C_init)




######## Simulation ########
# define u_bar 
u_bar_list = np.zeros(t.size)
# mdot_bar_i_list_ubar = np.zeros(t.size)
# L, T, S, CS
P = P_init
C = C_init
T = control.feedback(P * C, 1, -1)
S = control.feedback(1, P * C, -1)
CS = control.minreal(C * S)  # Need this to compute u(t)
t_tr = t[:int(75/dt)]

# response for full time length
_, z_bar_deviations_tr = control.forced_response(T, t, r - noise)
_, u_bar_deviations_tr = control.forced_response(CS, t, r - noise)

# print(u_bar_deviations_tr)
# cut down time to transient part
u_bar_list[:int(75/dt)] = u_bar_deviations_tr[:int(75/dt)] # u_bar in deviations!!

"""
###### this is just to get an idea of what the ubar will look like past 75s
#### note that it is actually computed later, this is just for visualization if needed
u_bar_list[int(75/dt) :] = 0 # fill rest with 0 deviation

# For steady state, use absolute mdot_bar_i directly
for i in range(1, 6):
    u_bar_list[int(75/dt) + int(12/dt)*(i-1)] = jump # jump when we change configuration 

print("u_bar_list", u_bar_list)

# print("barrr", u_bar_list[int(75/dt):])
# tau_u = 2
# Gf = 1 / (tau_u * s + 1)

# Filter the portion after 75s only
t_post = t[int(75/dt):]
u_post = u_bar_list[int(75/dt):]

# filter
b = 5 # for now need to manually play with the filter to get the right dip size
_, u_post_filt = control.forced_response(1 / (1 / b * s + 1), t_post, u_post * 5) # 5 here was also added to make the step the right length

u_bar_list[int(75/dt):] = u_post_filt # this is now the input reference to follow u_bar after 75s
# print("ubar", u_bar_list)
# print("u_post_filt", u_post_filt)
#############
"""

# now control the forced response of the system past 75 s due to the jumps (dips in deltamdot)
idx_75 = int(75 / dt)
t_post = t[idx_75:]
# u_post = u_bar_list[idx_75:]


seg_len = int(12 / dt)
z_post = np.zeros_like(t_post)
u_post_75 = np.zeros_like(t_post)

# for i in range(1, 6):
#     k0 = (i-1) * seg_len
#     k1 = min(i * seg_len, len(t_post))
#     if k0 >= len(t_post):
#         break

#     t_seg = t_post[k0:k1]
#     u_seg = u_desired[k0:k1]

#     # Get plant and controller for this segment
#     P_i = Pi_func(i)
#     C_i = C_func(i)
#     CS_i = control.minreal(C_i * control.feedback(1, P_i, -1))  # u = CS_i * r
#     T_i = control.feedback(P_i * C_i, 1, -1)
#     print("P_i", P_i)
#     print("C_i", C_i)
#     print("CS_i", CS_i)
#     print("T_i", T_i)

#     # Invert controller to get reference that produces u_seg
#     # u_seg = CS_i * r_seg  -->  r_seg = u_seg / CS_i
#     # _, r_seg = control.forced_response(1/CS_i, t_seg, u_seg)

#     # Store reference
#     # r_post_75[k0:k1] = r_seg

#     # Simulate plant response with reference and noise
#     noise_seg = noise[idx_75 + k0: idx_75 + k1]
#     _, y_seg = control.forced_response(T_i, t_seg, 0 +noise_seg)
#     _, u_seg_out = control.forced_response(CS_i, t_seg, 0 + noise_seg)

#     # Store outputs
#     z_post[k0:k1] = y_seg
#     u_post_75[k0:k1] = u_seg_out



for i in range(1, 6):
    k0 = (i-1) * seg_len
    k1 = min(i * seg_len, len(t_post))
    if k0 >= len(t_post):
        break
    
    print("Ci = ", Ci_func(i))
    print("Qdotgeni = ", Qi_func(i))
    P_i = Pi_func(i)
    C_i = C_func(i)
    T_i = control.feedback(P_i * C_i, 1, -1)
    S_i = 1 - T_i 
    CS_i = C_i * S_i

    t_seg = t_post[k0:k1]

    # Simulate plant response with reference and noise
    noise_seg = noise[idx_75 + k0: idx_75 + k1]
    _, y_seg = control.forced_response(T_i, t_seg, 0 + noise_seg)
    # _, u_seg = control.forced_response(CS_i, t_seg, 0 )
    _, u_seg = control.forced_response(CS_i, t_seg, 0 + noise_seg)


    # Store outputs
    z_post[k0:k1] = y_seg
    u_post_75[k0:k1] = u_seg





# patchwork z_bar_list
z_tr = z_bar_deviations_tr[:idx_75]
z_bar_list = np.zeros_like(t)
z_bar_list[:idx_75] = z_tr
z_bar_list[idx_75:] = z_post

# z_bar_list = z_bar_list * (e_nor_r / e_nor_r) # un-normalize



# Split data into transient and post-75s
t_transient = t[:idx_75]
t_post_75 = t[idx_75:]

# ref_transient = ref[:idx_75]
# ref_post_75 = ref[idx_75:]

# r = r * (e_nor_r / e_nor_r) # un-normalize

r_transient = r[:idx_75]
r_post_75 = r[idx_75:]

z_transient = z_bar_list[:idx_75]
z_post_75 = z_bar_list[idx_75:]

# u_bar_list = u_bar_list * (e_nor_r / e_nor_r) # un-normalize
u_transient = u_bar_list[:idx_75]
# u_post_75 = u_bar_list[idx_75:]

# Convert to absolute values
Tout_transient = z_transient + Tout_bar
Tout_post_75 = z_post_75 + Tout_bar


# make mdot_post_75 list with equilibrium mass flow rates
mdot_post_75_eq = np.zeros(int((t_end - 75)/dt))
for i in range(1 ,6):
    # time window for this configuration 
    k0 = (i-1) * seg_len 
    k1 = min(i * seg_len, len(t_post)) 
    
    if k0 >= len(t_post): 
        break 

    mdot_post_75_eq[k0:k1] = mdot_bar_i_list[i-1]

# print("brap", mdot_post_75)
# print(mdot_post_75.size)
# print(u_post_75.size)

# print(mdot_bar_i_list)




# remember here u are deviations again
mdot_transient = (u_transient + mdot_bar_init) * kgs2slpm
mdot_post_75 = (u_post_75 + mdot_post_75_eq) * kgs2slpm


# Figure 1: Transient part (0-75s)
fig1, ax1 = plt.subplots(4, 1, figsize=(height * gr, height * 1.6), sharex=True)

# Temperature deviation
ax1[0].plot(t_transient, r_transient, '--', label=r'$\overline{\delta T_{\mathrm{out}}}(t)$', color='C3', linewidth=2)
ax1[0].plot(t_transient, z_transient, '-', label=r'$\delta T_{\mathrm{out}}(t)$', color='C0', linewidth=2)
ax1[0].set_ylabel(r'$\delta T_{\mathrm{out}}$ (°C)')
ax1[0].set_title('Temperature Deviation (0-75 s)')
ax1[0].legend()
ax1[0].grid(True)

# Absolute temperature
ax1[1].plot(t_transient, r_transient + Tout_bar, '--', label=r'$\overline{T_{\mathrm{out}}}(t)$', color='C3', linewidth=2)
ax1[1].plot(t_transient, Tout_transient, '-', label=r'$T_{\mathrm{out}}(t)$', color='C0', linewidth=2)
ax1[1].set_ylabel(r'$T_{\mathrm{out}}$ (°C)')
ax1[1].set_title('Absolute Temperature')
ax1[1].legend()
ax1[1].grid(True)

# Mass flow deviation
ax1[2].plot(t_transient, u_transient * kgs2slpm, '-', label=r'$\delta\dot{m}(t)$', color='C1', linewidth=2)
ax1[2].axhline(0, label=r'$\overline{\delta\dot{m}}(t)$', color='C0', linestyle='--', linewidth=2, alpha=0.7)
ax1[2].set_ylabel(r'$\delta\dot{m}$ (slpm)')
ax1[2].set_title('Mass Flow Deviation')
ax1[2].legend()
ax1[2].grid(True)

# Absolute mass flow
ax1[3].plot(t_transient, mdot_transient, '-', label=r'$\dot{m}(t)$', color='C1', linewidth=2)
ax1[3].axhline(mdot_bar_init , label=r'$\overline{\dot{m}}(t)$', color='C0', linestyle='--', linewidth=2, alpha=0.7)
ax1[3].set_ylabel(r'$\dot{m}$ (slpm)')
ax1[3].set_xlabel(r'Time (s)')
ax1[3].set_title('Absolute Mass Flow Rate')
ax1[3].legend()
ax1[3].grid(True)

fig1.tight_layout()





# Figure 2: Post-75s part (gain-scheduled)
# fig2, ax2 = plt.subplots(4, 1, figsize=(height * gr, height * 1.6), sharex=True)
fig2, ax2 = plt.subplots(3, 1, figsize=(height * gr, height * 1.6), sharex=True)


# Temperature deviation
ax2[0].plot(t_post_75, r_post_75, '--', label=r'$\overline{\delta T_{\mathrm{out}}}(t)$', color='C3', linewidth=2)
ax2[0].plot(t_post_75, z_post_75, '-', label=r'$\delta T_{\mathrm{out}}(t)$', color='C0', linewidth=2)
ax2[0].set_ylabel(r'$\delta T_{\mathrm{out}}$ (°C)')
ax2[0].set_title('Temperature Deviation (>75s)')
ax2[0].legend()
ax2[0].grid(True)

# # Absolute temperature
# ax2[1].plot(t_post_75, r_post_75 + Tout_bar, '--', label=r'$\overline{T_{\mathrm{out}}}(t)$', color='C3', linewidth=2)
# ax2[1].plot(t_post_75, Tout_post_75, '-', label=r'$T_{\mathrm{out}}(t)$', color='C0', linewidth=2)
# ax2[1].set_ylabel(r'$T_{\mathrm{out}}$ (°C)')
# ax2[1].set_title('Absolute Temperature')
# ax2[1].legend()
# ax2[1].grid(True)

# Mass flow deviation
ax2[1].plot(t_post_75, u_post_75 * kgs2slpm, '-', label=r'$\delta\dot{m}(t)$', color='C1', linewidth=2)
ax2[1].axhline(0, label=r'$\overline{\delta\dot{m}_i}(t)$', color='C0', linestyle='--', linewidth=2, alpha=0.7)
ax2[1].set_ylabel(r'$\delta\dot{m}$ (slpm)')
ax2[1].set_title('Mass Flow Deviation')
ax2[1].legend()
ax2[1].grid(True)

# Absolute mass flow
ax2[2].plot(t_post_75, mdot_post_75, '-', label=r'$\dot{m}(t)$', color='C1', linewidth=2)
ax2[2].plot(t_post_75, mdot_post_75_eq * kgs2slpm, label=r'$\overline{\dot{m}_i}(t)$', color='C0', linestyle='--', linewidth=2, alpha=0.7)
ax2[2].set_ylabel(r'$\dot{m}$ (slpm)')
ax2[2].set_xlabel(r'Time (s)')
ax2[2].set_title('Absolute Mass Flow Rate')
ax2[2].legend()
ax2[2].grid(True)

fig2.tight_layout()
plt.show()



# fig, ax = plt.subplots(3, 1, figsize=(height * gr, 1.4 * height), sharex=True)

# # Reference
# ax[0].plot(t, ref, linestyle='--', label=r'Raw ref $\delta T_{out,ref}$')
# ax[0].plot(t, r, linewidth=2, label=r'Filtered ref $r(t)$')
# ax[0].set_ylabel(r'$\delta T_{out}(t)\ (^\circ C)$')
# ax[0].set_title('Reference signal')
# ax[0].legend()

# # Total output (transient + gain-scheduled response)
# ax[1].plot(t, z_bar_list,linewidth=2,label=r'$\overline{z}(t)$')
# ax[1].set_ylabel(r'$\delta {T_{\mathrm{out}}}(t)\ (^\circ C)$')
# ax[1].set_title('Scheduled equilibrium output')
# ax[1].legend()

# # Total input (transient + jumps) 
# ax[2].plot(t, u_bar_list, linewidth=2,label=r'$\overline{u}(t)$')
# ax[2].set_ylabel(r'$\delta{\dot{m}}(t)\ (slpm)$')
# ax[2].set_xlabel(r'Time (s)')
# ax[2].set_title('Scheduled equilibrium mass-flow rate deviations')
# ax[2].legend()

# fig.tight_layout()
# plt.show()


# # Plot deviations
# fig1, ax1 = plt.subplots(3, 1, figsize=(height * gr, height * 1.3))

# ax1[0].plot(t, r, '--', label=r'$\delta T_{\mathrm{out,ref}}(t)$', color='C3', linewidth=2)
# ax1[0].plot(t, deltaTout, '-', label=r'$\delta T_{\mathrm{out}}(t)$', color='C0', linewidth=2)
# # ax1[0].axhline(0, color='', linestyle=':', linewidth=1, alpha=0.5)
# ax1[0].set_ylabel(r'$\delta T_{\mathrm{out}}$ (°C)')
# ax1[0].set_title('Temperature Deviation')
# ax1[0].legend()
# ax1[0].grid(True)

# ax1[1].plot(t, deltamdot, '-', label=r'$\delta\dot{m}(t)$', color='C1', linewidth=2)
# ax1[1].axhline(0, color='C0', linestyle='--', linewidth=2, alpha=0.7)
# ax1[1].set_ylabel(r'$\delta\dot{m}$ (slpm)')
# ax1[1].set_title('Mass Flow Deviation')
# ax1[1].legend()
# ax1[1].grid(True)

# ax1[2].plot(t,  (delta_Tout_ref - deltaTout),  '-', color='C2', linewidth=2)
# # ax1[2].axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)
# ax1[2].set_ylabel(r'Error (°C)')
# ax1[2].set_xlabel(r'Time (s)')
# ax1[2].set_title('Tracking Error')
# ax1[2].grid(True)

# fig1.tight_layout()

# # Plot absolute values
# fig2, ax2 = plt.subplots(2, 1, figsize=(height * gr, height))

# ax2[0].plot(t, r + Tout_bar, '--', label=r'$\overline{T_{\mathrm{out}}}(t)$', color='C3', linewidth=2)
# ax2[0].plot(t, Tout, '-', label=r'$T_{\mathrm{out}}(t)$', color='C0', linewidth=2)
# # ax2[0].axhline(Tout_bar, color='C3', linestyle='--', linewidth=2, alpha=0.7)
# ax2[0].set_ylabel(r'$T_{\mathrm{out}}$ (°C)')
# ax2[0].set_title('Absolute Temperature')
# ax2[0].legend()
# ax2[0].grid(True)

# ax2[1].plot(t, mdot, '-', label=r'$\dot{m}(t)$', color='C1', linewidth=2)
# ax2[1].axhline(mdot_bar_init, color='C0', linestyle='--', linewidth=2, alpha=0.7)
# ax2[1].set_ylabel(r'$\dot{m}$ (slpm)')
# ax2[1].set_xlabel(r'Time (s)')
# ax2[1].set_title('Absolute Mass Flow Rate')
# ax2[1].legend()
# ax2[1].grid(True)

# fig2.tight_layout()
# plt.show()

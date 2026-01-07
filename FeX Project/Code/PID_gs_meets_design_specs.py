"""
In this code, the gain scheduling approach is used to do loopshaping on 
each of the "snapshot plants". 

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
Qdot_gen_i_init = 100e3 # J (100 kJ/s)
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
        li = 1 
    elif (l <= 4 * seg) and (l > 3 * seg): 
        li = 2 
    elif (l <= 3 * seg) and (l > 2 * seg): 
        li = 3
    elif (l <= 2 * seg) and (l > 1 * seg): 
        li = 4
    else: 
        li = 5
    
    return li


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
    """
    Ci = Ci_func(li)
    Pi = ((cp_bed * (Tin - Tout_bar)) / (Ci * s + cp_bed * mdot_bar)) * (22/0.5) # normalizaiton constant
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
    we know which set of gains to use. The gains satisfy the inequalities
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
def C_func(li, disable=False): 
    ki, kp, kd = gain_scheduling(li)
    C = kp + (ki / s) + kd * (s / (tau_d * s + 1)) 
    C = C #* 0.2 #0.55 # added gain to satisfy design specs 

    # turn off controller (need this because sim is expecting 
    # a controller with a specific number of states)
    if disable: 
        C = 1e-12 + (1e-12 / s) + 1e-12 * (s / (tau_d * s + 1))
    return C


# plant and controller
#(right now the plant is FIXED, need to change that at some point to plant func of Ci)
# nevermind, actually the plant moves wit Ci, which changes with time so all is good!
P_init = (cp_bed * (Tin - Tout_bar)) / (Ci_nom * s + cp_bed * mdot_bar) # this is just for the dimensions actually 
C_init = C_func(5)


##### Loopshaping #####
# Common parameters

# Conversion
rps2Hz = lambda w: w / 2 / np.pi
Hz2rps = lambda w: w * 2 * np.pi

# Golden ratio
gr = (1 + np.sqrt(5)) / 2

# Figure height
height = 4.25

# Laplace variable
s = control.tf('s')

# Frequencies for Bode plot, in rad/s
w_shared_low, w_shared_high, N_w = np.log10(Hz2rps(10**(-3))), np.log10(Hz2rps(10**(1))), 5000
w_shared = np.logspace(w_shared_low, w_shared_high, N_w)

# Frequencies for Nyquist plot, in rad/s
w_shared_low_2, w_shared_high_2, N_w_2 = np.log10(Hz2rps(10**(-3))), np.log10(Hz2rps(10**(1))), 5000
w_shared_2 = np.logspace(w_shared_low_2, w_shared_high_2, N_w_2)

# Golden ratio
gr = (1 + np.sqrt(5)) / 2

# Figure height
height = 4.25

# 1. l1 = 0.30m 
# 2. l2 = 0.24m
# 3. l3 = 0.18m
# 4. l4 = 0.12m
# 5. l5 = 0.06m

#"""
# loopshaping plots for each length
for i in range(1, 6):
    li = (6 - i) * seg

    Ci = Ci_func(li)
    Pi = Pi_func(li)
    C  = C_func(li)

    # Nominal plant (if needed)
    P_nom = Pi
    print("P_nom = ", P_nom)
    print("The controller is", C)

    # Extract nominal coefficients
    num_nom = P_nom.num[0][0]  
    den_nom = P_nom.den[0][0]  
    print("den_nom", den_nom) 

    # Off-nominal plants
    P_off_nom = []

    # 5% uncertainty
    var_factors = [0.95, 1, 1.05]  # [-5%, 0%, +5%]

    # generate all combinations
    for num_var in var_factors:
        for den1_var in var_factors:
            for den2_var in var_factors:

                # Apply variations
                num_varied = [num_nom[0] * num_var]

                den_varied = [
                    den_nom[0] * den1_var,                  # 1
                    den_nom[1],
                ]

                # Create off-nominal plant
                P_var = control.tf(num_varied, den_varied)

                P_off_nom.append(P_var)

    # # normalize the plant 
    # k_nom = num_nom[0] / den_nom[0]  
    # a_nom = den_nom[1] / den_nom[0]   

    # # generate all combinations (9 off-nom plants)
    # for k_var in var_factors:
    #     for a_var in var_factors:

    #         num_varied = [k_nom * k_var]
    #         den_varied = [1.0, a_nom * a_var]

    #         # Create off-nominal plant
    #         P_var = control.tf(num_varied, den_varied)
    #         P_off_nom.append(P_var)

    print("P_offnom = ", P_off_nom)
    # compute residuals
    N = len(P_off_nom)
    R = unc_bound.residuals(P_nom, P_off_nom)
     
    # bode plot
    w_shared = np.logspace(-3, 1, N_w) * 2 * np.pi

    # Compute magnitude part of R(s) in both dB and in absolute units
    mag_max_dB, mag_max_abs = unc_bound.residual_max_mag(R, w_shared)

    # find the optimal nW2 (that results in the least error from unc_bound)
    error_list = []
    for nW2 in range(5):
        residuals = unc_bound.residuals(P_nom,P_off_nom)
        W2_try = (unc_bound.upperbound(omega=w_shared, upper_bound=mag_max_abs, degree=nW2)).minreal()
        mag_W2_abs_try, _, _ = control.frequency_response(W2_try, w_shared)
        error = unc_bound.residual_max_mag(residuals, w_shared=w_shared) - mag_W2_abs_try
        error_scalar = np.linalg.norm(error)
        error_list.append(error_scalar)

    # Find the nW2 value that gives the minimum error
    nW2_min = min(range(len(error_list)), key=lambda i: error_list[i])
    print(nW2_min)

    # Calculate optimal upper bound transfer function.
    W2 = (unc_bound.upperbound(omega=w_shared, upper_bound=mag_max_abs, degree=nW2_min)).minreal()
    print("The optimal weighting function W_2(s) is ", W2)

    # Compute magnitude part of W_2(s) in absolute units
    mag_W2_abs, _, _ = control.frequency_response(W2, w_shared)
    # Compute magnitude part of W_2(s) in dB
    mag_W2_dB = 20 * np.log10(mag_W2_abs)

    fig, ax = plt.subplots(2, 1)
    ax[0].set_xlabel(r'$f$ (Hz)')
    ax[0].set_ylabel(r'Magnitude (dB)')
    ax[1].set_xlabel(r'$f$ (Hz)')
    ax[1].set_ylabel(r'Magnitude (absolute)')
    for i in range(N):
        mag_abs, _, _ = control.frequency_response(R[i], w_shared)
        mag_dB = 20 * np.log10(mag_abs)
        # Magnitude plot (dB)
        ax[0].semilogx(w_shared/(2*np.pi), mag_dB, '--', color='C0', linewidth=1)
        # Magnitude plot (absolute).
        ax[1].semilogx(w_shared/(2*np.pi), mag_abs, '--', color='C0', linewidth=1)

    # Magnitude plot (dB).
    ax[0].semilogx(w_shared/(2*np.pi), mag_max_dB, '-', color='C4', label='upper bound')
    # Magnitude plot (absolute).
    ax[1].semilogx(w_shared/(2*np.pi), mag_max_abs, '-', color='C4', label='upper bound')
    # Magnitude plot (dB).
    ax[0].semilogx(w_shared/(2*np.pi), mag_W2_dB, '-', color='seagreen', label='optimal bound')
    # Magnitude plot (absolute).
    ax[1].semilogx(w_shared/(2*np.pi), mag_W2_abs, '-', color='seagreen', label='optimal bound')
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    # fig.tight_layout()
    # fig.savefig(path.joinpath('1st_order_unstable_W2.pdf'))

    # Plot
    plt.show()  


    # loopshaping 
    # design specifications
    gamma_n, w_n_l = 0.75, Hz2rps(5) # 2 for 0.55
    gamma_r, w_r_h = 0.05, Hz2rps(0.005)
    gamma_u, w_u_l = 0.0682, Hz2rps(5) # 2 for 0.55
    gamma_d, w_d_h = 0.15, Hz2rps(0.005) 

    # Set up design specifications plot
    w_r = np.logspace(w_shared_low, np.log10(w_r_h), 100)
    w_d = np.logspace(w_shared_low, np.log10(w_d_h), 100)
    w_n = np.logspace(np.log10(w_n_l), w_shared_high, 100)
    w_u = np.logspace(np.log10(w_u_l), w_shared_high, 100)

    # In dB
    gamma_r_dB = 20 * np.log10(gamma_r) * np.ones(w_r.shape[0],)
    gamma_d_dB = 20 * np.log10(gamma_d) * np.ones(w_d.shape[0],)
    gamma_n_dB = 20 * np.log10(gamma_n) * np.ones(w_n.shape[0],)
    gamma_u_dB = 20 * np.log10(gamma_u) * np.ones(w_u.shape[0],)

    # Weight W_1(s) according to Zhou et al.
    k = 2
    epsilon = 10**(-40 / 20)
    M1 = 10**(40 / 20)
    # w_r_h_Hz = 0.012
    # w1 = Hz2rps(w_r_h_Hz + 0.2)
    w1 = 0.5

    W1 = ((s / M1**(1 / k) + w1) / (s + w1 * (epsilon)**(1 / k)))**k
    W1_inv = 1 / W1

    # # Plot both weights, W1 and W2 (and their inverses).
    # fig, ax = srp.bode_mag_W1_W2(W1, W2, w_d_h, w_n_l, w_shared, Hz = True)
    # fig.set_size_inches(height * gr, height, forward=True)
    # ax.legend(loc='upper right')
    # # fig.savefig('x.pdf')

    # fig, ax = srp.bode_mag_W1_inv_W2_inv(W1, W2, gamma_r, w_r_h, w_d_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared, Hz = True)
    # fig.set_size_inches(height * gr, height, forward=True)
    # ax.legend(loc='lower right')
    # # fig.savefig('x.pdf')

    # # Nyquist of open-loop plant without control
    # wmin, wmax, N_w_robust_nyq = np.log10(Hz2rps(10**(-4))), np.log10(Hz2rps(10**(4))), 1000
    # count, fig, ax = srp.robust_nyq(P_nom, P_off_nom, W2, wmin, wmax, N_w_robust_nyq)
    # fig.tight_layout()
    # # fig.savefig('x.pdf')

    fig_L, ax = srp.bode_mag_L(P_nom, C, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared, Hz = True)
    fig_L.set_size_inches(height * gr, height, forward=True)
    ax.legend(loc='lower left')

    fig_RP, ax = srp.bode_mag_rob_perf(P_nom, C, W1, W2, w_shared, Hz = True)
    fig_RP.set_size_inches(height * gr, height, forward=True)
    ax.legend(loc='lower left')

    fig_RP_RD, ax = srp.bode_mag_rob_perf_RD(P_nom, C, W1, W2, w_shared, Hz = True)
    fig_RP_RD.set_size_inches(height * gr, height, forward=True)
    ax.legend(loc='upper right')

    # fig_S_T, ax = srp.bode_mag_S_T(P_nom, C, gamma_r, w_r_h, w_d_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared, Hz = True)
    # ax.legend(loc='lower center')
    # # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # fig_S_T_W1_inv_W2_inv, ax = srp.bode_mag_S_T_W1_inv_W2_inv(P_nom, C, W1, W2, gamma_r, w_r_h, gamma_d, w_d_h, gamma_n, w_n_l, gamma_u, w_u_l, w_shared_low, w_shared_high, w_shared, Hz = True)
    # fig_S_T_W1_inv_W2_inv.set_size_inches(height * gr, height, forward=True)
    # ax.legend(loc='lower center')
    # # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # fig_L_P, ax = srp.bode_mag_L_P(P_nom, C, gamma_d, w_d_h, gamma_u, w_u_l, w_shared, Hz = True)
    # fig_L_P.set_size_inches(height * gr, height, forward=True)
    # ax.legend(loc='lower left')

    # fig_L_P_C, ax = srp.bode_mag_L_P_C(P_nom, C, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared, Hz = True)
    # fig_L_P_C.set_size_inches(height * gr, height, forward=True)
    # ax.legend(loc='lower left')

    # fig_margins, ax, gm, pm, vm, wpc, wgc, wvm = srp.bode_margins(P_nom, C, w_shared, Hz = True)
    # fig_margins.set_size_inches(height * gr, height, forward=True)
    # print(f'\nGain margin is', 20 * np.log10(gm),
    #     '(dB) at phase crossover frequency', wpc, '(rad/s)')
    # print(f'Phase margin is', pm, '(deg) at gain crossover frequency',
    #     wgc, '(rad/s)')
    # print(f'Vector margin is', vm, 'at frequency', wvm, '(rad/s)\n')

    fig_Gof4, ax = srp.bode_mag_Gof4(P_nom, C, gamma_r, w_r_h, gamma_d, w_d_h, gamma_n, w_n_l, gamma_u, w_u_l, w_shared_low, w_shared_high, w_shared, Hz = True)
    fig_Gof4.set_size_inches(height * gr, height, forward=True)

    # Nyquist
    fig_Nyquist, ax_Nyquist = plt.subplots()
    count, contour = control.nyquist_plot(control.minreal(P_nom * C),
                                        omega=w_shared_2,
                                        plot=True,
                                        return_contour=True)
    # ax_Nyquist.axis('equal')
    fig_Nyquist.tight_layout()

#"""








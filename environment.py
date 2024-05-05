import numpy as np
from scipy.special import erfcinv

epsilon_u = 10**-7

v_s_u = 2.2*10**-6
v_m_u = 6*10**-7
v_t_u = 2*10**-8

v_s_e = 2*10**-6
v_m_e = 4*10**-7
v_t_e = 2*10**-7

C_s = 10**-6
C_m = 5*10**-8
C_t = 2*10**-8

H_s = v_s_e - C_s  # 10**-5
H_m = v_m_e - C_m  # 10**-6
H_t = v_t_e - C_t  # 5*10**-7

N = 1

N_0 = 10**(-174/10)  # /1000

L_b = [1, 8, 8]

W_s = 20*10**6  # bandwidth of BSs
W_m = 100*10**6
W_t = 500*10**6

T_u = 0.5*10**-3  # Tau

lamda = 100*10**-6

l_M_1 = [25, 0]
l_M_2 = [-25, 0]

l_T_1 = [0, 10]
l_T_2 = [-10, -10]
l_T_3 = [-10, 10]

l_S = [-5, 0]

U = [0, 5]
E = [0, 5]

L_blk = 10
W_blk = 10
L_u = 256  # The length of URRLC packets

f = [2.4*10**9, 73*10**9, 0.8*10**12]  # the operation frequency of the BS

# ptx = [10, 1, 1]  # the transmission power
ptx_dBm = [40, 30, 30]  # the transmission power
ptx = 10**(np.array(ptx_dBm) / 10)
# h = [abs(randn(1,1)+ 1j*randn(1,1)) abs(randn(1,1)+ 1j*randn(1,1))]
h_s_f = np.random.randn(1) + 1j*np.random.randn(1)
h_m_f = np.random.randn(1) + 1j*np.random.randn(1)
h_s = np.abs(h_s_f)**2
h_m = np.abs(h_m_f)**2
# h = [h_s, h_m]
h = [np.exp(1), np.exp(1)]
# h = [0.1, 0.1]

alpha = [2.1, 2.5, 3.0]

G_s = [1, 1, 1]

theta_b = [2*np.arcsin(2.782/(L_b[0]*np.pi)),
           2*np.arcsin(2.782/(np.pi*L_b[1])),
           2*np.arcsin(2.782/(np.pi*L_b[2]))]

# G_b_max = (2*np.pi*L_b[1]**2*np.sin(1.5*np.pi/L_b[1]))/(theta_b[1]*L_b[1]**2*np.sin(1.5*np.pi/L_b[1]) +(2*np.pi-theta_b[1]))

G_b = np.array([1, 16.0159, 16.0159])

G_i = [1, 1, 1]
# G_b_min = (2*np.pi)/(theta_b[1]*L_b[1]**2*np.sin(1.5*np.pi/L_b[1]) + (2*np.pi-theta_b[1]))

# p_i = ptx(1)*G_s(1)*G_i*h_si*(c/(4*np.pi*f_s))^2*d_si^(-alpha_s)

coverage_user_g1_u = np.array([1])
coverage_user_g2_u = np.array([1])
coverage_user_g3_u = np.array([1])
coverage_user_g4_u = np.array([1])
coverage_user_g5_u = np.array([1])
coverage_user_g6_u = np.array([1])

user_choose_g1_u = np.array([1])
user_choose_g2_u = np.array([1])
user_choose_g3_u = np.array([1])
user_choose_g4_u = np.array([1])
user_choose_g5_u = np.array([1])
user_choose_g6_u = np.array([1])

coverage_user_g1_e = np.array([1])
coverage_user_g2_e = np.array([1])
coverage_user_g3_e = np.array([1])
coverage_user_g4_e = np.array([1])
coverage_user_g5_e = np.array([1])
coverage_user_g6_e = np.array([1])

user_choose_g1_e = np.array([1])
user_choose_g2_e = np.array([1])
user_choose_g3_e = np.array([1])
user_choose_g4_e = np.array([1])
user_choose_g5_e = np.array([1])
user_choose_g6_e = np.array([1])

group_e = 60  # Số user

group_u = 60

rate_control = -2

T = 100
delta_t = 0.1
t_choice = np.arange(0, T + delta_t, delta_t)

T_size = t_choice.size

n_RB = 3
h = np.ones((group_e + group_u, n_RB))

p_mm_1_e = np.zeros((group_e + group_u))
p_mm_2_e = np.zeros((group_e + group_u))
p_mm_3_e = 0
p_mm_4_e = 0
p_mm_5_e = 0
p_mm_6_e = np.zeros((group_e + group_u))

p_mm_1_u = np.zeros((group_e + group_u))
p_mm_2_u = np.zeros((group_e + group_u))
p_mm_3_u = 0
p_mm_4_u = 0
p_mm_5_u = 0
p_mm_6_u = np.zeros((group_e + group_u))

def Distance(l_a, l_b):
    dis_a_b = np.sqrt((l_a[0] - l_b[0])**2 + (l_a[1] - l_b[1])**2)
    return dis_a_b

def p_LoS(d_bi):
    p = np.minimum(18 / d_bi, 1) * (1 - np.exp(-d_bi / 63)) + np.exp(-d_bi / 63)
    # p = 0.1
    return p

#=========================================================================================
# eMBB user

def calculate_power_eMBB():

    global p_mm_1_e, p_mm_2_e, p_mm_3_e, p_mm_4_e, p_mm_5_e, p_mm_6_e, h

    # eMBB MMWave 1
    d_mi_1_e = Distance(U, l_M_1)
    p_LoS_mm_1_e = p_LoS(d_mi_1_e)
    for i in range(group_e + group_u):
        p_mm_1_e[i] = p_LoS_mm_1_e * ptx[1] * G_b[1] * G_i[1] * h[i, 0] * (3 * 10**8 / (4 * np.pi * f[1]))**2 * d_mi_1_e**(-alpha[1])

    # eMBB MMWave 2
    d_mi_2_e = Distance(U, l_M_2)
    p_LoS_mm_2_e = p_LoS(d_mi_2_e)
    for i in range(group_e + group_u):
        p_mm_2_e[i] = p_LoS_mm_2_e * ptx[1] * G_b[1] * G_i[1] * h[i, 1] * (3 * 10**8 / (4 * np.pi * f[1]))**2 * d_mi_2_e**(-alpha[1])

    # eMBB THz 1
    d_mi_3_e = Distance(U, l_T_1)
    p_LoS_thz_3_e = p_LoS(d_mi_3_e)
    p_mm_3_e = p_LoS_thz_3_e * ptx[2] * G_b[2] * G_i[2] * (3 * 10**8 / (4 * np.pi * f[2]))**2 * d_mi_3_e**(-alpha[2]) * np.exp(-0.03 * d_mi_3_e)

    # eMBB THz 2
    d_mi_4_e = Distance(U, l_T_2)
    p_LoS_thz_4_e = p_LoS(d_mi_4_e)
    p_mm_4_e = p_LoS_thz_4_e * ptx[2] * G_b[2] * G_i[2] * (3 * 10**8 / (4 * np.pi * f[2]))**2 * d_mi_4_e**(-alpha[2]) * np.exp(-0.03 * d_mi_4_e)

    # eMBB THz 3
    d_mi_5_e = Distance(U, l_T_3)
    p_LoS_thz_5_e = p_LoS(d_mi_5_e)
    p_mm_5_e = p_LoS_thz_5_e * ptx[2] * G_b[2] * G_i[2] * (3 * 10**8 / (4 * np.pi * f[2]))**2 * d_mi_5_e**(-alpha[2]) * np.exp(-0.03 * d_mi_5_e)

    # eMBB Sub-6GHz
    d_mi_6_e = Distance(U, l_S)
    for i in range(group_e + group_u):
        p_mm_6_e[i] = ptx[0] * G_b[0] * G_i[0] * h[i, 2] * (3 * 10**8 / (4 * np.pi * f[0]))**2 * d_mi_6_e**(-alpha[0])

#=========================================================================================
# URRLC user

def calculate_power_URRLC():

    global p_mm_1_u, p_mm_2_u, p_mm_3_u, p_mm_4_u, p_mm_5_u, p_mm_6_u, h 

    # URRLC MMWave 1
    d_mi_1_u = Distance(U, l_M_1)
    p_LoS_mm_1_u = p_LoS(d_mi_1_u)
    for i in range(group_e + group_u):
        p_mm_1_u[i] = p_LoS_mm_1_u * ptx[1] * G_b[1] * G_i[1] * h[i, 0] * (3 * 10**8 / (4 * np.pi * f[1]))**2 * d_mi_1_u**(-alpha[1])

    # URRLC MMWave 2
    d_mi_2_u = Distance(U, l_M_2)
    p_LoS_mm_2_u = p_LoS(d_mi_2_u)
    for i in range(group_e + group_u):
        p_mm_2_u[i] = p_LoS_mm_2_u * ptx[1] * G_b[1] * G_i[1] * h[i, 1] * (3 * 10**8 / (4 * np.pi * f[1]))**2 * d_mi_2_u**(-alpha[1])

    # URRLC THz 1
    d_mi_3_u = Distance(U, l_T_1)
    p_LoS_thz_3_u = p_LoS(d_mi_3_u)
    p_mm_3_u = p_LoS_thz_3_u * ptx[2] * G_b[2] * G_i[2] * (3 * 10**8 / (4 * np.pi * f[2]))**2 * d_mi_3_u**(-alpha[2]) * np.exp(-0.03 * d_mi_3_u)

    # URRLC THz 2
    d_mi_4_u = Distance(U, l_T_2)
    p_LoS_thz_4_u = p_LoS(d_mi_4_u)
    p_mm_4_u = p_LoS_thz_4_u * ptx[2] * G_b[2] * G_i[2] * (3 * 10**8 / (4 * np.pi * f[2]))**2 * d_mi_4_u**(-alpha[2]) * np.exp(-0.03 * d_mi_4_u)

    # URRLC THz 3
    d_mi_5_u = Distance(U, l_T_3)
    p_LoS_thz_5_u = p_LoS(d_mi_5_u)
    p_mm_5_u = p_LoS_thz_5_u * ptx[2] * G_b[2] * G_i[2] * (3 * 10**8 / (4 * np.pi * f[2]))**2 * d_mi_5_u**(-alpha[2]) * np.exp(-0.03 * d_mi_5_u)

    # URRLC Sub-6GHz
    d_mi_6_u = Distance(U, l_S)
    for i in range(group_e + group_u):
        p_mm_6_u[i] = ptx[0] * G_b[0] * G_i[0] * h[i, 2] *(3 * 10**8 / (4 * np.pi * f[0]))**2 * d_mi_6_u**(-alpha[0])

#=========================================================================================

def update_slow_fading():
    global h
    h = np.random.normal(np.exp(1), 0.05*np.exp(1), (group_e + group_u, n_RB))

#=========================================================================================

#eBMM 
def utility_THz_5(num_choose_5):
    global W_t, group_e, group_u, p_mm_5_e, N_0, H_t
   
    p_b_e = p_mm_5_e
    W_b_e = W_t / (num_choose_5)
    r_b_e = W_b_e * np.log2(1 + p_b_e / (N_0 * W_b_e))
    u_THz_5 = r_b_e * H_t

    return u_THz_5

def utility_THz_4(num_choose_4):
    global W_t, group_e, group_u, p_mm_4_e, N_0, H_t

    p_b_e = p_mm_4_e
    W_b_e = W_t / (num_choose_4)
    r_b_e = W_b_e * np.log2(1 + p_b_e / (N_0 * W_b_e))
    u_THz_4 = r_b_e * H_t

    return u_THz_4

def utility_THz_3(num_choose_3):
    global W_t, group_e, group_u, p_mm_3_e, N_0, H_t

    p_b_e = p_mm_3_e
    W_b_e = W_t / (num_choose_3)
    r_b_e = W_b_e * np.log2(1 + p_b_e / (N_0 * W_b_e))
    u_THz_3 = r_b_e * H_t

    return u_THz_3

def utility_Sub_6(num_choose_6, i):
    global W_s, group_e, group_u, p_mm_6_e, N_0, H_s

    p_b_e = p_mm_6_e[i]
    W_b_e = W_s / (num_choose_6)
    r_b_e = W_b_e * np.log2(1 + p_b_e / (N_0 * W_b_e))
    u_Sub_6 = r_b_e * H_s

    return u_Sub_6

def utility_mmWave_2(num_choose_2, i):
    global W_m, group_e, group_u, p_mm_2_e, N_0, H_m

    p_b_e = p_mm_2_e[i]
    W_b_e = W_m / (num_choose_2)
    r_b_e = W_b_e * np.log2(1 + p_b_e / (N_0 * W_b_e))
    u_mmWave_2 = r_b_e * H_m

    return u_mmWave_2

def utility_mmWave_1(num_choose_1, i):
    global W_m, group_e, group_u, p_mm_1_e, N_0, H_m

    p_b_e = p_mm_1_e[i]
    W_b_e = W_m / (num_choose_1)
    r_b_e = W_b_e * np.log2(1 + p_b_e / (N_0 * W_b_e))
    u_mmWave_1 = r_b_e * H_m

    return u_mmWave_1

#=========================================================================================

#URRLC

def utility_THz_5_u(num_choose_5):
    global W_t, group_e, group_u, p_mm_5_u, N_0, v_t_u, C_t, epsilon_u, T_u, L_u

    p_b_u = p_mm_5_u
    W_b_u = W_t / (num_choose_5)
    V_b_u = 1 - 1 / (1 + p_b_u / (N_0 * W_b_u))
    r_b_u = W_b_u / np.log(2) * (np.log(1 + p_b_u / (N_0 * W_b_u)) - np.sqrt(V_b_u / (T_u * W_b_u)) * erfcinv(2 * epsilon_u)**(-1))
    u_THz_5_u = v_t_u * (r_b_u - L_u / T_u) - C_t * r_b_u

    return u_THz_5_u


def utility_THz_4_u(num_choose_4):
    global W_t, group_e, group_u, p_mm_4_u, N_0, v_t_u, C_t, epsilon_u, T_u, L_u

    p_b_u = p_mm_4_u
    W_b_u = W_t / (num_choose_4)
    V_b_u = 1 - 1 / (1 + p_b_u / (N_0 * W_b_u))
    r_b_u = W_b_u / np.log(2) * (np.log(1 + p_b_u / (N_0 * W_b_u)) - np.sqrt(V_b_u / (T_u * W_b_u)) * erfcinv(2 * epsilon_u)**(-1))
    u_THz_4_u = v_t_u * (r_b_u - L_u / T_u) - C_t * r_b_u

    return u_THz_4_u

def utility_THz_3_u(num_choose_3):
    global W_t, group_e, group_u, p_mm_3_u, N_0, v_t_u, C_t, epsilon_u, T_u, L_u

    p_b_u = p_mm_3_u
    W_b_u = W_t / (num_choose_3)
    V_b_u = 1 - 1 / (1 + p_b_u / (N_0 * W_b_u))
    r_b_u = W_b_u / np.log(2) * (np.log(1 + p_b_u / (N_0 * W_b_u)) - np.sqrt(V_b_u / (T_u * W_b_u)) * erfcinv(2 * epsilon_u)**(-1))
    u_THz_3_u = v_t_u * (r_b_u - L_u / T_u) - C_t * r_b_u

    return u_THz_3_u

def utility_Sub_6_u(num_choose_6, i):
    global W_s, group_u, group_e, p_mm_6_u, N_0, v_s_u, C_s, epsilon_u, T_u, L_u

    p_b_u = p_mm_6_u[i]
    W_b_u = W_s / (num_choose_6)
    V_b_u = 1 - 1 / (1 + p_b_u / (N_0 * W_b_u))
    r_b_u = W_b_u / np.log(2) * (np.log(1 + p_b_u / (N_0 * W_b_u)) - np.sqrt(V_b_u / (T_u * W_b_u)) * erfcinv(2 * epsilon_u)**(-1))
    u_Sub_6_u = v_s_u * (r_b_u - L_u / T_u) - C_s * r_b_u

    return u_Sub_6_u

def utility_mmWave_2_u(num_choose_2, i):
    global W_m, group_u, group_e, p_mm_2_u, N_0, v_m_u, epsilon_u, T_u, C_m, L_u

    p_b_u = p_mm_2_u[i]
    W_b_u = W_m / (num_choose_2)
    V_b_u = 1 - 1 / (1 + p_b_u / (N_0 * W_b_u))
    r_b_u = W_b_u / np.log(2) * (np.log(1 + p_b_u / (N_0 * W_b_u)) - np.sqrt(V_b_u / (T_u * W_b_u)) * erfcinv(2 * epsilon_u)**(-1))
    u_mmWave_2_u = v_m_u * (r_b_u - L_u / T_u) - C_m * r_b_u

    return u_mmWave_2_u

def utility_mmWave_1_u(num_choose_1, i):
    global W_m, group_u, group_e, p_mm_1_u, N_0, v_m_u, C_m, T_u, epsilon_u, L_u

    p_b_u = p_mm_1_u[i]
    W_b_u = W_m / (num_choose_1)
    V_b_u = 1 - 1 / (1 + p_b_u / (N_0 * W_b_u))
    r_b_u = W_b_u / np.log(2) * (np.log(1 + p_b_u / (N_0 * W_b_u)) - np.sqrt(V_b_u / (T_u * W_b_u)) * erfcinv(2 * epsilon_u)**(-1))
    u_mmWave_1_u = v_m_u * (r_b_u - L_u / T_u) - C_m * r_b_u

    return u_mmWave_1_u

def act_for_training(actions):
    reward = 0
    action = actions[:, 0]
    reward_eBMM = np.zeros((group_e))
    reward_URRLC = np.zeros((group_u))
    num_choose_1 = 0
    num_choose_2 = 0
    num_choose_3 = 0
    num_choose_4 = 0
    num_choose_5 = 0
    num_choose_6 = 0

    for i in range(group_e + group_u):
        if action[i] == 0:
            num_choose_1 += 1
        if action[i] == 1:
            num_choose_2 += 1
        if action[i] == 2:
            num_choose_3 += 1
        if action[i] == 3:
            num_choose_4 += 1
        if action[i] == 4:
            num_choose_5 += 1
        if action[i] == 5:
            num_choose_6 += 1
    
    for i in range(group_e):
        if action[i] == 0:
            reward_eBMM[i] = utility_mmWave_1(num_choose_1, i)
        if action[i] == 1:
            reward_eBMM[i] = utility_mmWave_2(num_choose_2, i)
        if action[i] == 2:
            reward_eBMM[i] = utility_THz_3(num_choose_3)
        if action[i] == 3:
            reward_eBMM[i] = utility_THz_4(num_choose_4)
        if action[i] == 4:
            reward_eBMM[i] = utility_THz_5(num_choose_5)
        if action[i] == 5:
            reward_eBMM[i] = utility_Sub_6(num_choose_6, i)

    for i in range(group_u):
        if action[group_e + i] == 0:
            reward_URRLC[i] = utility_mmWave_1_u(num_choose_1, group_e + i)
        if action[group_e + i] == 1:
            reward_URRLC[i] = utility_mmWave_2_u(num_choose_2, group_e + i)
        if action[group_e + i] == 2:
            reward_URRLC[i] = utility_THz_3_u(num_choose_3)
        if action[group_e + i] == 3:
            reward_URRLC[i] = utility_THz_4_u(num_choose_4)
        if action[group_e + i] == 4:
            reward_URRLC[i] = utility_THz_5_u(num_choose_5)
        if action[group_e + i] == 5:
            reward_URRLC[i] = utility_Sub_6_u(num_choose_6, group_e + i)
    
    if np.sum(reward_eBMM) + np.sum(reward_URRLC) < 3130:
        reward = np.sum(reward_eBMM) + np.sum(reward_URRLC)
    else:
        reward = 6000
    # print(reward)
    a = np.array([num_choose_1, num_choose_2, num_choose_3, num_choose_4, num_choose_5, num_choose_6])
    # print(a)

    return reward / (group_e + group_u), a, np.sum(reward_eBMM) + np.sum(reward_URRLC), np.sum(reward_eBMM)/group_e*6, np.sum(reward_URRLC)/group_u*6

def new_random_game():
    update_slow_fading()
    calculate_power_eMBB()
    calculate_power_URRLC()
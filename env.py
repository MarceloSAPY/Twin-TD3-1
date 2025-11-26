#%matplotlib inline
import numpy as np
from entity import *
from channel import *
from math_tool import *
from datetime import datetime
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from render import Render
from data_manager import DataManager
# s.t every simulition is the same model
np.random.seed(2)

######################################################
# new for energy 
# energy related parameters of rotary-wing UAV
# based on Energy Minimization in Internet-of-Things System Based on Rotary-Wing UAV
P_i = 790.6715
P_0 = 580.65
U2_tip = (200) ** 2
s = 0.05
d_0 = 0.3
p = 1.225
A = 0.79
delta_time = 0.1/1000 #0.1ms

# add ons hover veloctiy
# based on https://www.intechopen.com/chapters/57483
m = 1.3 # mass: assume 1.3kg https://www.droneblog.com/average-weights-of-common-types-of-drones/#:~:text=In%20most%20cases%2C%20toy%20drones,What%20is%20this%3F
g = 9.81 # gravity
T = m * g # thrust
v_0 = (T / (A * 2 * p)) ** 0.5

def get_energy_consumption(v_t):
    '''
    arg
    1) v_t = displacement per time slot
    '''
    energy_1 = P_0 \
                + 3 * P_0 * (abs(v_t)) ** 2 / U2_tip \
                + 0.5 * d_0 * p * s * A * (abs(v_t))**3
    
    energy_2 = P_i * ((
                    (1 + (abs(v_t) ** 4) / (4 * (v_0 ** 4))) ** 0.5 \
                    - (abs(v_t) ** 2) / (2 * (v_0 **2)) \
                ) ** 0.5)
    
    energy = delta_time * (energy_1 + energy_2)
    return energy 

ENERGY_MIN = get_energy_consumption(0.25)
ENERGY_MAX = get_energy_consumption(0)

######################################################


class MiniSystem(object):
#class MiniSystem(K=1):
    """
    define mini RIS communication system with one UAV
        and one RIS and one user, one attacker
    """
    def __init__(self, UAV_num = 1, RIS_num = 1, user_num = 1, attacker_num = 0, fre = 28e9, \
                 RIS_ant_num = 32, UAV_ant_num=8, if_dir_link = 1, if_with_RIS = True, \
                 if_move_users = True, if_movements = True, reverse_x_y = (True, True), \
                 if_UAV_pos_state = True, reward_design = 'ssr', project_name = None, step_num=100):
        self.if_dir_link = if_dir_link
        self.if_with_RIS = if_with_RIS
        self.if_move_users = if_move_users
        self.if_movements = if_movements
        self.if_UAV_pos_state = if_UAV_pos_state
        self.reverse_x_y = reverse_x_y
        self.user_num = user_num
        self.attacker_num = attacker_num
        self.border = [(-25,25), (0, 50)]
        # --- NEW: Hybrid RIS Parameters () ---
        # 1. Noise Power Setup (Based on Nguyen et al., 2024, Table 1 & Sec V)
        # User noise power: -80 dBm
        self.sigma_u_dBm = -80 
        self.sigma_u = 10 ** (self.sigma_u_dBm / 10) * 1e-3 # Convert dBm to Watts (Linear)
        # 1.init entities: 1 UAV, 1 RIS, many users and attackers
        self.data_manager = DataManager(file_path='./data', project_name = project_name, \
        store_list = ['beamforming_matrix', 'reflecting_coefficient', 'UAV_state', 'user_capacity', 'secure_capacity', 'attaker_capacity','G_power', 'reward','UAV_movement','total_power'])
        # 1.1 init UAV position and beamforming matrix
        self.UAV = UAV(
            coordinate=self.data_manager.read_init_location('UAV', 0), 
            ant_num= UAV_ant_num, 
            max_movement_per_time_slot=0.25)
        self.UAV.G = np.mat(np.ones((self.UAV.ant_num, user_num), dtype=complex), dtype=complex)
        self.power_factor = 100
        self.UAV.G_Pmax =  np.trace(self.UAV.G * self.UAV.G.H) * self.power_factor
        # 1.2 init RIS
        self.RIS = RIS(\
        coordinate=self.data_manager.read_init_location('RIS', 0), \
        coor_sys_z=self.data_manager.read_init_location('RIS_norm_vec', 0), \
        ant_num=RIS_ant_num)
        # 1.3 init users
        self.user_list = []
        
        for i in range(user_num):
            user_coordinate = self.data_manager.read_init_location('user', i)
            user = User(coordinate=user_coordinate, index=i)
            user.noise_power = self.sigma_u_dBm # Update to -80 dBm
            self.user_list.append(user)

        # 1.4 init attackers
        self.attacker_list = []
        
        for i in range(attacker_num):
            attacker_coordinate = self.data_manager.read_init_location('attacker', i)
            attacker = Attacker(coordinate=attacker_coordinate, index=i)
            attacker.capacity = np.zeros((user_num))
            attacker.noise_power = -114
            self.attacker_list.append(attacker)
        # 1.5 generate the eavesdrop capacity array , shape: P X K
        self.eavesdrop_capacity_array= np.zeros((attacker_num, user_num))
        
        # 1.6 reward design
        self.reward_design = reward_design # reward_design is ['ssr' or 'see']

        # 1.7 step_num
        self.step_num = step_num
                
        # 2. Active RIS Noise / Self-Interference (Eq 3 & 5)
        # sigma_r^2 = (eta + 1) * sigma_u^2
        # eta = 1 dB reflects residual Self-Interference (SI)
        self.eta_dB = 1 
        self.eta_lin = 10 ** (self.eta_dB / 10)
        # Noise power at RIS (Thermal Noise + Residual SI)
        self.sigma_r = (self.eta_lin + 1) * self.sigma_u 

        # 3. Amplification Limit (Sec II.A)
        # Maximum gain for active elements is 40 dB
        self.a_max_dB = 40
        # Convert Power Gain (dB) to Voltage/Amplitude Gain limit: 
        # Gain_dB = 20 * log10(Amplitude) -> Amplitude = 10^(dB/20)
        self.a_max = 10 ** (self.a_max_dB / 20)
        # 3.1 --- GAP 3 FIX: POWER BUDGET ---
        # Define the maximum power the active RIS is allowed to consume.
        # Paper uses 0 dBm (1 mW) or 5 dBm.
        self.p_max_ris_dBm = 5 # Let's give it 5 dBm (3.1 mW) to be safe
        self.p_max_ris = 10 ** (self.p_max_ris_dBm / 10) * 1e-3
        # 4. Active Elements (Assuming partial active, e.g., 4 elements)
        self.num_active_elements = 4 # You can parameterize this later
        # ------------------------------------------
        # 2.init channel
        self.H_UR = mmWave_channel(self.UAV, self.RIS, fre)
        self.h_U_k = []
        self.h_R_k = []
        self.h_U_p = []
        self.h_R_p = []
        for user_k in self.user_list:
            self.h_U_k.append(mmWave_channel(user_k, self.UAV, fre))
            self.h_R_k.append(mmWave_channel(user_k, self.RIS, fre))
        for attacker_p in self.attacker_list:
            self.h_U_p.append(mmWave_channel(attacker_p, self.UAV, fre))
            self.h_R_p.append(mmWave_channel(attacker_p, self.RIS, fre))

        # 3 update user and attaker channel capacity
        self.update_channel_capacity()

        # 4 draw system
        self.render_obj = Render(self)      
        
    def reset(self):
        """
        reset UAV, users, attackers, beamforming matrix, reflecting coefficient
        """
        # 1 reset UAV
        self.UAV.reset(coordinate=self.data_manager.read_init_location('UAV', 0))
        # 2 reset users
        for i in range(self.user_num):
            user_coordinate = self.data_manager.read_init_location('user', i)
            self.user_list[i].reset(coordinate=user_coordinate)
        # 3 reset attackers
        for i in range(self.attacker_num):
            attacker_coordinate = self.data_manager.read_init_location('attacker', i)
            self.attacker_list[i].reset(coordinate=attacker_coordinate)
        # 4 reset beamforming matrix
        self.UAV.G = np.mat(np.ones((self.UAV.ant_num, self.user_num), dtype=complex), dtype=complex)
        self.UAV.G_Pmax = np.trace(self.UAV.G * self.UAV.G.H) * self.power_factor
        # 5 reset reflecting coefficient
        """self.RIS = RIS(\
        coordinate=self.data_manager.read_init_location('RIS', 0), \
        coor_sys_z=self.data_manager.read_init_location('RIS_norm_vec', 0), \
        ant_num=16)"""
        self.RIS.Phi = np.mat(np.diag(np.ones(self.RIS.ant_num, dtype=complex)), dtype = complex)
        # 6 reset time
        self.render_obj.t_index = 0
        # 7 reset CSI
        self.H_UR.update_CSI()
        for h in self.h_U_k + self.h_U_p + self.h_R_k + self.h_R_p:
            h.update_CSI()
        # 8 reset capcaity
        self.update_channel_capacity()

    def step(self, action_0 = 0, action_1 = 0, G = 0, Phi = 0, set_pos_x = 0, set_pos_y = 0):
        """
        test step only move UAV and update channel
        """
        # 0 update render
        
        self.render_obj.t_index += 1
        # 1 update entities
        
        if self.if_move_users:
            self.user_list[0].update_coordinate(0.2, -1/2 * math.pi)
            self.user_list[1].update_coordinate(0.2, -1/2 * math.pi)

        if self.if_movements:
            move_x = action_0 * self.UAV.max_movement_per_time_slot
            move_y = action_1 * self.UAV.max_movement_per_time_slot
            
            ######################################################
            # new for energy 
            v_t = (move_x ** 2 + move_y ** 2) ** 0.5
            #self.data_manager.store_data([v_t],'velocity')
            ######################################################

            if self.reverse_x_y[0]:
                move_x = -move_x
            
            if self.reverse_x_y[1]:
                move_y = -move_y
                
            self.UAV.coordinate[0] +=move_x
            self.UAV.coordinate[1] +=move_y
            self.data_manager.store_data([move_x, move_y], 'UAV_movement')
        else:
            set_pos_x = map_to(set_pos_x, (-1, 1), self.border[0])
            set_pos_y = map_to(set_pos_y, (-1, 1), self.border[1])
            self.UAV.coordinate[0] = set_pos_x
            self.UAV.coordinate[1] = set_pos_y

        # 2 update channel CSI
        
        for h in self.h_U_k + self.h_U_p + self.h_R_k + self.h_R_p:
            h.update_CSI()
        # !!! test to make direct link zero
        if self.if_dir_link == 0:
            for h in self.h_U_k + self.h_U_p:
                h.channel_matrix = np.mat(np.zeros(shape = np.shape(h.channel_matrix)), dtype=complex)
        if self.if_with_RIS == False:
            self.H_UR.channel_matrix = np.mat(np.zeros((self.RIS.ant_num, self.UAV.ant_num)), dtype=complex)
        else:
            self.H_UR.update_CSI()
        # 3 update beamforming matrix & reflecting phase shift
        """
        self.UAV.G = G
        self.RIS.Phi = Phi
        """
        self.UAV.G = convert_list_to_complex_matrix(G, (self.UAV.ant_num, self.user_num)) * math.pow(self.power_factor, 0.5)
        
        # fix beamforming matrix
        #self.UAV.G = np.mat(np.ones((self.UAV.ant_num, self.user_num), dtype=complex), dtype=complex) * math.pow(self.power_factor, 0.5)
        if self.if_with_RIS:
            # 1. Convert Action to Matrix (Standard Phase Shift)
            # This creates a diagonal matrix with entries e^(j*theta), amplitude = 1
            self.RIS.Phi = convert_list_to_complex_diag(Phi, self.RIS.ant_num)
            
            # --- GAP 1 FIX: ACTION SCALING ---
            # The agent outputs actions for Phase, but we must enforce Amplification for Active Elements.
            # We assume the first N_a elements are the active ones.
            
            N_a = getattr(self, 'num_active_elements', 4)
            a_max = getattr(self, 'a_max', 100) # Default to 100 (40dB) if not set
            
            # Extract the diagonal (the coefficients)
            # We need to convert the matrix to an array to edit it easily
            phi_diag = np.diag(self.RIS.Phi).copy()
            
            # Scale the Active Elements
            # We set their amplitude to a_max (Max Gain)
            # Note: Ideally, the agent would control gain, but since we have fixed output dimensions, 
            # forcing Max Gain is a standard simplification for Hybrid RIS.
            for i in range(N_a):
                phi_diag[i] = phi_diag[i] * a_max
            
            # Reconstruct the diagonal matrix
            self.RIS.Phi = np.mat(np.diag(phi_diag))
            # ---------------------------------
        # 4 update channel capacity in every user and attacker
        self.update_channel_capacity()
        # 5 store current system state to .mat
        self.store_current_system_sate()
        # ... inside step() ...
        
        # 6 get new state
        new_state = self.observe()
        
        # 7 get reward (OLD LOGIC - DISABLED)
        # reward = self.reward() 
        
        # Initialize reward to 0 just in case
        reward = 0
        
        # 7.1 reward with energy efficiency
        ######################################################
        if self.reward_design == 'fair':            
            # --- A. Calculate Total Power Consumption (The Cost) ---            
            # 1. UAV Propulsion Power (Existing global function)
            # v_t was calculated earlier in step() loop
            power_propulsion = get_energy_consumption(v_t) / delta_time 
            
            # 2. UAV Transmission Power
            # Sum of squared magnitudes of beamforming vector G
            power_transmit = np.trace(self.UAV.G * self.UAV.G.H).real
            
           # 3. Hybrid RIS Active Power
           # a) Calculate Incident Signal Matrix
            # Signal traveling UAV -> RIS = H_UR * G
            # Result is a matrix of shape (N_RIS_Elements, K_Users)
            incident_signal_matrix = self.H_UR.channel_matrix * self.UAV.G                                    
            # b) Calculate Incident Power per Element
            # MODIFIED: Use np.square() to force element-wise operation on numpy matrices
            incident_power_per_element = np.sum(np.square(np.abs(incident_signal_matrix)), axis=1)

            # c) Calculate Active Power Consumption
            # We only consume power for the ACTIVE elements (first N_a)
            N_a = getattr(self, 'num_active_elements', 4)
            sigma_r = getattr(self, 'sigma_r', 1e-9) # Noise floor from Step 2
            
            # Extract Amplification Factors (|alpha|^2) for active elements
            # Psi is the diagonal matrix of RIS coefficients
            Psi_diag = np.diag(self.RIS.Phi)
            active_gains_sq = np.abs(Psi_diag[:N_a])**2
            
            # Extract Incident Power for active elements
            # Flatten to ensure it matches the shape of active_gains_sq
            # OLD: active_incident_power = np.array(incident_power_per_element[:N_a]).flatten()
            # NEW: Cap the incident power to avoid explosion
            # Many implementations assume automatic gain control (AGC) limits input power
            raw_incident_power = np.array(incident_power_per_element[:N_a]).flatten()
            active_incident_power = np.clip(raw_incident_power, 0, 1.0) # Cap at 1 Watt (example)
            
            # Apply Equation (7): Sum( |alpha|^2 * (sigma_r^2 + Incident_Power) )
            power_ris_active = np.sum(active_gains_sq * (sigma_r**2 + active_incident_power))
            # --- GAP 3 FIX: POWER CONSTRAINT PENALTY ---
            # If the agent tries to use more power than the hardware allows, punish it.
            # This forces the agent to learn "Efficient Amplification".
            
            power_penalty = 0
            if power_ris_active > self.p_max_ris:
                # Penalty proportional to the violation amount
                # Multiplied by a factor (e.g., 10 or 100) to make it hurt
                diff = power_ris_active - self.p_max_ris
                power_penalty = -10 * diff 
                
                # Option: You could also clamp the power to the max for the total_power calc
                # power_ris_active = self.p_max_ris 
            # -------------------------------------------

            # 4. Total System Power
            # 4. Total System Power
            # This is the denominator of your EE metric
            total_power = power_propulsion + power_transmit + power_ris_active
            
            # (We will use 'total_power' in Step 5 to calculate the final Reward)
            # For now, to prevent errors, let's just define a temporary placeholder reward
            reward = 0
            # ... inside step() ... inside 'fair' block ...
            # (After calculating total_power from Step 4)
            # --- B. Calculate Fairness Reward (Step 5) ---            
            # 1. Compute Rates for ALL users (Max-Min Logic)
            rates = []
            for k in range(self.user_num):
                # Use the updated capacity function (which includes Hybrid Noise from Step 3)
                # Note: calculate_capacity_of_user_k returns log10(1+SINR). 
                # For standard rate (nats/s/Hz), we might want log(1+SINR) (natural log).
                # The original code used math.log10, so we stick to it for consistency, 
                # or convert to natural log if comparing with paper (multiply by ln(10)).
                r_k = self.calculate_capacity_of_user_k(k) 
                rates.append(r_k)
            
            # 2. Identify the Bottleneck User (Fairness)
            # The reward is limited by the WORST user.
            min_rate = np.min(rates)            
            # 3. Calculate Fairness-Aware Energy Efficiency
            # Objective = Min_Rate / Total_Power
            # Unit: Bits (or nats) per Joule            
            # Small epsilon to prevent division by zero if power is 0 (unlikely)
            epsilon = 1e-6
            
            reward = min_rate / (total_power + epsilon)
            
            # 4. Scaling (Important for DRL stability)
            # EE values can be very small (e.g., 1e-6). DRL agents struggle with tiny gradients.
            # We scale it up to a reasonable range (e.g., 0-10 or 0-100).
            reward_scale_factor = 1000 
            reward = reward * reward_scale_factor

            # (Optional) Penalty for extremely low fairness (Dead Zone)
            # If the worst user has 0 rate, give a large negative penalty to force coverage.
            if min_rate < 1e-2:
                reward -= 5
        
        if self.reward_design == 'see':
            # new for see
            energy = energy_raw = get_energy_consumption(v_t)
            energy -= ENERGY_MIN
            energy /= (ENERGY_MAX - ENERGY_MIN)
            energy_penalty = -1 * 0.1 * abs(reward) * energy # -1 * 0.1 * reward * energy
            if reward > 0:
                reward += energy_penalty
        ######################################################
        
        # 8 calculate if UAV is cross the bourder
        reward = math.tanh(reward) # new for energy (ori not commented)
        done = False
        x, y = self.UAV.coordinate[0:2]
        if x < self.border[0][0] or x > self.border[0][1]:
            done = True
            reward = -10
        if y < self.border[1][0] or y > self.border[1][1]:
            done = True
            reward = -10
        self.data_manager.store_data([reward],'reward')
        return new_state, reward, done, []

    def reward(self):
        """
        used in function step to get the reward of current step
        """
        reward = 0
        reward_ = 0
        P = np.trace(self.UAV.G * self.UAV.G.H)
        if abs(P) > abs(self.UAV.G_Pmax) :
            reward = abs(self.UAV.G_Pmax) - abs(P)
            reward /= self.power_factor 
        else:
            for user in self.user_list:
                r = user.capacity - max(self.eavesdrop_capacity_array[:, user.index])
                if r < user.QoS_constrain:
                    reward_ += r - user.QoS_constrain
                else:
                    reward += r/(self.user_num*2)
            if reward_ < 0:
                reward = reward_ * self.user_num * 10
     
        return reward
    
    def observe(self):
        """
        used in function main to get current state
        the state is a list with 
        MODIFIED: Added User Rates (Gap 2)
        """
        # users' and attackers' comprehensive channel
        comprehensive_channel_elements_list = []
        for entity in self.user_list + self.attacker_list:
            tmp_list = list(np.array(np.reshape(entity.comprehensive_channel, (1,-1)))[0])
            comprehensive_channel_elements_list += list(np.real(tmp_list)) + list(np.imag(tmp_list)) 
        UAV_position_list = []
        if self.if_UAV_pos_state:
            UAV_position_list = list(self.UAV.coordinate)

        # --- GAP 2 FIX: Add User Rates to State ---
        # Now the beamforming agent sees exactly who is suffering
        user_rates_list = [user.capacity for user in self.user_list]
        
        return comprehensive_channel_elements_list + UAV_position_list + user_rates_list
        

    def store_current_system_sate(self):
        """
        function used in step() to store system state
        """
        # 1 store beamforming matrix
        row_data = list(np.array(np.reshape(self.UAV.G, (1, -1)))[0,:])
        self.data_manager.store_data(row_data, 'beamforming_matrix')
        # 2 store reflecting coefficient matrix
        row_data = list(np.array(np.reshape(diag(self.RIS.Phi), (1,-1)))[0,:])      
        self.data_manager.store_data(row_data, 'reflecting_coefficient')
        # 3 store UAV state
        row_data = list(self.UAV.coordinate)
        self.data_manager.store_data(row_data, 'UAV_state')
        # 4 store user_capicity
        row_data = [user.secure_capacity for user in self.user_list] \
        + [user.capacity for user in self.user_list]
        # 5 store G_power
        row_data = [np.trace(self.UAV.G*self.UAV.G.H), self.UAV.G_Pmax]
        self.data_manager.store_data(row_data, 'G_power')
        # --- NEW: Store Total Power ---
        # We must ensure self.total_power is set before this function is called!
        row_data = [getattr(self, 'total_power', 0)] 
        self.data_manager.store_data(row_data, 'total_power')
        for user in self.user_list:
            row_data.append(user.capacity)
        self.data_manager.store_data(row_data, 'user_capacity')

        row_data = []
        for attacker in self.attacker_list:
            row_data.append(attacker.capacity)
        self.data_manager.store_data(row_data, 'attaker_capacity')

        row_data = []
        for user in self.user_list:
            row_data.append(user.secure_capacity)
        self.data_manager.store_data(row_data, 'secure_capacity')


    def update_channel_capacity(self):
        """
        function used in step to calculate user and attackers' capacity 
        """
        # 1 calculate eavesdrop rate
        #for attacker in self.attacker_list:
        #    attacker.capacity = self.calculate_capacity_array_of_attacker_p(attacker.index)
        #    self.eavesdrop_capacity_array[attacker.index, :] = attacker.capacity
        #    # remmeber to update comprehensive_channel
        #    attacker.comprehensive_channel = self.calculate_comprehensive_channel_of_attacker_p(attacker.index)
        # 2 calculate unsecure rate
        for user in self.user_list:
            # Calculate Capacity (using the new noise model from Step 3)
            user.capacity = self.calculate_capacity_of_user_k(user.index)
            
            # 3. Calculate secure rate (DISABLED - Not needed for Fairness)
            # user.secure_capacity = self.calculate_secure_capacity_of_user_k(user.index)
            
            # Update Channel State (REQUIRED for Agent Observation)
            user.comprehensive_channel = self.calculate_comprehensive_channel_of_user_k(user.index)

    def calculate_comprehensive_channel_of_attacker_p(self, p):
        """
        used in update_channel_capacity to calculate the comprehensive_channel of attacker p
        """
        h_U_p = self.h_U_p[p].channel_matrix
        h_R_p = self.h_R_p[p].channel_matrix
        Psi = diag_to_vector(self.RIS.Phi)
        H_c = vector_to_diag(h_R_p).H * self.H_UR.channel_matrix
        return h_U_p.H + Psi.H * H_c

    def calculate_comprehensive_channel_of_user_k(self, k):
        """
        used in update_channel_capacity to calculate the comprehensive_channel of user k
        """
        h_U_k = self.h_U_k[k].channel_matrix
        h_R_k = self.h_R_k[k].channel_matrix
        Psi = diag_to_vector(self.RIS.Phi)
        H_c = vector_to_diag(h_R_k).H * self.H_UR.channel_matrix
        return h_U_k.H + Psi.H * H_c

    def calculate_capacity_of_user_k(self, k):
        """
        function used in update_channel_capacity to calculate one user
        MODIFIED: Includes Hybrid RIS Active Noise (Step 3)
        Based: equation 3, equation  7 power of active elements of ris
        """     
        # 1. Identify User Noise Power (Linear Watts)
        # Existing code stores noise_power in dBm. Convert to Watts.
        user_noise_power = dB_to_normal(self.user_list[k].noise_power) * 1e-3
        
        h_U_k = self.h_U_k[k].channel_matrix
        h_R_k = self.h_R_k[k].channel_matrix
        Psi = diag_to_vector(self.RIS.Phi)
        H_c = vector_to_diag(h_R_k).H * self.H_UR.channel_matrix
        
        G_k = self.UAV.G[:, k]
        G_k_ = 0
        if len(self.user_list) == 1:
            G_k_ = np.mat(np.zeros((self.UAV.ant_num, 1), dtype=complex), dtype=complex)
        else:
            G_k_1 = self.UAV.G[:, 0:k]
            G_k_2 = self.UAV.G[:, k+1:]
            G_k_ = np.hstack((G_k_1, G_k_2))
            
        # 2. Calculate Signal Power (Numerator)
        alpha_k = math.pow(abs((h_U_k.H + Psi.H * H_c) * G_k), 2)
        
        # 3. Calculate Interference Power (from other users)
        interference = math.pow(np.linalg.norm((h_U_k.H + Psi.H * H_c)*G_k_), 2)
        
        # --- NEW: Hybrid RIS Amplified Noise (Step 3) ---
        # Formula: sum( |h_{2,k,n}|^2 * |alpha_n|^2 ) * sigma_r
        # Only active elements amplify noise.
        
        # Get number of active elements (defined in Step 2)
        N_a = getattr(self, 'num_active_elements', 4) 
        
        # Flatten arrays for element-wise calculation
        h_R_k_flat = np.array(h_R_k).flatten() # Channel from RIS to User
        Psi_flat = np.array(Psi).flatten()     # RIS Coefficients (Active + Passive)
        
        active_noise_power = 0.0
        # the code assume the first N_a elements are the active ones (indices 0 to N_a-1)
        # This matches the paper's simulation setup where A = {1, ..., Na}
        # 
        if hasattr(self, 'sigma_r'):
            for n in range(N_a):
                # |Channel|^2 * |Gain|^2
                channel_gain_sq = abs(h_R_k_flat[n]) ** 2
                amp_gain_sq = abs(Psi_flat[n]) ** 2
                
                # Add noise contribution: Gain * Noise_Floor_at_RIS
                active_noise_power += channel_gain_sq * amp_gain_sq * self.sigma_r
        else:
            # Fallback if Step 2 wasn't run correctly
            print("Warning: sigma_r not defined. Skipping Active Noise.")
            
        # 4. Total Denominator (Interference + Active Noise + User Noise)
        beta_k = interference + active_noise_power + user_noise_power
        
        # Return Rate (Using log10 as per original codebase convention)
        return math.log10(1 + abs(alpha_k / beta_k))

    def calculate_capacity_array_of_attacker_p(self, p):
        """
        function used in update_channel_capacity to calculate one attacker capacities to K users
        output is a K length np.array ,shape: (K,)
        """
        K = len(self.user_list)
        noise_power = self.attacker_list[p].noise_power
        h_U_p = self.h_U_p[p].channel_matrix
        h_R_p = self.h_R_p[p].channel_matrix
        Psi = diag_to_vector(self.RIS.Phi)
        H_c = vector_to_diag(h_R_p).H * self.H_UR.channel_matrix
        if K == 1:
            G_k = self.UAV.G
            G_k_ = np.mat(np.zeros((self.UAV.ant_num, 1), dtype=complex), dtype=complex)
            alpha_p = math.pow(abs((h_U_p.H + Psi.H * H_c) * G_k), 2)
            beta_p = math.pow(np.linalg.norm((h_U_p.H + Psi.H * H_c)*G_k_), 2) + dB_to_normal(noise_power) * 1e-3
            return np.array([math.log10(1 + abs(alpha_p / beta_p))])
        else:
            result = np.zeros(K)
            for k in range(K):
                G_k = G_k = self.UAV.G[:, k]
                G_k_1 = self.UAV.G[:, 0:k]
                G_k_2 = self.UAV.G[:, k+1:]
                G_k_ = np.hstack((G_k_1, G_k_2))
                alpha_p = math.pow(abs((h_U_p.H + Psi.H * H_c) * G_k), 2)
                beta_p = math.pow(np.linalg.norm((h_U_p.H + Psi.H * H_c)*G_k_), 2) + dB_to_normal(noise_power) * 1e-3
                result[k] = math.log10(1 + abs(alpha_p / beta_p))
            return result

    def calculate_secure_capacity_of_user_k(self, k=2):
        """
        function used in update_channel_capacity to calculate the secure rate of user k
        """
        user = self.user_list[k]
        R_k_unsecure = user.capacity
        R_k_maxeavesdrop = max(self.eavesdrop_capacity_array[:, k])
        secrecy_rate= max(0, R_k_unsecure - R_k_maxeavesdrop)
        return secrecy_rate

    def get_system_action_dim(self):
        """
        function used in main function to get the dimention of actions
        """
        result = 0
        # 0 UAV movement
        result += 2
        # 1 RIS reflecting elements
        if self.if_with_RIS:
            result += self.RIS.ant_num   
        else:
            result += 0
        # 2 beamforming matrix dimention
        result += 2 * self.UAV.ant_num * self.user_num 
        return result

    def get_system_state_dim(self):
        """
        function used in main function to get the dimention of states
        MODIFIED: Added dimension for User Rates (Gap 2)
        """
        result = 0
        # users' and attackers' comprehensive channel
        result += 2 * (self.user_num + self.attacker_num) * self.UAV.ant_num
        # UAV position
        if self.if_UAV_pos_state:
            result += 3
            
        # --- GAP 2 FIX: Add User Rates Dimension ---
        result += self.user_num 
        # -------------------------------------------
        return result

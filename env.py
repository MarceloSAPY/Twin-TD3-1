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
delta_time = 0.1#0.1ms

# add ons hover veloctiy
m = 1.3 # mass: assume 1.3kg 
g = 9.81 # gravity
T = m * g # thrust
v_0 = (T / (A * 2 * p)) ** 0.5

def get_energy_consumption(v_t):
    '''
    Energy Consumption Model Explanation
    arg
    1) v_t = displacement per time slot
    '''
    energy_1 = P_0 \
                + 3 * P_0 * (abs(v_t)) ** 2 / U2_tip \
                + 0.5 * d_0 * p * s * A * (abs(v_t))**3
    
# Calcular el término interno por separado
    term_inside_sqrt = (1 + (abs(v_t) ** 4) / (4 * (v_0 ** 4))) ** 0.5 - (abs(v_t) ** 2) / (2 * (v_0 **2))
    
    # BLINDAJE: Si es negativo por error de decimales, lo forzamos a 0
    term_inside_sqrt = max(0.0, term_inside_sqrt)
    
    energy_2 = P_i * (term_inside_sqrt ** 0.5) 
        
    energy = delta_time * (energy_1 + energy_2)
    return energy 

ENERGY_MIN = get_energy_consumption(0.25)
ENERGY_MAX = get_energy_consumption(0)

######################################################


class MiniSystem(object):
    # Minimum reward value for clipping
    REWARD_MIN_CLIP = -5

#class MiniSystem(K=1):
    """
    define mini RIS communication system with one UAV
        and one RIS and one user.
    """
    def __init__(self, UAV_num = 1, RIS_num = 1, user_num = 1, fre = 28e9, \
                 RIS_ant_num = 32, UAV_ant_num=8, if_dir_link = 1, if_with_RIS = True, \
                 if_move_users = False, if_movements = True, reverse_x_y = (True, True), \
                 if_UAV_pos_state = True, reward_design = 'ssr', project_name = None, step_num=400):
        self.if_dir_link = if_dir_link
        self.if_with_RIS = if_with_RIS
        self.if_move_users = if_move_users
        self.if_movements = if_movements
        self.if_UAV_pos_state = if_UAV_pos_state
        # Si la red dice 1.0 (Moverse al máximo en X) y reverse_x es True, 
        # el código lo convierte en -1.0 (Moverse a la izquierda en lugar de a la derecha).
        self.reverse_x_y = reverse_x_y
        self.user_num = user_num
        self.border = [(-200,200), (0, 400)]
        # --- NEW: Hybrid RIS Parameters () ---
        # 1. Noise Power Setup (Based on Nguyen et al., 2024, Table 1 & Sec V)
        # User noise power: -80 dBm
        # Usas -80 dBm porque estás simulando un receptor con un ancho de banda de 
        # 20 MHz que opera en condiciones reales
        self.sigma_u_dBm = -80 
        self.sigma_u = 10 ** (self.sigma_u_dBm / 10) * 1e-3 # Convert dBm to Watts (Linear)
        # 1.init entities: 1 UAV, 1 RIS, many users
        self.data_manager = DataManager(file_path='./data', project_name = project_name, \
        store_list = [
            'beamforming_matrix',
            'reflecting_coefficient', 
            'UAV_state', 
            'user_capacity', 
            'G_power', 
            'reward',
            'UAV_movement',
            'total_power',
            'user_positions',
            "RIS_position"])
        # 1 init UAV position and beamforming matrix
        #============================================================   
        # 1.1 init UAV position and beamforming matrix
        self.UAV = UAV(
            coordinate=self.data_manager.read_init_location('UAV', 0), 
            ant_num= UAV_ant_num, 
            max_movement_per_time_slot=1.5)
        
        # --- CORRECCIÓN: FÍSICA REALISTA Y COMPATIBLE ---
        
        # 1.1.1 Definir Potencia Máxima (30 dBm = 1 Watt)
        self.p_uav_dBm = 30 
        self.UAV.G_Pmax = 10 ** (self.p_uav_dBm / 10) * 1e-3 
        
        # 1.1.2. Definir power_factor (CRÍTICO para evitar el crash)
        # Lo igualamos a G_Pmax (1.0). Así el código sigue funcionando 
        # pero la escala base es 1 Watt, no 100.
        self.power_factor = self.UAV.G_Pmax 
        
        # 1.1.3. Variable de compatibilidad para tu test (Opcional, para que pase el assert)
        self.p_uav_watts = self.UAV.G_Pmax

        # 1.1.4. Inicializar G con Ceros (Seguro para el arranque)
        # La lógica inteligente (MRT) se aplicará en reset()
        self.UAV.G = np.mat(np.zeros((self.UAV.ant_num, user_num), dtype=complex))
        #============================================================
        # 1.2 init RIS
        # Colocamos el RIS en el borde superior (y=50), centrado en X=0.
        # Esto simula un RIS montado en una fachada o pared.
        ris_pos = np.array([0, 400, 20]) 
        # Vector Normal: Apunta hacia adentro del mapa (hacia -y)
        # Esto es vital para la física de reflexión si usas modelos avanzados,
        # pero visualmente ayuda a entender que "mira" hacia los usuarios.
        ris_normal = np.array([0, -1, 0])
        #self.RIS = RIS(\
        #coordinate=self.data_manager.read_init_location('RIS', 0), \
        #coor_sys_z=self.data_manager.read_init_location('RIS_norm_vec', 0), \
        #ant_num=RIS_ant_num)
        self.RIS = RIS(
            coordinate=ris_pos, 
            coor_sys_z=ris_normal, 
            ant_num=RIS_ant_num
        )
        
        # ---------------------------------------------------------
        # CONFIGURACIÓN DEL HRIS (Basado en Nguyen et al., 2024)
        # --------------------------------------------------------- 
        # a. Límite de Amplificación (Sección II.A)
        # Nguyen indica que los elementos activos pueden tener una ganancia máxima (a_max)
        # de hasta 40 dB. Esto limita qué tan "fuerte" puede ser el reflejo.
        self.a_max_dB = 40  
        # Convertimos Ganancia de Potencia (dB) a Ganancia de Amplitud (Lineal):
        # 20 * log10(a_max) = 40 dB  -->  a_max = 10^(40/20) = 100
        self.a_max = 10 ** (self.a_max_dB / 20)
        # b. Presupuesto de Potencia del RIS (Sección V y Ec. 8f)
        # El HRIS tiene su propia fuente de alimentación limitada.
        # En sus simulaciones usan 0 dBm o 5 dBm. Usaremos 5 dBm para dar margen.
        self.p_max_ris_dBm = 5 
        self.p_max_ris = 10 ** (self.p_max_ris_dBm / 10) * 1e-3 # Watts (~3.1 mW)
        # c. Número de Elementos Activos (Sección V)
        # "Hybrid RIS equipped with only 4 active elements..."
        # Definimos cuántos de los 32 elementos tienen amplificador.
        self.num_active_elements = 4         
        # d. Modelo de Ruido y Auto-Interferencia (Sección II.B y V)
        # Los elementos activos introducen ruido térmico y auto-interferencia residual (SI).
        # Nguyen modela esto como un aumento en el piso de ruido.
        # "residual SI... as low as 1 dB over the noise floor" (eta = 1 dB)
        self.eta_dB = 1 
        self.eta_lin = 10 ** (self.eta_dB / 10)
        
        # Potencia de ruido efectiva en el RIS (Ec. 3):
        # sigma_r^2 = sigma_u^2 * (eta + 1)
        self.sigma_r = self.sigma_u * (self.eta_lin + 1)
        
        # ---------------------------------------------------------
        # 1.3 init users
        self.user_list = []
        for i in range(user_num):
            # Posición temporal, se sobrescribe en reset()
            user_coordinate = [0, 0, 0] 
            user = User(coordinate=user_coordinate, index=i)
            user.noise_power = self.sigma_u_dBm 
            self.user_list.append(user)
        
        # 1.6 reward design
        self.reward_design = reward_design # reward_design is [fair=fairness]

        # 1.7 step_num
        self.step_num = step_num
        # --- TDMA MEMORY ---
        # Para la Opción B: Rastrear la tasa promedio acumulada de cada usuario
        self.user_rates_accumulated = np.zeros(self.user_num)
        self.active_user_k = 0 # El usuario que tiene el turno actual
        # =============================------------------------------------------
        # 2.init channel
        # UAV -> RIS (Rx: RIS, Tx: UAV)
        self.H_UR = mmWave_channel(self.UAV, self.RIS, fre)
        self.h_U_k = []
        self.h_R_k = []
        for user_k in self.user_list:
            # UAV -> user (Rx: user, Tx: UAV)
            self.h_U_k.append(mmWave_channel(self.UAV, user_k, fre))
            # Ris -> user (Rx: user, Tx: RIS)
            self.h_R_k.append(mmWave_channel(self.RIS, user_k, fre))
        self.H_UR.update_CSI()
        for h in self.h_U_k + self.h_R_k:
            h.update_CSI()
        # 3 update user  channel capacity
        self.update_channel_capacity()

        # 4 draw system
        self.render_obj = Render(self)      
        
    def reset(self):
        """
        reset UAV, users, beamforming matrix, reflecting coefficient
        """
        # 1 reset UAV
        #self.UAV.reset(coordinate=self.data_manager.read_init_location('UAV', 0))
        #=================
        # Keeping Z height from the original file read if available, otherwise default to 100.
        # --- CORRECCIÓN AQUÍ: Usar np.array() ---
        #original_pos = self.data_manager.read_init_location('UAV', 0)
        #start_x = 0   
        #start_y = 25  
        #start_z = original_pos[2] if len(original_pos) > 2 else 100 
        
        # CAMBIO: [ ... ]  --->  np.array([ ... ])
        #   self.UAV.reset(coordinate=np.array([start_x, start_y, start_z]))

        # Esto obliga al agente a decidir si acercarse al RIS o quedarse con los usuarios cercanos.
        start_x = 0   
        start_y = 200   # Inicio del mapa
        start_z = 60 # Altura de vuelo
        
        self.UAV.reset(coordinate=np.array([start_x, start_y, start_z]))
        # ---------------------------------------
        #=================
        # 2 reset users
        fixed_positions = [
            np.array([-180, 50, 0]),   # User 0
            np.array([180, 380, 0]),   # User 1
            np.array([-150, 350, 0]),  # User 2
            np.array([50, 20, 0])      # User 3
        ]
     # Safety check for user_num
        for i in range(self.user_num):
            if i < len(fixed_positions):
                self.user_list[i].reset(coordinate=fixed_positions[i])
            else:
             # Fallback for extra users
                self.user_list[i].reset(coordinate=np.array([0, 0, 0]))
        #for i in range(self.user_num):
        #    rand_x = np.random.uniform(self.border[0][0], self.border[0][1])
        #    rand_y = np.random.uniform(self.border[1][0], self.border[1][1])
        #    rand_z = 0
        #    self.user_list[i].reset(coordinate=[rand_x, rand_y, rand_z])
        
        # 3 Reset RIS
        self.RIS.Phi = np.mat(np.diag(np.ones(self.RIS.ant_num, dtype=complex)), dtype=complex)
        
        # 4 Update CSI (CRITICAL before MRT)
        self.render_obj.t_index = 0
        self.H_UR.update_CSI()
        for h in self.h_U_k + self.h_R_k:
            h.update_CSI()
        
        # --- INICIALIZACIÓN MRT (Maximum Ratio Transmission) ---
        # G_Pmax ya está definido en __init__ como constante física (1 Watt = 30 dBm)
        # No lo recalculamos aquí; solo lo usamos como presupuesto de potencia
        power_per_user = self.UAV.G_Pmax / self.user_num
        G_init = np.zeros((self.UAV.ant_num, self.user_num), dtype=complex)
        
        for k in range(self.user_num):
            h_val = self.h_U_k[k].channel_matrix
            norm_h = np.linalg.norm(h_val)
            
            if norm_h > 1e-10:
                w_k = h_val / norm_h
            else:
                w_k = np.mat(np.ones((self.UAV.ant_num, 1), dtype=complex)) / np.sqrt(self.UAV.ant_num)
            
            # Ensure w_k is column vector
            if w_k.shape[0] == 1:
                w_k = w_k.H
            
            #G_init[:, k] = np.sqrt(power_per_user) * w_k
            G_init[:, k] = (np.sqrt(power_per_user) * w_k).flatten()
        
        self.UAV.G = np.mat(G_init, dtype=complex)
        # [INICIO BLOQUE FALTANTE]
        # ==========================================
        # FASE 2: Lógica de Selección TDMA (Opción B: Max-Min Demand)
        # ==========================================
        # Decidir quién transmite en ESTE paso basado en la historia
        
        # Si estamos al principio o hay empate de ceros, aleatorio para romper simetría
        if np.sum(self.user_rates_accumulated) == 0:
             self.active_user_k = np.random.randint(0, self.user_num)
        else:
             # El usuario con MENOR tasa acumulada tiene prioridad
             self.active_user_k = np.argmin(self.user_rates_accumulated)
        
        self.user_rates_accumulated = np.zeros(self.user_num)        
        self.active_user_k = np.random.randint(0, self.user_num) # Comenta esto       
        self.update_channel_capacity()   
        # --- TDMA MEMORY ---
        # Importante: En el paso 0, elegimos al azar para arrancar sin sesgos
        #self.active_user_k = 3 # ¡Fuerza al usuario cercano!
    
    def step(self, action_0=0, action_1=0, G=0, Phi=0, set_pos_x=0, set_pos_y=0):
        # 0. Actualizar reloj
        self.render_obj.t_index += 1
        self.current_velocity = getattr(self, 'current_velocity', 0.0)

        # 1. Movimiento Usuarios (Si aplica)
        if self.if_move_users:
            for user in self.user_list:
                user.update_coordinate(0.2, -0.5 * math.pi)

        # 2. Movimiento UAV
        if self.if_movements:
            move_x = action_0 * self.UAV.max_movement_per_time_slot
            move_y = action_1 * self.UAV.max_movement_per_time_slot
            v_t = (move_x ** 2 + move_y ** 2) ** 0.5
            self.current_velocity = v_t

            if self.reverse_x_y[0]: move_x = -move_x
            if self.reverse_x_y[1]: move_y = -move_y

            self.UAV.coordinate[0] += move_x
            self.UAV.coordinate[1] += move_y
            self.data_manager.store_data([move_x, move_y], 'UAV_movement')
        else:
            self.current_velocity = 0.0

        # 3. Actualizar Canales
        for h in self.h_U_k + self.h_R_k: h.update_CSI()
        if self.if_with_RIS: self.H_UR.update_CSI()

        # --- SELECCIÓN TDMA ---
        if np.sum(self.user_rates_accumulated) == 0:
             self.active_user_k = np.random.randint(0, self.user_num)
        else:
             self.active_user_k = np.argmin(self.user_rates_accumulated)

        # 4. Acciones Beamforming/RIS
        if isinstance(G, (list, tuple, np.ndarray)):
            self.UAV.G = convert_list_to_complex_matrix(G, (self.UAV.ant_num, self.user_num)) * math.pow(self.power_factor, 0.5)
        
        if self.if_with_RIS:
            self.RIS.Phi = convert_list_to_complex_diag(Phi, self.RIS.ant_num)
            # HRIS scaling
            N_a = min(getattr(self, 'num_active_elements', 4), self.RIS.ant_num)
            a_max = getattr(self, 'a_max', 100)
            phi_diag = np.array(np.diag(self.RIS.Phi), dtype=complex).copy()
            for i in range(N_a): phi_diag[i] *= a_max
            self.RIS.Phi = np.mat(np.diag(phi_diag))

        # 5. Física y Capacidades
        self.update_channel_capacity()

        # --- CORRECCIÓN CRÍTICA: ACUMULAR ANTES DEL REWARD ---
        # Actualizamos la memoria histórica AHORA para que el reward lo vea
        for i, user in enumerate(self.user_list):
            self.user_rates_accumulated[i] += user.capacity

        # 6. Calcular Recompensa
        reward = self.reward() # Ahora reward() verá los acumulados actualizados

        # 7. Guardar y Observar
        self.store_current_system_sate()
        new_state = self.observe()

        # 8. Boundary Check
        done = False
        x, y = self.UAV.coordinate[0:2]
        if x < self.border[0][0] or x > self.border[0][1] or \
           y < self.border[1][0] or y > self.border[1][1]:
            done = True
            reward = -10 

        self.data_manager.store_data([reward], 'reward')
        return new_state, reward, done, []

# In env.py
    def reward(self):
        # --- LÓGICA MAX-MIN PARA TDMA ---
        
        # 1. Calcular las tasas promedio HASTA AHORA (incluyendo este paso)
        # self.render_obj.t_index es el paso actual (1, 2, ... 400)
        current_step = max(1, self.render_obj.t_index)
        
        # Tasas promedio históricas
        avg_rates = self.user_rates_accumulated / current_step
        
        # El objetivo es maximizar el MÍNIMO de estos promedios
        min_avg_rate = np.min(avg_rates)
        sum_avg_rate = np.sum(avg_rates)

        # --- CÁLCULO DE ENERGÍA (Igual que antes) ---
        dt = delta_time
        v_t = getattr(self, 'current_velocity', 0.0) 
        e_fly = get_energy_consumption(v_t) 
        p_trans_watts = np.trace(self.UAV.G * self.UAV.G.H).real
        e_trans = p_trans_watts * dt
        
        # Energía RIS
        phi_vec = np.array(np.diag(self.RIS.Phi)).flatten() 
        P_STATIC_PER_ELEMENT = 0.01 
        p_ris_static = self.RIS.ant_num * P_STATIC_PER_ELEMENT
        P_IN_APPROX = 1e-8 
        AMPLIFIER_EFFICIENCY = 0.3
        p_ris_dynamic = (np.sum(np.abs(phi_vec)**2) * P_IN_APPROX) / AMPLIFIER_EFFICIENCY
        e_ris = (p_ris_static + p_ris_dynamic) * dt

        total_energy = e_fly + e_trans + e_ris
        self.total_power = total_energy / dt 

        # --- RECOMPENSA FINAL ---
        P_ref = 1500.0 
        w_min = 20.0 
        w_energy = 0.05 # Un poco más alto para que importe
        
        # Usamos min_avg_rate en lugar de min_rate instantáneo
        reward = (w_min * min_avg_rate) - (w_energy * (self.total_power / P_ref))
        
        # Clip de Seguridad
        clip_val = getattr(self, 'REWARD_MIN_CLIP', -5)
        if reward < clip_val:
            reward = clip_val
            
        return reward

     

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
        
        # 5 store G_power
        row_data = [np.trace(self.UAV.G*self.UAV.G.H), self.UAV.G_Pmax]
        self.data_manager.store_data(row_data, 'G_power')
        
        # 6. Guardar Potencia Total (Independiente)
        # Usamos getattr para evitar errores si no se ha calculado aún
        p_val = getattr(self, 'total_power', 0)
        self.data_manager.store_data([p_val], 'total_power')
        
        # ---------------------------------------------------------
        # AQUÍ BORRÉ EL BLOQUE DUPLICADO QUE CAUSABA EL ERROR
        # ---------------------------------------------------------

        # 7. Guardar Capacidad de Usuarios (LIMPIAR LISTA ANTES)
        row_data_users = [] # <--- Importante: Usar variable nueva o limpiar
        for user in self.user_list:
            row_data_users.append(user.capacity)
        
        # Validación de seguridad: Si está vacía, guardar ceros (evita el crash de plot)
        if not row_data_users:
            row_data_users = [0] * self.user_num
            
        self.data_manager.store_data(row_data_users, 'user_capacity')

        # 8. Guardar Posiciones (Tu código nuevo)
        user_coords_flat = []
        for user in self.user_list:
            user_coords_flat.extend(user.coordinate)
        self.data_manager.store_data(user_coords_flat, 'user_positions')

        # --- NUEVO: GUARDAR POSICIÓN DEL RIS ---
        # Convertimos a lista para asegurarnos de que sea serializable
        # self.RIS.coordinate es un np.array([0, 50, 10])
        ris_pos_list = list(self.RIS.coordinate)
        self.data_manager.store_data(ris_pos_list, 'RIS_position')

    def update_channel_capacity(self):
        """
        Calculates and updates each user's channel capacity and also updates
        the comprehensive_channel attribute for each user.
        """
        # 1 calculate eavesdrop rate
        # 2 calculate unsecure rate
        for user in self.user_list:
            # Calculate Capacity (using the new noise model from Step 3)
            user.capacity = self.calculate_capacity_of_user_k(user.index)
            
            # 3. Calculate secure rate (DISABLED - Not needed for Fairness)            
            # Update Channel State (REQUIRED for Agent Observation)
            user.comprehensive_channel = self.calculate_comprehensive_channel_of_user_k(user.index)


    def calculate_comprehensive_channel_of_user_k(self, k):
        """
        Calculates the combined effective channel for user k, including direct and reflected components.

        Args:
            k (int): Index of the user.

        Returns:
            np.ndarray: Combined channel vector of shape (1, M), where M is the number of UAV antennas.

        Notes:
            - h_U_k: Direct channel from UAV to user, shape (1, M).
            - h_R_k: Channel from RIS to user, shape (1, N), where N is the number of RIS elements.
            - H_UR: Channel from UAV to RIS, shape (N, M).
            - The reflected channel is computed as h_R_k @ Phi @ H_UR, resulting in shape (1, M).
            - The returned value h_combined is the sum of the direct and reflected channels, shape (1, M).
        """
        # Canales base
        h_U_k = self.h_U_k[k].channel_matrix # (1, M)
        h_R_k = self.h_R_k[k].channel_matrix # (1, N)
        H_UR  = self.H_UR.channel_matrix     # (N, M)
        
        # Canal Reflejado: h_RU * Phi * H_UR
        # Nota: Usamos multiplicación de matrices directa, es más seguro y claro.
        h_reflected = h_R_k @ self.RIS.Phi @ H_UR
        
        # Canal Total: Directo + Reflejado
        # CORRECCIÓN DE DIMENSIONES: No usar .H en h_U_k (ya es 1xM)
        h_combined = h_U_k + h_reflected
        
        return h_combined

    def calculate_capacity_of_user_k(self, k):
        """
        Calcula la capacidad bajo protocolo TDMA (Sin interferencia inter-usuario).
        Si el usuario k NO es el usuario activo, su capacidad es 0.
        """
        # 1. Regla de Oro TDMA: Si no es tu turno, no recibes nada.
        if k != self.active_user_k:
            return 0.0

        # 2. Física del Canal (Solo para el usuario activo)
        noise_power = self.user_list[k].noise_power
        
        # Obtenemos los canales combinados (ya calculados en update_channel_capacity)
        # Nota: calculate_comprehensive... se llama ANTES de esto en update_channel_capacity
        # Pero aquí recalculamos el SINR usando los punteros directos para claridad.
        h_U_k = self.h_U_k[k].channel_matrix
        h_R_k = self.h_R_k[k].channel_matrix
        
        # Canal Total = Directo + Reflejado
        H_combined = h_U_k + h_R_k @ self.RIS.Phi @ self.H_UR.channel_matrix
        
        # 3. Beamforming: Solo nos importa la columna k de la matriz G
        # (El agente sigue generando G completa, pero físicamente solo "apuntamos" una columna)
        G_k = self.UAV.G[:, k]
        
        # 4. Potencia de Señal
        signal = (np.abs(H_combined @ G_k)**2).item()
        
        # 5. SINR -> SNR (En TDMA la interferencia de otros usuarios es 0)
        interference = 0 
        
        snr = signal / (dB_to_normal(noise_power)) # Sin interferencia
        
        # 6. Capacidad de Shannon
        return math.log2(1 + snr)
            
        

    def calculate_secure_capacity_of_user_k(self, k):
        # Not used when optimizing fairness; return 0 to keep compatibility
        return 0.0

    def get_system_action_dim(self):
        """
        function used in main function to get the dimension of actions
        CORRECTED: Must return Movement + RIS + Beamforming dimensions
        """
        result = 0
        # 0 UAV movement (x, y)
        result += 2
        
        # 1 RIS reflecting elements (Phase shifts)
        if self.if_with_RIS:
            result += self.RIS.ant_num   
        
        # 2 beamforming matrix dimension (Real + Imag parts), i.e., 2 * UAV_ant_num * user_num
        # Each complex entry in the beamforming matrix is split into real and imaginary parts
        result += 2 * self.UAV.ant_num * self.user_num 
        return result
# ... (other methods like get_system_state_dim) ...

    def observe(self):
        """
        Observación Normalizada para DRL estable.
        """
        # 1. Información de Canal (Escalada)
        # Multiplicamos por 1 millón para que los valores estén en rango [-1, 1] aprox
        CHANNEL_SCALE = 1e6 
        
        comprehensive_channel_elements_list = [] 
        targets = self.user_list 
        
        for entity in targets:
            # Obtenemos el vector complejo
            raw_channel = np.array(np.reshape(entity.comprehensive_channel, (1,-1)))[0]
            
            # Escalamos
            scaled_channel = raw_channel * CHANNEL_SCALE
            
            # Separamos Real e Imaginario
            tmp_list = list(scaled_channel)
            comprehensive_channel_elements_list += list(np.real(tmp_list)) + list(np.imag(tmp_list)) 
        
        # 2. Información de Posición (Normalizada)
        # Mapa X: [-200, 200] -> Dividir por 200
        # Mapa Y: [0, 400]    -> Dividir por 400
        # Mapa Z: [0, 100]    -> Dividir por 100
        UAV_position_list = []
        if self.if_UAV_pos_state:
            x, y, z = self.UAV.coordinate
            norm_x = x / 200.0
            norm_y = y / 400.0
            norm_z = z / 100.0
            UAV_position_list = [norm_x, norm_y, norm_z]

        # 3. Información de Tasas (Ya suelen estar entre 0 y 10, ok)
        rates_list = []
        for user in self.user_list:
            val = user.capacity if user.capacity is not None else 0.0
            rates_list.append(val)

        # Retornamos la lista plana
        return comprehensive_channel_elements_list + UAV_position_list + rates_list
        
    def get_system_state_dim(self):
        """
        function used in main function to get the dimention of states
        """
        # users' comprehensive channel (real + imag parts)
        state_dim = 2 * (self.user_num) * self.UAV.ant_num
        
        # UAV position
        if self.if_UAV_pos_state:
            state_dim += 3
            
        # NEW: Add dimension for user rates
        state_dim += self.user_num 
        
        return state_dim

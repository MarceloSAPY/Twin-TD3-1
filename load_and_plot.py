import matplotlib.pyplot as plt
import numpy as np
import cmath
from scipy.io import loadmat
import pandas as pd
import os
import copy
import math
import argparse

# --- CONFIGURACIÓN DE ARGUMENTOS ---
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='Ruta a la carpeta de datos (ej: ./data/storage/scratch/td3_fair)')
parser.add_argument('--ep-num', type=int, default=500, help='Número total de episodios a procesar')
parser.add_argument('--step-num', type=int, default=400, help='Pasos por episodio')
args = parser.parse_args()

STORE_PATH = args.path
EP_NUM = args.ep_num
STEP_NUM = args.step_num
USER_NUM = 4 # Fijo para Hell Scenario

# Parámetros de Energía (Para cálculo post-mortem si es necesario)
delta_time = 0.1

class LoadAndPlot(object):
    def __init__(self, store_path, ep_num):
        self.store_path = store_path + '/' if not store_path.endswith('/') else store_path
        self.ep_num = ep_num
        self.user_num = USER_NUM
        
        # Colores para distinguir usuarios
        self.color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
        
        # Cargar datos
        self.all_steps = self.load_all_steps()

    def load_one_ep(self, file_name):
        try:
            return loadmat(self.store_path + file_name)
        except Exception as e:
            # print(f"Error cargando {file_name}: {e}")
            return None

    def load_all_steps(self):
        print(f"Cargando datos de {self.ep_num} episodios...")
        
        # Estructura de datos limpia
        data = {
            'reward': [],
            'user_capacity': [[] for _ in range(self.user_num)],
            'uav_energy': [],
            'min_rate': [],  # Nueva métrica clave para Fairness
            'sum_rate': [],
            'fairness_index': [] # Jain's Fairness Index
        }

        valid_eps = 0
        for ep in range(self.ep_num):
            filename = f"simulation_result_ep_{ep}.mat"
            mat = self.load_one_ep(filename)
            
            if mat is None: continue
            valid_eps += 1
            
            # Extraer key base (ej: result_0, result_1...)
            key = f"result_{ep}"
            if key not in mat: continue
            
            # 1. Reward
            r = mat[key]["reward"][0][0].flatten()
            data['reward'].extend(r)
            
            # 2. User Capacities
            # user_capacity suele venir como (steps, users)
            caps = mat[key]["user_capacity"][0][0] 
            
            # Calcular métricas paso a paso
            for t in range(caps.shape[0]):
                step_rates = caps[t, :]
                
                # Guardar individuales
                for u in range(self.user_num):
                    if u < caps.shape[1]:
                        data['user_capacity'][u].append(step_rates[u])
                    else:
                        data['user_capacity'][u].append(0)
                
                # Calcular Fairness Metrics Instantáneas
                data['sum_rate'].append(np.sum(step_rates))
                data['min_rate'].append(np.min(step_rates))
                
                # Jain's Fairness Index: (Sum x)^2 / (n * Sum x^2)
                sq_sum = np.sum(step_rates ** 2)
                if sq_sum > 0:
                    jains = (np.sum(step_rates) ** 2) / (self.user_num * sq_sum)
                else:
                    jains = 0
                data['fairness_index'].append(jains)

        print(f"Datos cargados de {valid_eps} episodios válidos.")
        return data

    def smooth(self, data, window=500):
        """Suavizado para gráficos legibles"""
        if len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def plot(self):
        plot_dir = self.store_path + 'plots/'
        if not os.path.exists(plot_dir): os.makedirs(plot_dir)
        
        print("Generando gráficos...")

        # 1. REWARD
        plt.figure(figsize=(10,6))
        plt.plot(self.all_steps['reward'], alpha=0.2, color='gray', label='Raw')
        plt.plot(self.smooth(self.all_steps['reward']), color='blue', linewidth=2, label='Smoothed')
        plt.title('Training Reward Evolution')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(plot_dir + 'reward.png')
        plt.close()

        # 2. FAIRNESS INDEX (Jain's) - CRÍTICO PARA TU TESIS
        plt.figure(figsize=(10,6))
        plt.plot(self.smooth(self.all_steps['fairness_index'], window=1000), color='purple', linewidth=2)
        plt.title("Jain's Fairness Index (Moving Average)")
        plt.xlabel('Steps')
        plt.ylabel('Index (0 to 1)')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_dir + 'fairness_index.png')
        plt.close()

        # 3. MINIMUM RATE (Lo que intentamos maximizar)
        plt.figure(figsize=(10,6))
        plt.plot(self.smooth(self.all_steps['min_rate'], window=1000), color='red', linewidth=2)
        plt.title("Minimum User Rate (Moving Average)")
        plt.xlabel('Steps')
        plt.ylabel('Rate (bps/Hz)')
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_dir + 'min_rate.png')
        plt.close()

        # 4. TRAJECTORY (HELL SCENARIO 400x400)
        self.plot_trajectory(plot_dir)

    def plot_trajectory(self, save_path):
        # Definir posiciones FIJAS del Hell Scenario (No leer de Excel)
        fixed_users = [
            [-180, 50],   # User 0
            [180, 380],   # User 1
            [-150, 350],  # User 2
            [50, 20]      # User 3
        ]
        
        ris_pos = [0, 400]

        plt.figure(figsize=(8,8))
        
        # Plotear Usuarios
        for i, u in enumerate(fixed_users):
            plt.scatter(u[0], u[1], c='red', s=100, marker='o', label=f'User {i}' if i==0 else "")
            plt.text(u[0]+10, u[1], f"U{i}", fontsize=12, fontweight='bold')

        # Plotear RIS
        plt.scatter(ris_pos[0], ris_pos[1], c='green', s=150, marker='^', label='RIS')
        plt.text(ris_pos[0], ris_pos[1]-20, "RIS", color='green', fontweight='bold', ha='center')

        # Plotear Trayectorias (Seleccionadas: Inicio, Medio, Fin)
        # Intervalo para no saturar el gráfico
        episodes_to_plot = [0, int(self.ep_num*0.5), self.ep_num-1]
        colors = ['gray', 'orange', 'blue']
        labels = ['Start (Ep 0)', 'Mid', 'Final']

        for i, ep in enumerate(episodes_to_plot):
            filename = f"simulation_result_ep_{ep}.mat"
            mat = self.load_one_ep(filename)
            if mat is None: continue
            
            # Reconstruir trayectoria
            # Asumimos que guardaste 'UAV_state' o 'UAV_movement'
            # Intentamos leer UAV_state directo (posición absoluta)
            try:
                # Estructura usual: [x, y, z]
                # data_manager guarda lista plana, hay que tener cuidado
                # Si guardaste 'UAV_state' en data_manager, es lo mejor.
                # Si no, reconstruimos desde movements.
                
                # Opción A: Reconstrucción (Más segura si env.py guarda delta)
                key = f"result_{ep}"
                moves = mat[key]["UAV_movement"][0][0] # [[dx, dy], [dx, dy]...]
                
                path_x = [0] # Inicio en 0,0 (Hell Scenario reset)
                path_y = [0]
                
                for step in range(moves.shape[0]):
                    path_x.append(path_x[-1] + moves[step, 0])
                    path_y.append(path_y[-1] + moves[step, 1])
                
                plt.plot(path_x, path_y, color=colors[i], linewidth=2, label=labels[i])
                # Marcar fin
                plt.scatter(path_x[-1], path_y[-1], color=colors[i], marker='x')

            except Exception as e:
                print(f"No se pudo plotear trayectoria Ep {ep}: {e}")

        # Configuración del Mapa
        plt.xlim(-220, 220)
        plt.ylim(-20, 420)
        plt.title(f"UAV Trajectory Evolution (Hell Scenario 400x400)\nEpisodes: {self.ep_num}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.grid(True, linestyle='--')
        plt.legend()
        
        plt.savefig(save_path + 'trajectory_hell_scenario.png')
        plt.close()
        print("Gráficos generados en:", save_path)

if __name__ == '__main__':
    lp = LoadAndPlot(STORE_PATH, EP_NUM)
    lp.plot()
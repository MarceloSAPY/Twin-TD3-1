import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pandas as pd
import os
import copy
import math

# --- Funciones auxiliares ---
init_data_file = 'data/init_location.xlsx'

def read_init_location(entity_type='user', index=0):
    try:
        df = pd.read_excel(init_data_file, sheet_name=entity_type)
        if index < len(df):
            return np.array([df['x'][index], df['y'][index], df['z'][index]])
    except Exception as e:
        pass # Si falla, devolvemos ceros
    return np.zeros(3)

class LoadAndPlot(object):
    def __init__(self, store_paths, user_num=4, attacker_num=0, RIS_ant_num=32, ep_num=300, step_num=100):
        self.store_paths = store_paths
        self.user_num = user_num
        self.attacker_num = attacker_num
        self.RIS_ant_num = RIS_ant_num
        self.ep_num = ep_num
        self.step_num = step_num

    def plot(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        MARKER_SIZE = 8
        
        # Colores para las trayectorias
        color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # --- 1. DIBUJAR UAV INICIAL ---
        init_uav_coord = read_init_location(entity_type='UAV')
        # Nota: Invertimos ejes [1] es Y, [0] es X para el plot
        plt.plot([init_uav_coord[1]], [init_uav_coord[0]], marker="s", markersize=MARKER_SIZE, 
                 markeredgecolor="black", markerfacecolor="none", label='UAV Inicio')
        plt.text(init_uav_coord[1] + 1, init_uav_coord[0], 'UAV', fontsize=9)

        # --- 2. DIBUJAR RIS ---
        init_ris_coord = read_init_location(entity_type='RIS')
        plt.plot([init_ris_coord[1]], [init_ris_coord[0]], marker="d", markersize=MARKER_SIZE + 2, 
                 markeredgecolor="black", markerfacecolor="gold", label='RIS')
        plt.text(init_ris_coord[1], init_ris_coord[0] - 3, 'RIS', fontsize=11, fontweight='bold', ha='center')

        # --- 3. DIBUJAR TRAYECTORIAS (UAV) ---
        legends = [f'Experimento {i+1}' for i in range(len(self.store_paths))]
        
        for idx, (store_path, legend) in enumerate(zip(self.store_paths, legends)):
            # Intentar leer el último episodio disponible
            ep_idx = self.ep_num - 1
            filename = f'simulation_result_ep_{ep_idx}.mat'
            full_path = os.path.join(store_path, filename)
            
            if not os.path.exists(full_path):
                print(f"Archivo no encontrado: {full_path}. Saltando...")
                continue
                
            try:
                data = loadmat(full_path)
                uav_coord = [[init_uav_coord[0]], [init_uav_coord[1]]]
                uav_movt = data[f'result_{ep_idx}'][0][0][-1]
                
                # Reconstruir movimiento
                for j in range(uav_movt.shape[0]):
                    move_x = uav_movt[j][0]
                    move_y = uav_movt[j][1]
                    uav_coord[0].append(uav_coord[0][-1] + move_x)
                    uav_coord[1].append(uav_coord[1][-1] + move_y)
                
                c = color_list[idx % len(color_list)]
                plt.plot(uav_coord[1], uav_coord[0], c=c, label=legend, linewidth=2, alpha=0.8)
                
            except Exception as e:
                print(f"Error leyendo datos de trayectoria: {e}")

        # --- 4. DIBUJAR USUARIOS ---
        # Límites para generar posiciones aleatorias si faltan en el Excel
        # Coinciden con los bordes de env.py: x(-200,200), y(0,400)
        # Fixed positions from env.py (lines 239-244):
        # User 0: [-180, 50, 0]   - Far Left, Low
        # User 1: [180, 380, 0]   - Far Right, High (Near RIS)
        # User 2: [-150, 350, 0]  - Far Left, High
        # User 3: [50, 20, 0]     - Center, Low
        x_min, x_max = -200, 200
        y_min, y_max = 0, 400
        
        # Semilla fija para que los usuarios no "bailen" cada vez que sacas la gráfica
        np.random.seed(42) 

        for k in range(self.user_num):
            u_coord = read_init_location(entity_type='user', index=k)
            
            # LA CORRECCIÓN: 
            # Si u_coord es [0,0,0] (porque no estaba en el Excel), generamos uno aleatorio
            # Esto es solo visualización para representar la aleatoriedad del entrenamiento
            if np.all(u_coord == 0):
                # Generar posición dummy distintiva
                rand_x = np.random.uniform(x_min, x_max)
                rand_y = np.random.uniform(y_min, y_max)
                u_coord = np.array([rand_x, rand_y, 0])
            
            # Graficar Usuario
            # Usamos colores distintos para cada usuario para identificarlos fácil
            user_color = plt.cm.tab10(k) 
            plt.plot(u_coord[1], u_coord[0], marker="o", markersize=MARKER_SIZE, 
                     markeredgecolor="black", markerfacecolor=user_color, linestyle='None')
            
            # Etiqueta U1, U2, etc.
            plt.text(u_coord[1] + 1, u_coord[0] + 1, f'U{k+1}', fontsize=10, fontweight='bold', color=user_color)

        # --- Configuración Final del Gráfico ---
        plt.legend(loc='upper right', fontsize=8, framealpha=0.9)
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        
        # Ajustar límites según env.py border: x(-200,200), y(0,400)
        plt.xlim(-50, 450)  # y-axis in plot (border[1])
        plt.ylim(-220, 220) # x-axis in plot (border[0]) 
        
        plt.xlabel('y (m)') 
        plt.ylabel('x (m)')
        plt.title(f'Trayectoria UAV y Posición de {self.user_num} Usuarios\n(HRIS Energy Efficient)')
        
        # Invertir eje Y (convención común en estos mapas)
        plt.gca().invert_yaxis()
        
        output_file = 'data/trajectory_updated.png'
        plt.savefig(output_file)
        print(f"Gráfico guardado exitosamente en: {output_file}")
        # plt.show() # Descomentar si quieres verla al momento

if __name__ == '__main__':
    # Ruta de tu experimento actual
    current_experiment = 'data/storage/scratch/td3_fair_HRIS_Energy'
    
    LoadPlotObject = LoadAndPlot(
        store_paths=[current_experiment],
        user_num=4,          # 4 Usuarios
        RIS_ant_num=32,
        ep_num=500           # Ajusta si entrenaste menos episodios
    )
    LoadPlotObject.plot()
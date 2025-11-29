import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import argparse
import os
import math

# --- CONFIGURACIÓN DE ARGUMENTOS ---
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='Ruta a la carpeta de datos')
parser.add_argument('--ep-num', type=int, default=300, help='Número de episodios a procesar')
parser.add_argument('--user-num', type=int, default=4, help='Número de usuarios')
args = parser.parse_args()

class LoadAndPlot(object):
    def __init__(self, store_path, user_num=4, ep_num=300):
        self.store_path = store_path + '/' if not store_path.endswith('/') else store_path
        self.user_num = user_num
        self.ep_num = ep_num
        self.all_steps = self.load_all_steps()

    def load_one_ep(self, file_name):
        try:
            return loadmat(self.store_path + file_name)
        except:
            return None

    def load_all_steps(self):
        # Estructura segura
        data = {
            'reward': [],
            'sum_rate': [],
            'min_rate': [],
            'total_power': []
        }

        print(f"Leyendo {self.ep_num} episodios desde {self.store_path}...")
        valid_count = 0

        for i in range(self.ep_num):
            mat = self.load_one_ep(f"simulation_result_ep_{i}.mat")
            if mat is None: continue
            
            try:
                # Intentar leer la clave del episodio (ej: result_0)
                res_key = f"result_{i}"
                if res_key not in mat: continue
                
                res = mat[res_key]
                
                # 1. Cargar Reward (Si falla, ignora este episodio)
                if 'reward' in res.dtype.names:
                    data['reward'].extend(res["reward"][0][0].flatten())
                
                # 2. Cargar Power (Opcional)
                if 'total_power' in res.dtype.names:
                    data['total_power'].extend(res["total_power"][0][0].flatten())
                
                # 3. Cargar Rates (Manejo de errores de índice)
                if 'user_capacity' in res.dtype.names:
                    caps = res["user_capacity"][0][0]
                    # Verificar que tenga el tamaño correcto antes de leer
                    if caps.shape[1] >= self.user_num:
                        sum_r = np.sum(caps, axis=1)
                        min_r = np.min(caps, axis=1)
                        data['sum_rate'].extend(sum_r)
                        data['min_rate'].extend(min_r)
                    
                valid_count += 1
            except Exception as e:
                # Si un episodio está corrupto, lo saltamos sin crashear todo el programa
                # print(f"Saltando Ep {i}: {e}") 
                pass

        print(f"Éxito: {valid_count} episodios válidos procesados.")
        return data

    def plot_metric(self, data_key, title, ylabel, filename, window=100):
        data = self.all_steps[data_key]
        if len(data) == 0:
            print(f"Aviso: No hay datos para {filename}")
            return

        plt.figure(figsize=(8, 5))
        plt.plot(data, alpha=0.3, color='gray', label='Crudo')
        
        if len(data) > window:
            smooth = np.convolve(data, np.ones(window)/window, mode='valid')
            plt.plot(smooth, linewidth=2, color='blue', label='Promedio')
            
        plt.title(title)
        plt.xlabel("Pasos de Entrenamiento")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(self.store_path + filename)
        print(f"Gráfica guardada: {filename}")
        plt.close()

    def plot(self):
        self.plot_metric('reward', 'Recompensa', 'Reward', 'reward_plot.png')
        self.plot_metric('sum_rate', 'Tasa Suma', 'Bits/s/Hz', 'sum_rate_plot.png')
        self.plot_metric('min_rate', 'Min Rate (Equidad)', 'Bits/s/Hz', 'min_rate_fairness.png')
        self.plot_metric('total_power', 'Consumo Energía', 'Joules/Step', 'power_plot.png')

if __name__ == '__main__':
    plotter = LoadAndPlot(args.path, user_num=args.user_num, ep_num=args.ep_num)
    plotter.plot()
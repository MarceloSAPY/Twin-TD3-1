import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import argparse
import os
import cmath

# Configuración
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='Ruta a la carpeta de datos')
parser.add_argument('--ep', type=int, default=499, help='Episodio a analizar (por defecto el último)')
args = parser.parse_args()

def analyze_ris(path, episode_idx):
    filename = f"simulation_result_ep_{episode_idx}.mat"
    full_path = os.path.join(path, filename)
    
    if not os.path.exists(full_path):
        print(f"Error: No se encuentra {full_path}")
        return

    try:
        data = loadmat(full_path)
        key = f"result_{episode_idx}"
        
        # Cargar coeficientes del RIS
        # Forma usual en tu data_manager: Lista de complejos aplanada por paso
        ris_data_raw = data[key]["reflecting_coefficient"][0][0] # (steps, RIS_elements)
        
        steps = ris_data_raw.shape[0]
        elements = ris_data_raw.shape[1]
        
        print(f"Analizando Episodio {episode_idx}: {steps} pasos, {elements} elementos RIS.")

        # Seleccionar pasos clave: Inicio, Medio, Fin
        step_indices = [0, int(steps/2), steps-1]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': 'polar'})
        
        for idx, step in enumerate(step_indices):
            coeffs = ris_data_raw[step, :]
            
            # Extraer magnitud y fase
            magnitudes = np.abs(coeffs)
            phases = np.angle(coeffs)
            
            ax = axes[idx]
            
            # Plotear Fases y Magnitudes
            # Puntos azules: Pasivos (Magnitud ~1)
            # Puntos Rojos: Activos (Magnitud > 1)
            colors = ['red' if m > 1.1 else 'blue' for m in magnitudes]
            sizes = [50 if m > 1.1 else 20 for m in magnitudes]
            
            ax.scatter(phases, magnitudes, c=colors, s=sizes, alpha=0.75)
            ax.set_title(f"Step {step}\n(Red=Active, Blue=Passive)")
            ax.set_ylim(0, max(magnitudes)*1.1)
            
        plt.suptitle(f"RIS Configuration Evolution - Episode {episode_idx}", fontsize=16)
        save_file = os.path.join(path, f'plots/ris_analysis_ep{episode_idx}.png')
        plt.savefig(save_file)
        print(f"Gráfico guardado en: {save_file}")
        plt.close()

        # Análisis de Magnitud Promedio (Para verificar consumo energético)
        avg_mag = np.mean(np.abs(ris_data_raw), axis=1)
        plt.figure(figsize=(10, 5))
        plt.plot(avg_mag)
        plt.title("Promedio de Magnitud de Coeficientes RIS (Energía)")
        plt.xlabel("Step")
        plt.ylabel("Magnitud Promedio")
        plt.grid(True)
        plt.savefig(os.path.join(path, f'plots/ris_magnitude_ep{episode_idx}.png'))
        plt.close()

    except Exception as e:
        print(f"Error analizando RIS: {e}")

if __name__ == "__main__":
    analyze_ris(args.path, args.ep)
# test_crash.py
import numpy as np
from env import MiniSystem

print("--- Iniciando Test de Integridad de env.py ---")

try:
    # Intentamos crear el entorno. 
    # Si faltan variables iniciales, esto fallará inmediatamente.
    system = MiniSystem(user_num=4, RIS_ant_num=32, UAV_ant_num=4)
    print("✅ ÉXITO: El entorno se inicializó correctamente.")
    
    # Verificamos si G existe y tiene la forma correcta
    print(f"Estado de G: {system.UAV.G.shape}")
    print(f"Potencia Max (G_Pmax): {system.UAV.G_Pmax}")

except AttributeError as e:
    print(f"❌ FALLO CRÍTICO: Falta un atributo necesario.\nError: {e}")
except Exception as e:
    print(f"❌ FALLO INESPERADO: {e}")
import numpy as np
from env import MiniSystem

# 1. Iniciar el entorno
env = MiniSystem(user_num=4)
env.reset()

# 2. Imprimir distancias y capacidades iniciales
print("--- DIAGNÓSTICO DE FÍSICA (Hell Scenario) ---")
uav_pos = env.UAV.coordinate
print(f"Posición UAV: {uav_pos}")

for i, user in enumerate(env.user_list):
    # Calcular distancia real
    dist = np.linalg.norm(user.coordinate - uav_pos)
    
    # Capacidad actual (debería ser muy baja para los usuarios lejanos)
    cap = user.capacity
    
    print(f"\nUsuario {i}:")
    print(f"  Posición: {user.coordinate}")
    print(f"  Distancia al UAV: {dist:.2f} m")
    print(f"  Capacidad Inicial: {cap:.5f} bps/Hz")
    
    # Check de cordura
    if dist > 300 and cap > 1.0:
        print("  [ALERTA] Capacidad sospechosamente alta para esta distancia. Revisa potencia de ruido.")
    elif dist > 300 and cap < 0.1:
        print("  [OK] La señal está muriendo por la distancia. El agente necesitará moverse.")
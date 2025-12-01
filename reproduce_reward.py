
import numpy as np
from env import MiniSystem

def test_reward():
    env = MiniSystem(user_num=4, step_num=100)
    env.reset()
    
    print(f"{'Step':<5} | {'Active':<6} | {'MinAvgRate':<12} | {'Power':<10} | {'Reward':<10} | {'EnergyTerm':<10} | {'RateTerm':<10}")
    print("-" * 80)
    
    for i in range(10):
        # Random action
        action_0 = 0 # No movement
        action_1 = 0
        Phi = [0] * env.RIS.ant_num
        
        # Step
        # Note: env.step signature in env.py: step(self, action_0=0, action_1=0, G=0, Phi=0, set_pos_x=0, set_pos_y=0)
        # But main_train.py passes named args.
        
        new_state, reward, done, info = env.step(action_0=action_0, action_1=action_1, Phi=Phi)
        
        # Calculate terms manually to verify
        current_step = max(1, env.render_obj.t_index)
        avg_rates = env.user_rates_accumulated / current_step
        min_avg_rate = np.min(avg_rates)
        
        P_ref = 1500.0
        w_min = 20.0
        w_energy = 0.005
        
        rate_term = w_min * min_avg_rate
        energy_term = w_energy * (env.total_power / P_ref)
        
        print(f"{i:<5} | {env.active_user_k:<6} | {min_avg_rate:<12.6f} | {env.total_power:<10.2f} | {reward:<10.6f} | {energy_term:<10.6f} | {rate_term:<10.6f}")

if __name__ == "__main__":
    test_reward()

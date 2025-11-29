import numpy as np
from env import MiniSystem

def run_check():
    sys = MiniSystem(user_num=4)
    print("UAV ant num:", sys.UAV.ant_num)
    print("User num:", sys.user_num)
    print("p_uav_dBm:", getattr(sys, 'p_uav_dBm', None))
    print("p_uav_watts:", getattr(sys, 'p_uav_watts', None))
    print("power_factor:", getattr(sys, 'power_factor', None))
    print("UAV.G shape:", sys.UAV.G.shape)
    print("UAV.G dtype:", sys.UAV.G.dtype)
    print("UAV.G_Pmax:", getattr(sys.UAV, 'G_Pmax', None))
    # Basic consistency checks
    assert sys.UAV.G.shape == (sys.UAV.ant_num, sys.user_num)
    assert getattr(sys, 'p_uav_watts', 0) == getattr(sys.UAV, 'G_Pmax', getattr(sys, 'power_factor', None)) or getattr(sys, 'power_factor', None) is not None
    print("Sanity checks passed")

if __name__ == "__main__":
    run_check()
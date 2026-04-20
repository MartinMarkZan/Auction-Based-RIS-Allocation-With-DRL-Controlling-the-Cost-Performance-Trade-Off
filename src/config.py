from typing import Optional
import numpy as np

class Config:
    # ---- Reproducibility ----
    seed: int = 2

    # ---- Topology / geometry ----
    N_OP: int = 1       # number of operators
    N_UE: int = 20      # number of users
    N_RIS: int = 10     # number of reconfigurable intelligent surfaces
    N_BS: int = 2       # number of base stations
    M_UE: int = 1       # number of antennas at UEs
    M_RIS: int = 250    # number of RIS elements
    M_BS: int = 50      # number of antennas at BS
    ROI_size: np.ndarray = np.array([[-10, -10], [10, 10]]) * 5  # meters size of ROI

    # ---- Reinforcement learning ----
    beta: float = 15.0           # bid intensity
    util_alpha: float = 0.0     # alpha value of alpha-fair utility function
    model_name: str = f"2025_09_04_beta{beta}"   # name of the RL model

    # ---- Auction / economics ----
    start_price: float = 0.05            # price at beginning of auction
    increment: float = 0.05              # price increment from one round to the next
    budget: np.array = 1 * np.ones(N_BS) # available budget for bidding (can be different for operators)

    # ---- Training (script-level) ----
    num_vec_envs: int = 4               # number of vector environments
    n_steps: int = 2048                 # number of steps for each environment per update
    timesteps: int = int(3e6)           # total number of steps to train on

    # ---- Testing (script-level) ----
    budget_test: bool = True        # test different budgets
    performance_test: bool = True   # test performance
    eval_episodes: int = 200        # number of macroscopic fading realizations for evaluation
    NN: int = 20                    # number of microscopic fading realizations

    # ---- Channel ----
    K: np.array = np.array([1e2, 3*1e0]) # Rician K-factor for LOS and NLOS
    N0: float = -174.0       # dBm/Hz noise PSD
    F: float = 6             # dB noise figure
    Bs: float = 15e3         # subcarrier bandwidth
    Ps: float = 20.0         # dBm power per subcarrier
    fc: float = 26e9         # carrier frequency
    sf: float = 10           # shadow fading variance

    # ---- Rendering ----
    show_plot: bool = False

    # ---- Derived (computed from others) ----
    ps_lin: Optional[float] = 10**((Ps - 30) / 10)              # linear power
    lam: float = 3e8 / fc                                       # wavelength
    sigma_n2: float = 10**((N0 + 10*np.log10(Bs) + F - 30)/10)  # linear noise power
    f_name = str(N_BS) + 'BS_' + str(N_UE) + 'UE_' + str(N_RIS) + 'RIS_' + str(M_RIS) + 'M' # file name
    
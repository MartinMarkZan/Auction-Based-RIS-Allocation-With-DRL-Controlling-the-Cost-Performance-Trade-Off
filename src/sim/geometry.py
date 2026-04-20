import warnings
import numpy as np


def regular_noisy_placement(ndim: int, 
                            n_required_positions: int, 
                            ROI_size: np.ndarray, 
                            noise_var: float, 
                            rng: np.random.Generator,
    ) -> np.ndarray:
    """
    Regular placement within a region of interest with noise on top to observe different positions.

    Args:
        n_required_positions: Number of required positions per OP.
        ndim: Number of dimensions (not tested for more than 2).
        ROI_size (2, ndim): Region of interest array with min and max coordinates.
        noise_var: Variance of Gaussian perturbation.
        rng: Random number generator.

    Returns:
        rand_pos (n_required_positions, ndim): Array of generated positions.
    """
    dim_div = np.ceil(n_required_positions**(1 / ndim))  # number of positions per dimension
    dim_dist = (ROI_size[1] - ROI_size[0]) / dim_div  # distance between regular positions
    reg_grid = np.linspace(ROI_size[0] + dim_dist / 2, ROI_size[1] - dim_dist / 2, dim_div)  # grid of positions
    n_possible_positions = dim_div**ndim  # total number of possible positions
    possible_pos = np.array(np.meshgrid(reg_grid[:, 0], reg_grid[:, 1])).T.reshape(-1, 2)
    pos_choice = rng.choice(n_possible_positions, n_required_positions, replace=False)
    chosen_pos = possible_pos[pos_choice, :]
    rand_pos = chosen_pos + rng.normal(0, np.sqrt(noise_var), chosen_pos.shape)
    return rand_pos


def gen_POS(N_OP: int, 
            N_BS: int, 
            N_UE: int, 
            N_RIS: int, 
            ROI_size: np.ndarray, 
            rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate positions of UEs, BSs, and RISs. UE positions are uniformly random, 
    BS and RIS positions are regular with noise.

    Args:
        N_OP: Number of operators.
        N_BS: Number of BSs per operator.
        N_UE: Number of UEs per operator.
        N_RIS: Number of RISs.
        ROI_size (2, ndim): Region of interest array with min and max coordinates.
        rng: Random number generator.
        
    Returns:
        UE_pos (N_OP, N_UE, 2): Array of user positions for each operator.
        BS_pos (N_OP, N_BS, 2): Array of base station positions for each operator.
        RIS_pos (N_RIS, 2): Array of RIS positions.
    """
    UE_pos = rng.uniform(ROI_size[0], ROI_size[1], size=(N_OP, N_UE, 2))

    # Generate random base station positions for each operator.
    pos_noise_var = ((np.amax(ROI_size) - np.amin(ROI_size)) / 5)**2
    BS_pos = []
    for _ in range(N_OP):
        BS_pos.append(regular_noisy_placement(2, N_BS, ROI_size, pos_noise_var, rng))

    pos_noise_var = ((np.amax(ROI_size) - np.amin(ROI_size)) / 4)**2
    RIS_pos = regular_noisy_placement(2, N_RIS, ROI_size, pos_noise_var, rng)

    return UE_pos, BS_pos, RIS_pos


def gen_POS_cell_edge(N_OP: int, 
                      N_BS: int, 
                      N_UE: int, 
                      N_RIS: int, 
                      ROI_size: np.ndarray, 
                      rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate positions of UEs, BSs, and RISs. UE and RIS positions are 
    correlated random, BS positions are at the corners of the ROI.

    Args:
        N_OP: Number of operators.
        N_BS: Number of BSs per operator.
        N_UE: Number of UEs per operator.
        N_RIS: Number of RISs.
        ROI_size (2, ndim): Region of interest array with min and max coordinates.
        rng: Random number generator.
        
    Returns:
        UE_pos (N_OP, N_UE, 2): Array of user positions for each operator.
        BS_pos (N_OP, N_BS, 2): Array of base station positions for each operator.
        RIS_pos (N_RIS, 2): Array of RIS positions.
    """
    if N_BS != 2:
        # The problem is with the BS positions. 
        # ROI_size is 2x2 array, so we can only place two BSs at the corners.
        raise ValueError(f"Only implemented for two base stations.")
    if N_OP != 1:
        warnings.warn(f"BSs for different operators will be at the same position.")
    
    pos_noise_var = ((np.amax(ROI_size) - np.amin(ROI_size)) / 4)**2
    cross_corr = -0.75
    covariance = np.array([[1.0, cross_corr], [cross_corr, 1]]) * pos_noise_var
    UE_pos = rng.multivariate_normal(np.array([0,0]), covariance, size=(N_OP, N_UE))

    # Put BSs at the corners of the ROI.
    BS_pos = []
    for _ in range(N_OP):
        BS_pos.append(ROI_size)
    BS_pos = np.array(BS_pos)

    pos_noise_var = ((np.amax(ROI_size) - np.amin(ROI_size)) / 4)**2
    cross_corr = -0.9
    covariance = np.array([[1.0, cross_corr],[cross_corr, 1]]) * pos_noise_var
    RIS_pos = rng.multivariate_normal(np.array([0,0]), covariance, size=N_RIS)

    return UE_pos, BS_pos, RIS_pos

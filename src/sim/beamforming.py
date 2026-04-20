import numpy as np

from src.envs.ris_env import RISAuctionEnv
from src.sim.channels import Gauss_channel


def compute_beamforming_vector(env: RISAuctionEnv, BS_RIS_alloc: np.ndarray
    ) -> np.ndarray:
    """
    Compute the beamforming vectors for a given RIS allocation.

    Args:
        env: The environment object with BS_RIS_channel and other necessary data.
        BS_RIS_alloc (N_RIS,): array assigning each RIS to a base station.

    Returns:
        beamforming_vector (M_BS, 1, N_BS, N_OP, NN): complex array.
    """
    beamforming_vector = np.zeros((env.M_BS, 1, env.N_BS, env.N_OP, env.NN), dtype=np.complex128)

    for nb in range(env.N_BS):
        # Indices of RISs assigned to this BS.
        ris_idxs = np.where(BS_RIS_alloc == nb)[0]
        
        if ris_idxs.size == 0:
            # No RIS: random Gaussian beamforming.
            single_beamformer = Gauss_channel((env.M_BS, 1, env.N_OP, env.NN), env.rng)
            bf_norm = np.sum(np.abs(single_beamformer)**2, axis=0)
            single_beamformer = np.sqrt(env.ps_lin / bf_norm) * single_beamformer
            beamforming_vector[:, :, nb, :, :] = single_beamformer
        else:
            # With RIS: beamforming towards the RISs.
            single_beamformer = np.sum(np.conj(env.BS_RIS_channel[:, ris_idxs, nb, :]), axis=1, keepdims=False)
            bf_norm = np.sum(np.abs(single_beamformer)**2, axis=0)
            single_beamformer = np.sqrt(env.ps_lin / bf_norm) * single_beamformer

            single_beamformer = np.repeat(single_beamformer[:, np.newaxis, :, np.newaxis], env.NN, axis=3)
            
            beamforming_vector[:, :, nb, :, :] = single_beamformer

    return beamforming_vector

import numpy as np


def est_values_pos(N_BS: int, 
                   N_RIS: int, 
                   N_UE: int, 
                   M_BS: int, 
                   M_RIS: int, 
                   M_UE: int, 
                   BS_idx: int, 
                   ps_lin: float, 
                   sigma_n2: float, 
                   pow_ue_bs: np.ndarray, 
                   pow_ris_bs: np.ndarray, 
                   pow_ris_ue: np.ndarray, 
                   util_alpha: float, 
                   BS_UE_assoc: np.ndarray, 
                   RIS_allocs: np.ndarray, 
                   IBI: np.ndarray, 
                   LOS_UE_BS: np.ndarray, 
                   LOS_RIS_BS: np.ndarray, 
                   LOS_RIS_UE: np.ndarray, 
                   K: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate values of an agent only for certain RIS allocations accounting 
    for positions and K-factors.

    Args:
        N_BS: Number of BSs per operator.
        N_RIS: Number of RISs.
        N_UE: Number of UEs per operator.
        M_BS: Number of BS antenna elements.
        M_RIS: Number of RIS elements.
        M_UE: Number of UE antenna elements.
        BS_idx: Index of the base station (agent).
        ps_lin: Linear power.
        sigma_n2: Linear noise power.
        pow_ue_bs (N_UE, N_BS): Path gain between UEs and BSs.
        pow_ris_bs (N_RIS, N_BS): Path gain between RISs and BSs.
        pow_ris_ue (N_UE, N_RIS): Path gain between RISs and UEs.
        util_alpha: Alpha value of alpha-fair utility function.
        BS_UE_assoc (N_UE,): Allocation of each UEs to BSs.
        RIS_allocs (N_allocs, N_RIS): Array of RIS allocations to consider.
        IBI (N_BS, N_BS, N_RIS): Interference between BSs at RISs.
        LOS_UE_BS (N_UE, N_BS): LOS mask (0/1) between UEs and BSs.
        LOS_RIS_BS (N_RIS, N_BS): LOS mask (0/1) between RISs and BSs.
        LOS_RIS_UE (N_RIS, N_UE): LOS mask (0/1) between RISs and UEs.
        K: Rician K-factor for LOS and NLOS.

    Returns:
        value (N_allocs,): Estimated value for each RIS allocation.
        SINR_vals (N_users, N_allocs): Estimated SINR values for each user.
        power_vals (N_users, N_allocs): Estimated power values for each user.
        interference_vals (N_users, N_allocs): Estimated interference values for each user.
    """
    users = np.flatnonzero(BS_UE_assoc == BS_idx).astype(np.int32)
    n_users = users.size
    n_allocs = RIS_allocs.shape[0]

    assert n_users != 0, "No users associated with the BS."

    SINR_vals = np.zeros((n_users, n_allocs))
    power_vals = np.zeros((n_users, n_allocs))
    interference_vals = np.zeros((n_users, n_allocs))

    # Direct channels.
    kk_direct = 0 # BSs and UEs are NLOS, otherwise: kk_direct = K[1 - LOS_UE_BS]
    # Received power over Gauss direct channels.
    pow_direct_Gauss = ps_lin * pow_ue_bs * (1 / (1 + kk_direct))
    # Magnitude of direct signals over directional channels (needed for coherent combination with RIS parts).
    mag_direct_dir = np.sqrt(pow_ue_bs) * np.sqrt(kk_direct / (1 + kk_direct))
    
    # Loop over all considered RIS allocations.
    for rr in range(n_allocs):
        ris_alloc = RIS_allocs[rr, :]  # current RIS allocation
        n_ris_alloc = np.sum(ris_alloc)

        tmp_SINR = []
        tmp_power = []
        tmp_interference = []

        # Incoherent signals from unassigned RISs (includes Gaussian channels).
        # No need for a K-factor here, since Gauss and directional channels 
        # behave the same for incoherent RISs.
        ris_ue = pow_ris_ue * (1 - ris_alloc)
        # Received power over incoherent RIS channels.
        pow_ris_incoh = ps_lin * ris_ue @ (M_RIS * pow_ris_bs)

        # Incoherent signals from assigned RISs (these are received over Gauss channels).
        pow_ris_Gauss = np.zeros((N_UE, N_BS))
        # Here we need a K-factor to account for relative strength compared to coherent signals.
        kk_ue = K[1 - LOS_RIS_UE[:, users]]  # K-factor between UE and RIS.
        ris_ue = (1.0 / (1.0 + kk_ue)).T * pow_ris_ue[users, :] * ris_alloc
        # Received power over incoherent RIS channels.
        pow_ris_Gauss[users, :] = ps_lin * ris_ue @ (M_BS * M_RIS * pow_ris_bs)
        if n_ris_alloc > 0:
            pow_ris_Gauss[users, BS_UE_assoc[users]] /= n_ris_alloc

        assigned_riss = np.flatnonzero(ris_alloc > 0)

        # Coherent signals from assigned RISs.
        kk_ue = K[1 - LOS_RIS_UE[:, users]]  # K-factor between UE and RIS.

        for nu in users:
            kk_ue = K[1 - LOS_RIS_UE[:, nu]]  # K-factor between UE and RIS.
            # Coherent combination increases magnitude by M_RIS.
            ris_mag_coh = M_RIS * np.sqrt(pow_ris_bs[:, BS_UE_assoc[nu]])
            # Coherent intended signals from RISs.
            mag_ris_intended = np.sqrt((kk_ue[assigned_riss] / (1 + kk_ue[assigned_riss]))) * np.sqrt(pow_ris_ue[nu, assigned_riss]) * ris_mag_coh[assigned_riss] * np.sqrt(M_BS)
            if n_ris_alloc > 0:
                mag_ris_intended /= np.sqrt(n_ris_alloc)
            
            # Interfering BSs.
            interf_ind = np.delete(np.arange(N_BS, dtype=np.int32), BS_UE_assoc[nu])
            
            # mag_ris_interf = np.zeros((len(assigned_riss), len(interf_ind)))
            # nbc = 0
            # for nb in interf_ind:  # Coherent interfering signals from RISs.
            #     nrc = 0
            #     for nr in assigned_riss:
            #         mag_ris_interf[nrc, nbc] = np.sqrt((kk_ue[nr] / (1+kk_ue[nr]))) * np.sqrt(
            #             pow_ris_ue[nu, nr]) * np.sqrt(pow_ris_bs[nr, nb]) * IBI[BS_UE_assoc[nu], nb, nr]
            #         nrc += 1
            #     nbc += 1

            power_coherent = ps_lin * (sum(mag_ris_intended) + mag_direct_dir[nu, BS_UE_assoc[nu]])**2
            # interf_coherent = ps_lin * sum(np.sum(mag_ris_interf**2, axis=0) + (mag_direct_dir[nu, interf_ind])**2)
            power = pow_direct_Gauss[nu, BS_UE_assoc[nu]] + power_coherent + pow_ris_Gauss[nu, BS_UE_assoc[nu]] # + pow_ris_incoh[nu, BS_UE_assoc[nu]]
            interference = sum(pow_direct_Gauss[nu, interf_ind]) + sum(pow_ris_incoh[nu, interf_ind]) # + interf_coherent + + sum(pow_ris_Gauss[nu, interf_ind])
            tmp_SINR.append(power / (interference + sigma_n2))
            tmp_power.append(power)
            tmp_interference.append(interference)

        SINR_vals[:, rr] = np.array(tmp_SINR)
        power_vals[:, rr] = np.array(tmp_power)
        interference_vals[:, rr] = np.array(tmp_interference)
    
    rates = np.log2(1 + SINR_vals)

    if util_alpha == 1:
        value = np.sum(np.log(rates), axis=0)
    else:
        value = np.sum(rates**(1 - util_alpha) / (1 - util_alpha), axis=0)
    
    value = value / n_users
    return value, SINR_vals, power_vals, interference_vals


def get_SINR(N_OP: int,
             N_BS: int, 
             N_UE: int,  
             sigma_n2: float, 
             NN: int, 
             BS_UE_assoc: np.ndarray, 
             beamforming_vector: np.ndarray, 
             direct_signal: np.ndarray, 
             RIS_signal: np.ndarray = np.array([]),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get signal-to-interference-plus-noise ratio (SINR) of the links.
    
    Args:
        N_OP: Number of operators.
        N_BS: Number of BSs per operator.
        N_UE: Number of UEs per operator.
        sigma_n2: Linear noise power.
        NN: Number of microscopic fading realizations.
        BS_UE_assoc (N_UE, N_OP): Allocation of each UEs to BSs.
        beamforming_vector (M_BS, 1, N_BS, N_OP, NN): complex array.
        direct_signal (M_BS, 1, N_UE, N_BS, N_OP, NN): Direct (BS–UE) channel.
        ris_signal (M_BS, 1, N_UE, N_BS, N_OP, NN): RIS-assisted (BS–UE) channel.
    
    Returns:
        direct_SINR (N_UE, N_OP, NN): Direct-only SINR.
        UE_power (N_UE, N_OP, NN): Received power of intended signals.
        Int_power (N_UE, N_OP, NN): Received power of interfering signals.
    """
    total_signal = direct_signal + (RIS_signal if RIS_signal.size else 0)
    total_signal = np.einsum('i...,i...->...', beamforming_vector, total_signal)
    power = np.reshape(np.abs(total_signal)**2, (N_UE, N_BS, N_OP, NN))

    SINR = np.zeros((N_UE, N_OP, NN))
    UE_power = np.zeros((N_UE, N_OP, NN))
    Int_power = np.zeros((N_UE, N_OP, NN))

    for no in range(N_OP):
        for nu in range(N_UE):
            # Intended signal power.
            UE_power[nu, no, :] = power[nu, BS_UE_assoc[nu, no], no, :]
            # Interference from other BSs.
            Int_power[nu, no, :] = np.sum(power[nu, :, no, :], axis=0) - UE_power[nu, no, :]
            # SINR.
            SINR[nu, no, :] = 10 * np.log10(UE_power[nu, no, :] / (Int_power[nu, no, :] + sigma_n2))
    
    return SINR, UE_power, Int_power


def calculate_sum_rate(total_SINR: np.ndarray) -> np.ndarray:
    """
    Calculate the sum rate based on the total SINR. Because of the time-orthogonality
    between users, the sum rate is divided by the number of users.

    Args:
        total_SINR (users, NN): SINR (dB).
    Returns:
        Sum rate (NN): Sum rate (bits/s/Hz).
    """
    return np.sum(np.log2(1 + 10**(total_SINR / 10)), axis=0) / total_SINR.shape[0]


def calculate_min_rate(total_SINR: np.ndarray) -> np.ndarray:
    """
    Calculate the min rate based on the total SINR. Because of the time-orthogonality
    between users, the sum rate is divided by the number of users.

    Args:
        total_SINR (users, NN): SINR (dB).
    Returns:
        Min rate (NN): Min rate (bits/s/Hz).
    """
    return np.min(np.log2(1 + 10**(total_SINR / 10)), axis=0) / total_SINR.shape[0]

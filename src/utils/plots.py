import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF

from src.sim.valuation import est_values_pos


def reject_outliers(data: np.ndarray, m: float=2) -> np.ndarray:
    """Reject outliers from data based on the mean and standard deviation."""
    return data[np.abs(data - np.mean(data)) < m * np.std(data)]


def plot_reward(reward_aggregator: np.ndarray, eval_episodes: int, save_folder: Path):
    """Create reward plot."""
    plt.figure(figsize=(7, 5))
    ep_num = np.arange(0, eval_episodes)
    # Plot results where the budget is equal for both agents.
    plt.plot(ep_num, reward_aggregator[0, :, 0])
    plt.plot(ep_num, reward_aggregator[0, :, 1])
    plt.savefig(f'{save_folder}/Rewards.jpg', dpi=300, bbox_inches='tight')
    plt.clf()


def plot_budget_reward_ecdf(reward_aggregator: np.ndarray, budgets: np.ndarray, N_BS: int, save_folder: Path):
    """Create plot where the two agents are compared with different budgets."""
    plt.figure(figsize=(7, 5))
    linestyles = ['-', '--', '-.', ':']
    for budget_idx, _ in enumerate(budgets):
        for nb in range(N_BS):
            ecdf_rewards = ECDF(np.ndarray.flatten(reward_aggregator[budget_idx, :, nb]))
            if budget_idx == 0:
                label = f'Base station {nb}'
            else:
                label = None
            plt.plot(ecdf_rewards.x, ecdf_rewards.y, label=label, linestyle=linestyles[budget_idx], color=plt.colormaps["Dark2"].colors[nb])
    plt.xlabel('Reward')
    plt.ylabel('Empirical cumulative distribution')
    plt.title(f'Rewards for different budgets')
    plt.legend()
    plt.savefig(f'{save_folder}/Budget_{reward_aggregator.shape[1]}.jpg', dpi=300, bbox_inches='tight')
    plt.clf()


def plot_geometry(N_BS: int, BS_pos: np.ndarray, RIS_pos: np.ndarray, 
                  UE_pos: np.ndarray, BS_UE_assoc: np.ndarray, 
                  BS_RIS_alloc: np.ndarray, method: str, save_folder: Path):
    """Plot positions of BSs, RISs, UEs, including RIS allocation."""
    plt.figure(figsize=(7, 5))
    for nb in range(N_BS):
        ue_indices, op_indices = np.where(BS_UE_assoc == nb)
        op_index = op_indices[0] # A BS can only be associated to one operator.
        plt.scatter(UE_pos[op_index][ue_indices, 0], UE_pos[op_index][ue_indices, 1], label=f'{nb}. BS UEs', 
                    color=plt.colormaps["Dark2"].colors[nb])
        plt.scatter(BS_pos[op_index][nb, 0], BS_pos[op_index][nb, 1], label=f'{nb}. BS', 
                    color=plt.colormaps["Dark2"].colors[nb], marker="^")
        ris_indices = np.where(BS_RIS_alloc == nb)
        plt.scatter(RIS_pos[ris_indices, 0], RIS_pos[ris_indices, 1], label=f'{nb}. BS RISs', 
                    color=plt.colormaps["Dark2"].colors[nb], marker="s")
    unallocated_ris_indices = np.where(BS_RIS_alloc == N_BS)
    plt.scatter(RIS_pos[unallocated_ris_indices, 0], RIS_pos[unallocated_ris_indices, 1], label=f'Unallocated RISs', 
        color='gray', marker="s")
    plt.legend(loc="center left")
    plt.tight_layout()
    plt.savefig(f'{save_folder}/plot_positions_{method}.jpg', dpi=300, bbox_inches='tight')
    plt.clf()


def plot_performance_ecdf(metric_aggregator: np.ndarray, save_folder: Path, include_without_ris: bool = True):
    """Plots an ECDF function of the given metric."""
    eval_episodes = metric_aggregator.shape[0]
    n_methods = metric_aggregator.shape[1]
    metric_aggregator = np.mean(metric_aggregator, axis=(2, 3))
    loop_length = n_methods if include_without_ris else n_methods - 1
    plt.figure(figsize=(7, 5))
    for method_idx in range(loop_length):
        sum_rates = metric_aggregator[:, method_idx]
        ecdf_total = ECDF(sum_rates)
        
        if method_idx == 0:
            linestyle = "-"
            label = 'RIS RL'
        elif method_idx == 1:
            linestyle = "--"
            label = 'RIS Greedy'
        elif method_idx == 2:
            linestyle = "-."
            label = 'RIS Distance-based'
        elif method_idx == 3:
            linestyle = ":"
            label = 'Without RISs'

        plt.plot(ecdf_total.x, ecdf_total.y, linestyle=linestyle, label=label)
    
    plt.legend()
    plt.xlabel('Sum rates bits/s/Hz')
    plt.ylabel('Empirical cumulative distribution')
    plt.grid(visible = True)
    plt.savefig(f'{save_folder}/Sum_rates_{eval_episodes}.jpg', dpi=300, bbox_inches='tight')
    plt.clf()


def save_sinr_accuracy_csv(env, total_SINR_true, total_SINR_estimates, save_folder, tag):
    """
    Saves a tidy CSV with one row per UE containing true vs estimated SINR (dB),
    plus BS/operator identifiers for coloring/grouping in plots.
    """
    rows = []
    op = 0
    for ue in range(env.N_UE):
        bs = int(env.BS_UE_assoc[ue])  # serving BS index for this UE

        sinr_true = float(total_SINR_true[ue, op])
        sinr_est  = float(total_SINR_estimates[ue, op])
        err = sinr_est - sinr_true

        rows.append({
            "ue": ue,
            "bs": bs,
            "op": op,
            "sinr_true_db": sinr_true,
            "sinr_est_db": sinr_est,
            "err_db": err,
            "abs_err_db": abs(err),
            "sq_err_db2": err**2
        })

    df_out = pd.DataFrame(rows)
    os.makedirs(save_folder, exist_ok=True)
    out_path = os.path.join(save_folder, f"sinr_accuracy_{tag}.csv")
    df_out.to_csv(out_path, index=False)
    return out_path


def plot_estimate_accuracy(env, total_SINR_RL, total_power_RL, total_interf_RL, 
    save_folder):
    no = 0
    total_SINR_true = 10 * np.log10(np.mean(10**(total_SINR_RL / 10), axis=2))
    total_power_true = 10 * np.log10(np.mean(total_power_RL, axis=2))
    total_interf_true = 10 * np.log10(np.mean(total_interf_RL, axis=2))
    total_SINR_estimates = np.zeros((env.N_UE, env.N_OP))
    total_power_estimates = np.zeros((env.N_UE, env.N_OP))
    total_interf_estimates = np.zeros((env.N_UE, env.N_OP))
    for agent_idx in range(len(env.agents)):
        # Value of current assignment.
        BS_RIS_alloc_onehot = np.zeros((1, env.N_RIS))
        BS_RIS_alloc_onehot[:] = env.BS_RIS_assignment[env.agents[agent_idx]]
        
        # Calculate estimates only for a specific agent.
        (value_won_RIS, SINR_val_estimates, power_vals, interference_vals) = est_values_pos(env.N_BS, env.N_RIS, env.N_UE, env.M_BS, env.M_RIS, env.M_UE, agent_idx, env.ps_lin, env.sigma_n2,
                            env.pow_ue_bs[:, :, no], env.pow_ris_bs[:, :, no], env.pow_ris_ue[:, :, no], env.util_alpha,
                            env.BS_UE_assoc[:, no], BS_RIS_alloc_onehot, env.IBI[:, :, :, no], env.LOS_UE_BS[:, :, no], env.LOS_RIS_BS[:, :, no],
                            env.LOS_RIS_UE[:, :, no], env.K)
        SINR_val_estimates = 10 * np.log10(SINR_val_estimates)
        power_vals = 10 * np.log10(power_vals)
        interference_vals = 10 * np.log10(interference_vals)

        users = [u for u in range(env.N_UE) if env.BS_UE_assoc[u] == agent_idx]  # Users of base station.

        total_SINR_estimates[users, no] = SINR_val_estimates[:, 0]
        total_power_estimates[users, no] = power_vals[:, 0]
        total_interf_estimates[users, no] = interference_vals[:, 0]
    
    tag = f"MBS{env.M_BS}"
    csv_path = save_sinr_accuracy_csv(env, total_SINR_true, total_SINR_estimates, save_folder, tag)

    plt.figure(figsize=(7, 5))
    for nb in range(env.N_BS):
        nu_indices, op_indices = np.where(env.BS_UE_assoc == nb)
        op_index = op_indices[0] # A BS can only be associated to one operator
        plt.plot(total_SINR_estimates[nu_indices, op_index], total_SINR_true[nu_indices, op_index], 'o', label='RL vs Estimation',
                    color=plt.colormaps["Dark2"].colors[nb])

    plt.axline((0, 0), slope=1)
    plt.xlabel('SINR estimates (dB)')
    plt.ylabel('SINR values (dB)')
    plt.yticks(np.arange(np.round(min(total_SINR_true.flatten())), np.round(max(total_SINR_true.flatten()) + 1), 5))
    plt.axis('square')
    plt.grid(visible = True)
    plt.savefig(f'{save_folder}/plot_SINR_true_vs_estimates.jpg', dpi=300, bbox_inches='tight')
    plt.clf()

    for nb in range(env.N_BS):
        nu_indices, op_indices = np.where(env.BS_UE_assoc == nb)
        op_index = op_indices[0] # A BS can only be associated to one operator
        plt.plot(total_power_estimates[nu_indices, op_index], total_power_true[nu_indices, op_index], 'o', label='RL vs Estimation',
                    color=plt.colormaps["Dark2"].colors[nb])
    
    plt.axline((-140, -140), slope=1)
    plt.xlabel('Power estimates (dB)')
    plt.ylabel('Power values (dB)')
    plt.yticks(np.arange(np.round(min(total_power_true.flatten())), np.round(max(total_power_true.flatten()) + 1), 5))
    plt.axis('square')
    plt.grid(visible = True)
    plt.savefig(f'{save_folder}/plot_power_true_vs_estimates.jpg', dpi=300, bbox_inches='tight')
    plt.clf()

    for nb in range(env.N_BS):
        nu_indices, op_indices = np.where(env.BS_UE_assoc == nb)
        op_index = op_indices[0] # A BS can only be associated to one operator
        plt.plot(total_interf_estimates[nu_indices, op_index], total_interf_true[nu_indices, op_index], 'o', label='RL vs Estimation',
                    color=plt.colormaps["Dark2"].colors[nb])

    plt.axline((-150, -150), slope=1)
    plt.xlabel('Interference estimates (dB)')
    plt.ylabel('Interference values (dB)')
    plt.yticks(np.arange(np.round(min(total_interf_true.flatten())), np.round(max(total_interf_true.flatten()) + 1 ), 5))
    plt.axis('square')
    plt.grid(visible = True)
    plt.savefig(f'{save_folder}/plot_interf_true_vs_estimates.jpg', dpi=300, bbox_inches='tight')
    plt.clf()
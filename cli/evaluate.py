from pathlib import Path
import time

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from statsmodels.distributions.empirical_distribution import ECDF

from src.utils.plots import plot_budget_reward_ecdf, plot_estimate_accuracy, plot_geometry, plot_performance_ecdf, plot_reward
from cli.train import make_vec_env
from src.config import Config
from src.sim.beamforming import compute_beamforming_vector
from src.sim.channels import RIS_alloc_response, RIS_channel, direct_channel
from src.sim.valuation import calculate_sum_rate, est_values_pos, get_SINR


plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11
    })


if __name__ == "__main__":
    start = time.time()

    cfg = Config()
    save_folder = Path("results") / f"{cfg.model_name}_{cfg.f_name}"
    best_model_path = save_folder / "best_model.zip"
    env, vec_env = make_vec_env(cfg, render_mode=None)
   
    agents = env.agents
    n_agents = len(agents)

    # Test: Test BS0 with different budgets, while keeping BS1 fixed.
    if cfg.budget_test:
        model = PPO.load(best_model_path, env=vec_env)

        budgets = np.array([1.0, 2.0, 4.0, 8.0]) # Budgets to test.
        env.max_budget["BS1"] = 1.0
        # Store episode rewards for all agents with different budget.
        reward_aggregator = np.zeros((len(budgets), cfg.eval_episodes, n_agents))

        for budget_idx, budget in enumerate(budgets):
            print(f"Testing with budget: {budget}")
            env.max_budget["BS0"] = budget  # Set budget for BS0.

            acc_rewards = np.zeros(n_agents) # Accumulate rewards of each episode.
            ep_rewards = [] # Store episode rewards.
            ep_counter = 0

            while ep_counter < cfg.eval_episodes:
                observe, _ = env.reset(ep_counter) # Initial state.
                terminated = False
                while not terminated:
                    obs = np.array([observe[agent] for agent in agents])
                    actions, _ = model.predict(obs, deterministic=True)
                    actions = {agent: actions[agent_idx, :] for agent_idx, agent in enumerate(agents)}
                    observe, rewards, dones, _, _ = env.step(actions)

                    terminated = True
                    for i in range(n_agents):
                        acc_rewards[i] += rewards[agents[i]]
                        # Check whether auction has ended.
                        terminated = np.logical_and(terminated, dones[agents[i]])
                    
                # Auction finished.
                ep_rewards.append(acc_rewards) # Store rewards of episode.
                acc_rewards = np.zeros(n_agents)
                ep_counter += 1
                observe, _ = env.reset(ep_counter)
            
            ep_rewards = np.array(ep_rewards)
            reward_aggregator[budget_idx, :, :] = ep_rewards

        plot_reward(reward_aggregator, cfg.eval_episodes, save_folder)
        plot_budget_reward_ecdf(reward_aggregator, budgets, env.N_BS, save_folder)

        # Reset budget to original value.
        env.max_budget = {agents[i]: cfg.budget[i] for i in range(env.N_BS)}

    # Test: Compare performance of RL algorithm and multiple heuristics.
    if cfg.performance_test:
        model = PPO.load(best_model_path, env=vec_env)
        env.show_plot = True

        # 4 algorithms: RL, greedy, distance-based, no RIS.
        n_methods = 4
        sum_rates_aggregator = np.zeros((cfg.eval_episodes, n_methods, env.N_BS, env.N_OP))
        costs_aggregator = np.zeros((cfg.eval_episodes, n_methods, env.N_BS, env.N_OP))
        n_riss_allocated_aggregator = np.zeros((cfg.eval_episodes, n_methods, env.N_BS, env.N_OP))
        bid_values_aggregator = [[], [], []] # 3 only: no RIS doesn't bid.

        ep_counter = 0
        
        while ep_counter < cfg.eval_episodes:
            if ep_counter % np.ceil((cfg.eval_episodes / 10)) == 0:
                print(f'Performance test progress: {int(100 * ep_counter / cfg.eval_episodes)}% (Iteration {ep_counter})', flush=True)

            # ---- Reinforcement learning auction ----
            observe, infos = env.reset(ep_counter)
            # print(f"Observe at reset: {observe}\n")
            terminated = False
            while not terminated:
                obs = np.array([observe[agent] for agent in agents])
                actions, _ = model.predict(obs, deterministic=True)
                # print(f"Predicted actions: {temp_actions}\n")
                actions = {agent: actions[agent_idx, :] for agent_idx, agent in enumerate(agents)}
                observe, rewards, dones, _, infos = env.step(actions)
                bid_values_aggregator[0].extend([infos[agent]["R1"] / np.sum(actions[agent]) for agent in agents if np.sum(actions[agent])])
                # print(f"Rewards: {rewards}\n")
                # print(f"Dones: {dones}\n")
                # print(f"Observe: {observe}\n")
                terminated = True
                for i in range(n_agents):
                    terminated = np.logical_and(terminated, dones[agents[i]])
            
            # Everything allocated to operator 0.
            RIS_alloc = np.ones((env.N_RIS,)) * 0

            # TODO: Reintroduce when we will have multiple operators.
            # Obtained RIS allocation operator-wise.
            # RIS_alloc = np.ones((N_RIS,)) * N_OP
            # for no in range(N_OP):
                # RIS_alloc[env.RIS_assignment[agents[no]]] = no
            
            # Obtained RIS allocation base-station-wise.
            BS_RIS_alloc = np.ones((env.N_RIS,)) * n_agents
            for nb in range(n_agents):
                BS_RIS_alloc[env.BS_RIS_assignment[agents[nb]]] = nb

            if ep_counter == 0 and env.show_plot:
                plot_geometry(env.N_BS, env.BS_pos, env.RIS_pos, env.UE_pos, 
                    env.BS_UE_assoc, BS_RIS_alloc, "RL", save_folder)
            
            beamforming_vector = compute_beamforming_vector(env, BS_RIS_alloc)
            direct_channel_BS_UE = direct_channel(env.N_OP, env.N_BS, env.N_UE, 
                env.M_BS, env.M_UE, env.NN, env.K, env.LOS_UE_BS, env.pow_ue_bs, 
                env.UE_BS_channel, env.BS_UE_channel, env.channels_BS_UE, env.BS_UE_assoc)
            RIS_resp = RIS_alloc_response(env.RIS_UE_channel, env.RIS_BS_channel, RIS_alloc, BS_RIS_alloc, env.BS_UE_assoc, env.rng)
            RIS_channel_BS_UE = RIS_channel(env.N_OP, env.N_BS, env.N_RIS, 
                env.N_UE, env.M_BS, env.M_RIS, env.M_UE, env.NN, env.K, 
                env.LOS_RIS_UE, env.LOS_RIS_BS, env.RIS_BS_channel, 
                env.BS_RIS_channel, env.channels_BS_RIS, env.RIS_UE_channel, 
                env.channels_RIS_UE, env.BS_UE_assoc, RIS_resp, 
                env.pow_ris_bs, env.pow_ris_ue, env.rng)
            total_SINR_RL, total_power_RL, total_interf_RL = get_SINR(env.N_OP, 
                env.N_BS, env.N_UE, env.sigma_n2, env.NN, env.BS_UE_assoc, 
                beamforming_vector, direct_channel_BS_UE, RIS_channel_BS_UE)

            for nb in range(n_agents):
                ue_indices, op_indices = np.where(env.BS_UE_assoc == nb)
                op_index = op_indices[0] # A BS can only be associated to one operator.
                sum_rate = np.mean(calculate_sum_rate(total_SINR_RL[ue_indices, op_index, :]))
                sum_rates_aggregator[ep_counter, 0, nb, op_index] = sum_rate
                costs_aggregator[ep_counter, 0, nb, op_index] = env.acc_cost[agents[nb]]
                n_riss_allocated_aggregator[ep_counter, 0, nb, op_index] = sum(env.BS_RIS_assignment[agents[nb]])
            
            if 0:
                plot_estimate_accuracy(env, total_SINR_RL, total_power_RL, total_interf_RL, save_folder)
                exit(0)

            # ---- Greedy auction ----
            observe, infos = env.reset(ep_counter)
            
            terminated = False
            while not terminated:
                obs = np.array([observe[agent] for agent in agents])
                    
                for i in range(0, n_agents):
                    bet_factor = 1
                    RISs_val = obs[i, 0:env.N_RIS]
                    RISs_sort_ind = RISs_val.argsort()  
                    action = np.zeros(RISs_val.shape, bool)
                    num_bets = int(min(np.floor(bet_factor * obs[i, -1] / max(obs[i, -2], 1e-6)), env.N_RIS))
                    if num_bets > 0:
                        action[RISs_sort_ind[-num_bets:]] = True
                    actions[agents[i]] = action.astype(bool)

                observe, rewards, dones, _, infos = env.step(actions)
                bid_values_aggregator[1].extend([infos[agent]["R1"] / np.sum(actions[agent]) for agent in agents if np.sum(actions[agent])])

                terminated = True
                for i in range(n_agents):
                    terminated = np.logical_and(terminated, dones[agents[i]])
                        
            # Obtained RIS allocation base-station-wise.
            BS_RIS_alloc = np.ones((env.N_RIS,)) * n_agents
            for no in range(n_agents):
                BS_RIS_alloc[env.BS_RIS_assignment[agents[no]]] = no

            if ep_counter == 0 and env.show_plot:
                plot_geometry(env.N_BS, env.BS_pos, env.RIS_pos, env.UE_pos, 
                    env.BS_UE_assoc, BS_RIS_alloc, "greedy", save_folder)

            beamforming_vector = compute_beamforming_vector(env, BS_RIS_alloc)
            direct_channel_BS_UE = direct_channel(env.N_OP, env.N_BS, env.N_UE, 
                env.M_BS, env.M_UE, env.NN, env.K, env.LOS_UE_BS, env.pow_ue_bs, 
                env.UE_BS_channel, env.BS_UE_channel, env.channels_BS_UE, env.BS_UE_assoc)
            RIS_resp = RIS_alloc_response(env.RIS_UE_channel, env.RIS_BS_channel, RIS_alloc, BS_RIS_alloc, env.BS_UE_assoc, env.rng)
            RIS_channel_BS_UE = RIS_channel(env.N_OP, env.N_BS, env.N_RIS, 
                env.N_UE, env.M_BS, env.M_RIS, env.M_UE, env.NN, env.K, 
                env.LOS_RIS_UE, env.LOS_RIS_BS, env.RIS_BS_channel, 
                env.BS_RIS_channel, env.channels_BS_RIS, env.RIS_UE_channel, 
                env.channels_RIS_UE, env.BS_UE_assoc, RIS_resp, 
                env.pow_ris_bs, env.pow_ris_ue, env.rng)
            (total_SINR_greedy, total_power_greedy, 
             total_interf_greedy) = get_SINR(env.N_OP, env.N_BS, env.N_UE, 
                env.sigma_n2, env.NN, env.BS_UE_assoc, beamforming_vector, 
                direct_channel_BS_UE, RIS_channel_BS_UE)

            for nb in range(env.N_BS):
                ue_indices, op_indices = np.where(env.BS_UE_assoc == nb)
                op_index = op_indices[0] # A BS can only be associated to one operator.
                sum_rate = np.mean(calculate_sum_rate(total_SINR_greedy[ue_indices, op_index, :]))
                sum_rates_aggregator[ep_counter, 1, nb, op_index] = sum_rate
                costs_aggregator[ep_counter, 1, nb, op_index] = env.acc_cost[agents[nb]]
                n_riss_allocated_aggregator[ep_counter, 1, nb, op_index] = sum(env.BS_RIS_assignment[agents[nb]])

            # ---- Distance-based auction ----
            observe, infos = env.reset(ep_counter)
            
            distances = []
            for nr in range(env.N_RIS):
                for no in range(env.N_OP):
                    for nb in range(env.N_BS):
                        distance = np.linalg.norm(env.BS_pos[no][nb, :] - env.RIS_pos[nr, :])
                        distances.append((nr, no, nb, distance))

            distance_based_val = np.zeros([env.N_BS, env.N_RIS])
            for nr, no, nb, dist in distances:
                # Value inversely proportional to distance.
                distance_based_val[nb, nr] = 1 / (dist + 1e-6)

            terminated = False
            while not terminated:
                obs = np.array([observe[agent] for agent in agents])
                
                for i in range(0, n_agents):
                    bet_factor = 1
                    RISs_val = distance_based_val[i]
                    # Don't bid on RISs without potential improvement.
                    # TODO: Remove comment to get slightly more clever heuristics.
                    # RISs_val[obs[i, 0:env.N_RIS] <= 0] = 0
                    RISs_sort_ind = RISs_val.argsort()
                    action = np.zeros(RISs_val.shape, bool)
                    num_bets = int(min(np.floor(bet_factor * obs[i, -1] / max(obs[i, -2], 1e-6)), env.N_RIS))
                    if num_bets > 0:
                        action[RISs_sort_ind[-num_bets:]] = True
                    actions[agents[i]] = action.astype(bool)
                
                observe, rewards, dones, _, infos = env.step(actions)
                bid_values_aggregator[2].extend([infos[agent]["R1"] / np.sum(actions[agent]) for agent in agents if np.sum(actions[agent])])

                terminated = True
                for i in range(n_agents):
                    terminated = np.logical_and(terminated, dones[agents[i]])
            
            # Obtained RIS allocation base-station-wise.
            BS_RIS_alloc = np.ones((env.N_RIS,)) * n_agents
            for no in range(n_agents):
                BS_RIS_alloc[env.BS_RIS_assignment[agents[no]]] = no

            if ep_counter == 0 and env.show_plot:
                plot_geometry(env.N_BS, env.BS_pos, env.RIS_pos, env.UE_pos, 
                    env.BS_UE_assoc, BS_RIS_alloc, "distance_based", save_folder)

            beamforming_vector = compute_beamforming_vector(env, BS_RIS_alloc)
            direct_channel_BS_UE = direct_channel(env.N_OP, env.N_BS, env.N_UE, 
                env.M_BS, env.M_UE, env.NN, env.K, env.LOS_UE_BS, env.pow_ue_bs, 
                env.UE_BS_channel, env.BS_UE_channel, env.channels_BS_UE, env.BS_UE_assoc)
            RIS_resp = RIS_alloc_response(env.RIS_UE_channel, env.RIS_BS_channel, RIS_alloc, BS_RIS_alloc, env.BS_UE_assoc, env.rng)
            RIS_channel_BS_UE = RIS_channel(env.N_OP, env.N_BS, env.N_RIS, 
                env.N_UE, env.M_BS, env.M_RIS, env.M_UE, env.NN, env.K, 
                env.LOS_RIS_UE, env.LOS_RIS_BS, env.RIS_BS_channel, 
                env.BS_RIS_channel, env.channels_BS_RIS, env.RIS_UE_channel, 
                env.channels_RIS_UE, env.BS_UE_assoc, RIS_resp, 
                env.pow_ris_bs, env.pow_ris_ue, env.rng)
            (total_SINR_distance_based, total_power_distance_based, 
             total_interf_distance_based) = get_SINR(env.N_OP, env.N_BS, 
                env.N_UE, env.sigma_n2, env.NN, env.BS_UE_assoc, 
                beamforming_vector, direct_channel_BS_UE, RIS_channel_BS_UE)

            for nb in range(env.N_BS):
                ue_indices, op_indices = np.where(env.BS_UE_assoc == nb)
                op_index = op_indices[0] # A BS can only be associated to one operator.
                sum_rate = np.mean(calculate_sum_rate(total_SINR_distance_based[ue_indices, op_index, :]))
                sum_rates_aggregator[ep_counter, 2, nb, op_index] = sum_rate
                costs_aggregator[ep_counter, 2, nb, op_index] = env.acc_cost[agents[nb]]
                n_riss_allocated_aggregator[ep_counter, 2, nb, op_index] = sum(env.BS_RIS_assignment[agents[nb]])

            # ---- Without RISs allocated ----
            # Deallocate everything
            BS_RIS_alloc_empty = np.ones((env.N_RIS,)) * env.N_BS

            beamforming_vector = compute_beamforming_vector(env, BS_RIS_alloc_empty)
            direct_channel_BS_UE = direct_channel(env.N_OP, env.N_BS, env.N_UE, 
                env.M_BS, env.M_UE, env.NN, env.K, env.LOS_UE_BS, env.pow_ue_bs, 
                env.UE_BS_channel, env.BS_UE_channel, env.channels_BS_UE, env.BS_UE_assoc)
            direct_SINR, direct_power, direct_interf = get_SINR(env.N_OP, 
                env.N_BS, env.N_UE, env.sigma_n2, env.NN, env.BS_UE_assoc, 
                beamforming_vector, direct_channel_BS_UE)

            for nb in range(env.N_BS):
                ue_indices, op_indices = np.where(env.BS_UE_assoc == nb)
                op_index = op_indices[0] # A BS can only be associated to one operator
                sum_rate = np.mean(calculate_sum_rate(direct_SINR[ue_indices, op_index, :]))
                sum_rates_aggregator[ep_counter, 3, nb, op_index] = sum_rate
                costs_aggregator[ep_counter, 3, nb, op_index] = 0  # no cost without RISs
                n_riss_allocated_aggregator[ep_counter, 3, nb, op_index] = 0

            ep_counter += 1

        plot_performance_ecdf(sum_rates_aggregator, save_folder)
        plot_performance_ecdf(costs_aggregator, save_folder, include_without_ris=False)

        sum_rates = np.mean(sum_rates_aggregator, axis=(0, 2, 3))
        costs = np.mean(costs_aggregator, axis=(0, 2, 3))
        n_riss_allocated = np.mean(n_riss_allocated_aggregator, axis=(0, 2, 3))
        bid_values = [np.mean(bid_values_aggregator[i]) for i in range(3)]
        bid_values.append(0) # Append 0 for the method without bids.

        data = pd.concat([pd.Series(sum_rates), pd.Series(costs), pd.Series(n_riss_allocated), pd.Series(bid_values)], axis=1)
        data.columns = ["Sum rate", "Cost", "Number of RISs allocated", "Bid value"]
        data.index = [f"RL beta {env.beta}", "Greedy", "Distance-based", "Without RISs"]
        data.to_csv(f"{save_folder}/Data.csv")
        
    end = time.time()
    print(f"End of execution. Elapsed time: {(end - start) / 60:.2f} minutes.", )

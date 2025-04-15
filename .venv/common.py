import numpy as np
import torch
from torch.distributions import Categorical

def select_action(obs, policy):
    dist = Categorical(policy(obs))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return (action.item(), log_prob.reshape(1))

def compute_returns(rewards, gamma):
    discounted_rewards = np.array([gamma**(i +1) * r for i,r in enumerate(rewards)][::-1])
    total_returns = np.cumsum(discounted_rewards)
    return np.flip(total_returns, axis=0).copy()

def run_episode(env, policy, maxlen=500):
    observations = []
    actions = []
    log_probs = []
    rewards = []

    # Reset the environment and start the episode.
    obs, info = env.reset()
    for _ in range(maxlen):
        # Convert observation to tensor and move to device
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        # Get action and log_prob from policy
        action, log_prob = select_action(obs_tensor, policy)

        # Save data
        observations.append(obs_tensor)
        actions.append(action)
        log_probs.append(log_prob)

        # Step environment
        obs, reward, term, trunc, _ = env.step(action)
        rewards.append(reward)

        if term or trunc:
            break

    return observations, actions, torch.stack(log_probs), rewards

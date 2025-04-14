import torch
import wandb
from networks import save_checkpoint
from common import run_episode, compute_returns

def reinforce_Cart_Pole(policy, env, run, gamma, lr, baseline, num_episodes,
                        eval_interval=100, eval_episodes=5, value_net=None, device='cpu'):

    if baseline not in ['none', 'std', 'value']:
        raise ValueError(f'Unknown baseline {baseline}')

    if baseline == 'value' and value_net is not None:
        value_opt = torch.optim.Adam(value_net.parameters(), lr=lr)
    else:
        value_opt = None

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    running_rewards = [0.0]
    best_return = 0.0

    policy.train()
    for episode in range(num_episodes):
        log = {}

        observations, actions, log_probs, rewards = run_episode(env, policy, device=device)
        log_probs = torch.stack(log_probs).to(device)
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32).to(device)

        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])
        log['episode_length'] = len(returns)
        log['return'] = returns[0].item()

        if running_rewards[-1] > best_return:
            save_checkpoint('BEST', policy, opt, wandb.run.dir)
            best_return = running_rewards[-1]

        if baseline == 'none':
            base_returns = returns
        elif baseline == 'std':
            base_returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        elif baseline == 'value':
            value_net.train()
            observations_tensor = torch.stack(observations).detach().to(device)
            values = value_net(observations_tensor).squeeze()
            advantages = returns - values

            # Update value network
            value_loss = torch.nn.functional.mse_loss(values, returns.detach())
            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()

            base_returns = advantages.detach()

        opt.zero_grad()
        policy_loss = (-log_probs * base_returns).mean()
        policy_loss.backward()
        opt.step()

        log['policy_loss'] = policy_loss.item()
        run.log(log)

        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            policy.eval()
            if value_net is not None:
                value_net.eval()
            total_rewards = []
            episode_lengths = []

            with torch.no_grad():
                for _ in range(eval_episodes):
                    _, _, _, rewards_eval = run_episode(env, policy, device=device)
                    total_rewards.append(sum(rewards_eval))
                    episode_lengths.append(len(rewards_eval))

            avg_reward = sum(total_rewards) / eval_episodes
            avg_length = sum(episode_lengths) / eval_episodes

            run.log({
                'eval/avg_reward': avg_reward,
                'eval/avg_length': avg_length,
            })

            print(f'[EVAL] Episode {episode + 1}: Avg. reward = {avg_reward:.2f}, Avg. length = {avg_length:.1f}')
            policy.train()

    policy.eval()
    if value_net is not None:
        value_net.eval()

    return running_rewards



def reinforce_Lunar_Lander(policy, env, run, gamma, lr, baseline, num_episodes,
                           eval_interval=100, eval_episodes=5, value_net=None, device='cpu'):
    if baseline not in ['none', 'std', 'value']:
        raise ValueError(f'Unknown baseline {baseline}')

    if value_net is not None:
        value_opt = torch.optim.Adam(value_net.parameters(), lr=lr)

    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    running_rewards = [0.0]
    best_return = 0.0

    policy.train()
    if value_net is not None:
        value_net.train()

    for episode in range(num_episodes):
        log = {}

        observations, actions, log_probs, rewards = run_episode(env, policy, maxlen=1000, device=device)
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32, device=device)

        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])
        log['return'] = returns[0].item()
        log['total_reward'] = sum(rewards)
        log['num_steps'] = len(rewards)
        log['crash'] = int(rewards[-1] == -100)
        log['successful_landing'] = int(rewards[-1] == 100)
        log['running_rewards'] = running_rewards[-1]

        if running_rewards[-1] > best_return:
            save_checkpoint('BEST', policy, opt, wandb.run.dir)
            best_return = running_rewards[-1]

        if baseline == 'none':
            base_returns = returns
        elif baseline == 'std':
            base_returns = (returns - returns.mean()) / returns.std()
        elif baseline == 'value':
            observations_tensor = torch.stack(observations).detach().to(device)
            values = value_net(observations_tensor).squeeze()
            advantages = returns - values

            # Update value network
            value_loss = torch.nn.functional.mse_loss(values, returns.detach())
            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()

            base_returns = advantages.detach()

        opt.zero_grad()
        policy_loss = (-log_probs.to(device) * base_returns).mean()
        policy_loss.backward()
        opt.step()
        log['policy_loss'] = policy_loss.item()
        run.log(log)

        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            policy.eval()
            if value_net is not None:
                value_net.eval()
            total_rewards = []
            episode_lengths = []

            with torch.no_grad():
                for _ in range(eval_episodes):
                    _, _, _, rewards_eval = run_episode(env, policy, device=device)
                    total_rewards.append(sum(rewards_eval))
                    episode_lengths.append(len(rewards_eval))

            avg_reward = sum(total_rewards) / eval_episodes
            avg_length = sum(episode_lengths) / eval_episodes

            run.log({
                'eval/avg_reward': avg_reward,
                'eval/avg_length': avg_length,
            })

            print(f'[EVAL] Episode {episode + 1}: Avg. reward = {avg_reward:.2f}, Avg. length = {avg_length:.1f}')
            policy.train()
            if value_net is not None:
                value_net.train()

    policy.eval()
    if value_net is not None:
        value_net.eval()
    return running_rewards









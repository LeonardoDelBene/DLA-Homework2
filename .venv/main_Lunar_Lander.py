import wandb
import gymnasium
import torch
from networks import PolicyNet, ValueNet
from reinforce import reinforce_Lunar_Lander

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    baseline = 'none'  # 'none' o 'std' o 'value'
    project_name = 'Homework-2-Lunar-Lander'
    run_name = "Baseline" + '_' + baseline
    gamma = 0.99
    lr = 1e-3
    episodes = 1000

    wandb.login(key="bfa1df1c98b555b96aa3777a18a6e8ca9b082d53")
    run = wandb.init(
        project=project_name,
        name=run_name,
        config={
            'learning_rate': lr,
            'baseline': baseline,
            'gamma': gamma,
            'num_episodes': episodes
        }
    )

    env = gymnasium.make('LunarLander-v3')

    policy = PolicyNet(env).to(device)
    value_net = ValueNet(env.observation_space.shape[0]).to(device)

    reinforce_Lunar_Lander(policy, env, run, gamma, lr, baseline, episodes, value_net=value_net, device=device)

    env.close()
    run.finish()


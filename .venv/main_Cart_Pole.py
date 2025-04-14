import wandb
import torch
import gymnasium
from networks import PolicyNet, ValueNet
from reinforce import reinforce_Cart_Pole
from common import run_episode


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = 'std' # 'none' o 'std' o 'value'
    project_name = 'Homework-2-CartPole'
    run_name = "Lunar-Ladder/Baseline" + '_' + baseline
    gamma = 0.99
    lr = 1e-3
    episodes = 1000
    
    print(baseline)
    wandb.login(key="bfa1df1c98b555b96aa3777a18a6e8ca9b082d53")
    run = wandb.init(
        project= project_name,
        name= run_name,
        config={
            'learning_rate': lr,
            'baseline': baseline,
            'gamma': gamma,
            'num_episodes': episodes
        }
    )

    env = gymnasium.make('CartPole-v1')

    policy = PolicyNet(env).to(device)
    value_net = ValueNet(env.observation_space.shape[0]).to(device)



    reinforce_Cart_Pole(policy, env, run,gamma, lr, baseline, episodes, value_net=value_net, device=device)

    env.close()
    run.finish()

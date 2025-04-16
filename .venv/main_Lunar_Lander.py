import wandb
import gymnasium
from networks import Policy_Lunar, Value_Lunar
from reinforce import reinforce_Lunar_Lander

if __name__ == '__main__':

    baseline = 'value'  # none, std o value
    project_name = 'Homework-2-Lunar-Lander'
    gamma = 0.99
    lr = 1e-3
    episodes = 10000

    run_name = 'Baseline-' + baseline
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

    policy = Policy_Lunar(env)
    value_net = Value_Lunar(env.observation_space.shape[0])

    reinforce_Lunar_Lander(policy, env, run, gamma, lr, baseline, episodes, value_net=value_net)

    run.finish()
    env.close()

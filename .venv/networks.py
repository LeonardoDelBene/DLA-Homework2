import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class PolicyNet(nn.Module):
    def __init__(self, env, width=128):
        super().__init__()
        hidden_layers = [nn.Linear(env.observation_space.shape[0],width), nn.ReLU()]
        hidden_layers += [nn.Linear(width,width), nn.ReLU()]

        self.hidden = nn.Sequential(*hidden_layers)
        self.out = nn.Linear(width,env.action_space.n)

    def forward(self, s):
        s = self.hidden(s)
        s = F.softmax(self.out(s), dim= -1)
        return s


def save_checkpoint(epoch, model, opt, dir):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict(),
        },
        os.path.join(dir, f'checkpoint-{epoch}.pt')
    )

def load_checkpoint(fname, model, opt=None):
    checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint['model_state_dict'])
    if opt:
        opt.load_state_dict(checkpoint['opt_state_dict'])
    return model


class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


class Policy_Lunar(nn.Module):
    def __init__(self, env, width=128):
        super().__init__()
        hidden_layers = [nn.Linear(env.observation_space.shape[0],width), nn.ReLU()]
        hidden_layers += [nn.Linear(width,width), nn.ReLU()]
        self.hidden = nn.Sequential(*hidden_layers)
        self.out = nn.Linear(width,env.action_space.n)

    def forward(self, s):
        s = self.hidden(s)
        s = F.softmax(self.out(s), dim= -1)
        return s

class Value_Lunar(nn.Module):
     def __init__(self, obs_dim, width = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, width),
                nn.ReLU(),
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, 1)
            )

     def forward(self, x):
            return self.net(x)
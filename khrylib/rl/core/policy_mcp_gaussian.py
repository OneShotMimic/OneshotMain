import torch.nn as nn
from khrylib.rl.core.distributions import DiagGaussian
from khrylib.rl.core.policy import Policy
from khrylib.utils.math import *


class GatingFunction(nn.Module):
    def __init__(self, state_dim, goal_dim, num_primitives=8):
        super(GatingFunction, self).__init__()
        self.sn1 = nn.Linear(state_dim, 512)
        self.sn2 = nn.Linear(512, 256)
        self.gn1 = nn.Linear(goal_dim, 512)
        self.gn2 = nn.Linear(512, 256)
        self.bottleneck = nn.Linear(512, 256)
        self.out = nn.Linear(256, num_primitives)

    def forward(self, state, goal):
        s = self.sn1(state).relu()
        s = self.sn2(s).relu()
        g = self.gn1(goal).relu()
        g = self.gn2(g).relu()
        axis = 1 if state.dim() == 2 else 0
        h = torch.cat((s, g), axis=axis)
        h = self.bottleneck(h).relu()
        return self.out(h).softmax(dim=1)

class MCPPolicyGaussian(Policy):
    def __init__(self, net, action_dim, net_out_dim=None, log_std=0, fix_std=False, num_primitives=8, goal_dim=39,summarize_action=False):
        super().__init__()
        self.type = 'gaussian'
        self.net = net
        if net_out_dim is None:
            net_out_dim = net.out_dim
        # TODO: Define gating function
        self.gating_function = GatingFunction(state_dim=net.state_dim, 
                                              goal_dim=goal_dim, num_primitives=num_primitives)

        # Define primitives
        self.num_primitives = num_primitives
        self.action_dim = action_dim
        self.action_mean = nn.Linear(net_out_dim, action_dim * self.num_primitives)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.1)

        # [n_primitives, n_action]
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim * self.num_primitives) * log_std, 
                                           requires_grad=not fix_std).view(self.num_primitives, action_dim)
        self.summarize_action = summarize_action
        self.cum_w = []
        self.similarities = []

    def freeze(self):
        self.net.requires_grad_(False)
        self.action_mean.requires_grad_(False)
        self.action_log_std.requires_grad_(False)

    def set_goal(self,g):
        if not torch.is_tensor(g):
            g = torch.from_numpy(g)
        self.g = g

    def forward(self, x):
        latent = self.net(x)
        if x.dim() != 1:
            goal = self.g.repeat(x.shape[0]).view(x.shape[0],-1).to(latent.device)
        else:
            goal = self.g.to(latent.device)
        w = self.gating_function(x,goal) # [batch_size, n_primitives]
        action_means = self.action_mean(latent).view(-1, self.num_primitives, self.action_dim) # [batch_size, n_primitives, action_dim]
        if self.summary_action:
            self.analyse_action(action_means.detach().cpu().numpy())
        action_std_ = torch.exp(self.action_log_std.expand_as(action_means)).to(latent.device) #[batch_size, n_primitives, action_dim]
        weighted_inv_std = w.unsqueeze(2) / action_std_ # [batch_size, n_primitives, action_dim]
        action_mean = 1/(weighted_inv_std.sum(dim=1)) * (weighted_inv_std*action_means).sum(dim=1) # [batch_size, action_dim]
        action_std = 1/(weighted_inv_std.sum(dim=1)) #[batch_size, action_dim]
        return DiagGaussian(action_mean, action_std)

    def forward_with_beta(self,x, w):
        latent = self.net(x)
        if w.dim() == 1:
            w = w.view(1,-1)
        if self.summarize_w:
            self.cum_w.append(w)
        action_means = self.action_mean(latent).view(-1, self.num_primitives, self.action_dim) # [batch_size, n_primitives, action_dim]
        action_std_ = torch.exp(self.action_log_std.expand_as(action_means)).to(latent.device) #[batch_size, n_primitives, action_dim]
        weighted_inv_std = w.unsqueeze(2) / action_std_ # [batch_size, n_primitives, action_dim]
        action_mean = 1/(weighted_inv_std.sum(dim=1)) * (weighted_inv_std*action_means).sum(dim=1) # [batch_size, action_dim]
        #action_std = 1/(weighted_inv_std.sum(dim=1)) #[batch_size, action_dim]
        #return DiagGaussian(action_mean, action_std)
        return action_mean.squeeze()

    def get_fim(self, x):
        """Currently not support TRPO"""
        raise NotImplementedError

    def summary_w(self):
        sum_w = np.asarray(self.cum_w).sum(axis=0)
        sum_w /= np.linalg.norm(sum_w)
        return sum_w

    def analyse_action(self, action):
        # Need to measure the similarity of each action
        action = action[0]
        for i in range(len(action)):
            action[i] = action[i] / np.linalg.norm(action[i])

        similarity = np.zeros((len(action),len(action)), dtype=np.float32)
        for i in range(len(action)):
            for j in range(i,len(action)):
                s = action[i].dot(action[j])
                similarity[i,j] = s
                similarity[j,i] = s
        self.similarities.append(similarity)

    def summary_action(self):
        self.similarities = np.asarray(self.similarities)
        return self.similarities
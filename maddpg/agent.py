import numpy as np
import torch as T
import torch.nn.functional as F
from networks import ActorNetwork, CriticNetwork

import os


class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions,
                 n_agents, agent_idx, chkpt_dir='models/MADDPG/', model='sample',
                 alpha=1e-3, beta=1e-3, fc1=64, fc2=64,
                 gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        agent_name = 'agent_%s' % agent_idx
        self.agent_idx = agent_idx

        chkpt_dir += model
        os.makedirs(chkpt_dir, exist_ok=True)


        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                  chkpt_dir=chkpt_dir,
                                  name=agent_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2,
                                    chkpt_dir=chkpt_dir,
                                    name=agent_name+'_critic')
        
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2,
                                         n_actions, chkpt_dir=chkpt_dir,
                                         name=agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2,
                                           chkpt_dir=chkpt_dir,
                                           name=agent_name+'_target_critic')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, evaluate=False):
        obs = T.tensor(observation[np.newaxis, :], dtype=T.float,
                         device=self.actor.device)
        actions = self.actor.forward(obs)
        noise = T.randn(size=(self.n_actions,)).to(self.actor.device)
        noise *= T.tensor(1 - int(evaluate))
        action = T.clamp(actions + noise,
                         T.tensor(-1.0, device=self.actor.device),
                         T.tensor(1.0, device=self.actor.device))
        return action.data.cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        tau = tau or self.tau

        src = self.actor
        dest = self.target_actor

        for param, target in zip(src.parameters(), dest.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

        src = self.critic
        dest = self.target_critic

        for param, target in zip(src.parameters(), dest.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self, memory, agent_list):
        if not memory.ready():
            return

        observations, states, actions, rewards,\
            observations_, states_, dones = memory.sample_buffer()

        device = self.actor.device

        states = T.tensor(np.array(states), dtype=T.float, device=device)
        actions = [T.tensor(actions[idx], device=device, dtype=T.float)
                   for idx in range(len(agent_list))]
        rewards = T.tensor(np.array(rewards), dtype=T.float, device=device)
        dones = T.tensor(np.array(dones), device=device)
        states_ = T.tensor(np.array(states_), dtype=T.float, device=device)

        # individual agent observations
        observations = [T.tensor(observations[idx],
                                 device=device, dtype=T.float)
                        for idx in range(len(agent_list))]
        # individual agent next observations
        observations_ = [T.tensor(observations_[idx],
                                     device=device, dtype=T.float)
                            for idx in range(len(agent_list))]


        with T.no_grad():
            new_actions = T.cat([agent.target_actor(observations_[idx])
                                 for idx, agent in enumerate(agent_list)],
                                dim=1)
            critic_value_ = self.target_critic.forward(
                                states_, new_actions).squeeze()
            critic_value_[dones[:, self.agent_idx]] = 0.0
            target = rewards[:, self.agent_idx] + self.gamma * critic_value_

        old_actions = T.cat([actions[idx] for idx in range(len(agent_list))],
                            dim=1)
        critic_value = self.critic.forward(states, old_actions).squeeze()
        critic_loss = F.mse_loss(target, critic_value)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic.optimizer.step()

        actions[self.agent_idx] = self.actor.forward(observations[self.agent_idx])
        actions = T.cat([actions[i] for i in range(len(agent_list))], dim=1)
        actor_loss = -self.critic.forward(states, actions).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor.optimizer.step()

        self.update_network_parameters()

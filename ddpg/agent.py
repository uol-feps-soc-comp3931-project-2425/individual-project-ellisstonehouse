import numpy as np
import torch as T
import torch.nn.functional as F
import os
from networks import ActorNetwork, CriticNetwork
from ddpg.buffer import ReplayBuffer


class Agent:
    def __init__(self, agent_idx, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 max_size=1000000, fc1_dims=400, fc2_dims=300,
                 batch_size=1024, chkpt_dir='models/DDPG/', model='sample'):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        agent_name = 'agent_%s' % agent_idx

        actor_dims = input_dims
        critic_dims = input_dims + n_actions

        fc1 = fc1_dims
        fc2 = fc2_dims


        chkpt_dir += model
        os.makedirs(chkpt_dir, exist_ok=True)

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        
        
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
        mu = self.actor.forward(obs).to(self.actor.device)
        noise = T.rand(self.n_actions).to(self.actor.device)
        noise *= T.tensor(1 - int(evaluate))
        mu_prime = mu + noise
        mu_prime = T.clamp(mu_prime, -1., 1.)

        return mu_prime.cpu().detach().numpy()[0]

    def remember(self, observation, action, reward, observation_, done):
        self.memory.store_transition(observation, action, reward, observation_, done)

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

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        observations, actions, rewards, observations_, done = \
            self.memory.sample_buffer(self.batch_size)

        observations = T.tensor(observations, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        observations_ = T.tensor(observations_, dtype=T.float).to(self.actor.device)

        new_actions = self.target_actor.forward(observations_)
        critic_value_ = self.target_critic.forward(observations_, new_actions)
        critic_value = self.critic.forward(observations, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(observations, self.actor.forward(observations))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

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

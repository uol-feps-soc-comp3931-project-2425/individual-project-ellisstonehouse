import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Define Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Output actions in the range [-1, 1]

# Define Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)



class DDPG:
    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.01, lr_actor=0.001, lr_critic=0.001):
        self.gamma = gamma
        self.tau = tau

        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)

        # Initialize target networks
        self.actor_target = Actor(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        # Copy weights from original to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.replay_buffer = deque(maxlen=100000)

    def act(self, state, noise_scale=0.1):
        # state = torch.FloatTensor(state).unsqueeze(0)
        state = torch.FloatTensor(state)
        action = self.actor(state).detach().numpy()
        action += noise_scale * np.random.normal(size=action.shape)
        return np.clip(action, -1, 1)
    

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        # Sample a batch of experiences
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        # Update critic
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_actions)
        target_Q = rewards + (1 - dones) * self.gamma * target_Q
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
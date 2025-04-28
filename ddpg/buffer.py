import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.observation_memory = np.zeros((self.mem_size, input_shape))
        self.new_observation_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, observation, action, reward, observation_, done):
        index = self.mem_cntr % self.mem_size
        self.observation_memory[index] = observation
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_observation_memory[index] = observation_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        observations = self.observation_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        observations_ = self.new_observation_memory[batch]
        dones = self.terminal_memory[batch]

        return observations, actions, rewards, observations_, dones

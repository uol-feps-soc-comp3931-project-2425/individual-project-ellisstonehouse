import numpy as np
import pybullet as p
import pybullet_data
import time
import random

from env import PredPreyEnv
from DDPG import DDPG

# Initialize environment
env = PredPreyEnv(arena_shape=(16, 16), num_predators=1, num_prey=1, GUI=True)
env.create_arena()

# Initialize DDPG agents
state_dim = 2  # Example: (x, y) position
action_dim = 2  # Example: (dx, dy) movement
predator_agent = DDPG(state_dim, action_dim)
prey_agent = DDPG(state_dim, action_dim)

# Training parameters
episodes = 10000
max_steps = 500
batch_size = 64
update_interval = 10  # Update networks every 10 steps

# Training loop
for episode in range(episodes):
    state = env.reset()
    episode_rewards = {"predator": 0, "prey": 0}


    for step in range(max_steps):
        # Get actions from agents
        
        predator_action = predator_agent.act(state["predator"])
        prey_action = prey_agent.act(state["prey"])


        
        # Take actions in the environment
        next_state, rewards, done = env.step(prey_action, predator_action)

        # Store experiences in replay buffers
        prey_agent.replay_buffer.append((state["prey"], prey_action, rewards["prey"], next_state["prey"], done))
        predator_agent.replay_buffer.append((state["predator"], predator_action, rewards["predator"], next_state["predator"], done))

        # Accumulate rewards
        episode_rewards["predator"] += rewards["predator"]
        episode_rewards["prey"] += rewards["prey"]

        # Update state
        state = next_state

        # Train agents periodically
        if step % update_interval == 0:
            prey_agent.update(batch_size)
            predator_agent.update(batch_size)

        # Check for episode termination
        if done:
            break

    # Log episode rewards
    print(f"Episode {episode + 1}: Predator Reward = {episode_rewards['predator']}, Prey Reward = {episode_rewards['prey']}")

# Close the environment
env.close()
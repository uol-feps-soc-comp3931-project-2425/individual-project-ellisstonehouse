import numpy as np
import pybullet as p
import pybullet_data
import time
import random

from env import PredPreyEnv


env = PredPreyEnv(arena_shape=(16, 16), num_predators=1, num_prey=1, GUI = True)



from math import sin, cos
episodes = 10000


env.create_arena()

### Start！

for i in range (episodes):

    prey_actions = [random.randint(0, 7) for _ in range(env.num_prey)]
    predator_actions = [random.randint(0, 7) for _ in range(env.num_predators)]
    
    next_state, rewards, done = env.step(prey_actions, predator_actions)



    # if(i%25 == 0):
    #     env.reset()


env.close()

    
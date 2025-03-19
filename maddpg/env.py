import numpy as np
import pybullet as p
import pybullet_data
import time
import random

PREDATOR = 1
PREY = 0


class PredPreyEnv():

    def __init__(self, num_predators=1, num_prey=1, GUI=False):


        self.home_start = [[x, y] for x in range(0, 2) for y in range(0, 6)]
        self.home_finish = [[x, y] for x in range(10, 12) for y in range(0, 6)]

        self.barriers = [[x, y] for x in range(-1, 13) for y in range(-1, 7) if x == -1 or x == 12 or y == -1 or y == 6]

        
        # Connect to PyBullet
        if GUI:
            self.physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
        else:
            self.physicsClient = p.connect(p.DIRECT)
        
        #p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.resetDebugVisualizerCamera(cameraDistance=6, cameraYaw=0, cameraPitch=-89, cameraTargetPosition = (6, 2, 6))

        p.setGravity(0,0,0)

        # Agent configuration
        self.num_predators = num_predators
        self.num_prey = num_prey

        # Agent variables
        self.agent_ids = []
        self.agent_roles = []
        self.agent_positions = []
        self.agent_velocities = []
        

        # Episode settings
        self.max_steps = 500
        self.current_step = 0
        self.reward_goal = 100
        self.reward_caught = -100
        self.reward_time_penalty = -1

        self.observation_space = []

        self.reset()



    

    def create_arena(self):

        p.loadURDF("plane_bulldog.urdf", [0, 0, 0], useFixedBase=True)

        
    def get_obs(self):

        obs = []

        templist = []

        # for each predator adds its coords, then concatenates on all prey coords
        for agent_id in range(len(self.agent_ids)):
            templist.append(self.agent_roles[agent_id])
            templist.extend(self.agent_positions[agent_id])
            templist.extend(self.agent_velocities[agent_id])
        
        obs = [np.array(templist, dtype=np.float32)] * 3
        
        return obs


    def calculate_rewards(self):

        rewards = np.zeros(self.num_predators + self.num_prey, dtype=np.float32)

        for agent_id, role in enumerate(self.agent_roles):
            
            # Rewards for Predators
            if role == PREDATOR:
                reward = 0.0
                # reward = -0.5  # Time penalty to encourage faster hunting

                predator_pos = self.agent_positions[agent_id]
                
                for prey_id, role2 in enumerate(self.agent_roles):
                    if role2 == PREY:
                        distance = np.linalg.norm(np.array(self.agent_positions[prey_id]) - np.array(predator_pos))
                        if distance < 1.0:  # Catch condition
                            reward += 100.0
                        # else:
                        #     reward += 2.0 / distance

                rewards[agent_id] = reward  # Assign reward to predator

            # # Rewards for Prey
            else:
                reward = 0.0
                # reward = 0.5

                prey_pos = self.agent_positions[agent_id]

                # Reward for reaching home
                for home_pos in self.home_finish:
                    if np.linalg.norm(np.array(prey_pos) - np.array(home_pos)) < 1.0:
                        reward += 100.0
                        break

                
                for pred_id, role2 in enumerate(self.agent_roles):
                    if role2 == PREDATOR:
                        distance = np.linalg.norm(np.array(self.agent_positions[pred_id]) - np.array(prey_pos))
                        if distance < 1.0:  # Catch condition
                            reward -= 100.0
                        # else:
                        #     reward -= 2.0 / distance
                
                rewards[agent_id] = reward

        return rewards  # Return as numpy array


    def done(self):

        dones = [False]*(self.num_predators + self.num_prey)

        # all prey are caught
        if all(self.agent_roles):
            dones = [True]*(self.num_predators + self.num_prey)

        prey_home = 0
        
        for agent_id, role in enumerate(self.agent_roles):
            if role == PREY:
                prey_pos = self.agent_positions[agent_id]

                for home_pos in self.home_finish:
                    if np.linalg.norm(np.array(prey_pos) - np.array(home_pos)) < 1.0:  # Home base condition
                        # prey_home+=1
                        # break
                        return [True]*(self.num_predators + self.num_prey)

        # if prey_home == (self.num_predators + self.num_prey - sum(self.agent_roles)):
        #     dones = [True]*(self.num_predators + self.num_prey)


        

        # # Check if predator catches a prey
        # for idx, predator_pos in enumerate(self.predator_positions):
        #     for prey_pos in self.prey_positions:
        #         if np.linalg.norm(np.array(prey_pos) - np.array(predator_pos)) < 1.0:  # Catch condition
        #             dones[idx] = True
        #             break

        # # Check if prey is caught
        # for idx, prey_pos in enumerate(self.prey_positions):
        #     for predator_pos in self.predator_positions:
        #         if np.linalg.norm(np.array(prey_pos) - np.array(predator_pos)) < 1.0:  # Catch condition
        #             dones[self.num_predators + idx] = True
        #             break
        
        #     # Check if prey is home
        #     for home_pos in self.home_finish:
        #         if np.linalg.norm(np.array(prey_pos) - np.array(home_pos)) < 1.0:  # Home base condition
        #             dones[self.num_predators + idx] = True
        #             break

        for agent_id, role in enumerate(self.agent_roles):
            
            # Rewards for Predators
            if role == 0:

                prey_pos = self.agent_positions[agent_id]

                for pred_id, role2 in enumerate(self.agent_roles):
                    if role2 == 1:
                        distance = np.linalg.norm(np.array(self.agent_positions[pred_id]) - np.array(prey_pos))
                        if distance < 1.0:  # Catch condition
                            p.changeVisualShape(self.agent_ids[agent_id], -1, rgbaColor=[1., 0.5, 0.5, 1])
                            self.agent_roles[agent_id] = 1


        return dones


    def change_velocity(self, agent, velocity):
        linVel, angVel = p.getBaseVelocity(agent)

        x = linVel[0] + velocity[0]
        y = linVel[1] + velocity[1]
        max_vel = 50
        
        if((x**2 + y**2)**.5 > max_vel):
            diff = (x**2 + y**2)**.5 / max_vel
            x /= diff
            y /= diff

        linVel = (x,y,0)
        p.resetBaseVelocity(agent, linVel, angVel)

    
    def step(self, actions):

        # Apply actions to predators
        for agent_id, agent in enumerate(self.agent_ids):
            self.change_velocity(agent, actions[agent_id])
            self.agent_positions[agent_id] = p.getBasePositionAndOrientation(agent)[0][0:2]

            vol, _ = p.getBaseVelocity(agent)
            self.agent_velocities[agent_id] = [vol[0], vol[1]]




        # Advance simulation
        p.stepSimulation()

        return self.get_obs(), self.calculate_rewards(), self.done()

    def render(self):
        pass
        

    def reset(self):

        for agent in self.agent_ids:
            p.removeBody(agent)
        
        # Set initial positions
        self.agent_ids = []
        self.agent_roles = []
        self.agent_positions = []
        self.agent_velocities = []

        sphereOrientation = p.getQuaternionFromEuler([0, 0, 0])

        for _ in range(self.num_predators):
            predator_pos = [5.5, 2.5]
            predator_id = p.loadURDF("sphere2red.urdf", [predator_pos[0], predator_pos[1], 0], sphereOrientation, globalScaling=1, useFixedBase=False)
            
            # Disable collisions with other objects
            p.setCollisionFilterGroupMask(predator_id, -1, collisionFilterGroup=1, collisionFilterMask=0)

            predator_vol, _ = p.getBaseVelocity(predator_id)

            self.agent_ids.append(predator_id)
            self.agent_roles.append(PREDATOR)
            self.agent_positions.append(predator_pos)
            self.agent_velocities.append([predator_vol[0], predator_vol[1]])


        # for prey_pos in [[0, 1], [0, 4]]:
        for _ in range(self.num_prey):
            prey_pos = random.choice(self.home_start)

            prey_id = p.loadURDF("sphere2red.urdf", [prey_pos[0], prey_pos[1], 0], sphereOrientation, globalScaling=1, useFixedBase=False)
            p.changeVisualShape(prey_id, -1, rgbaColor=[0.5, 0.5, 0.5, 1])

            # Disable collisions with other objects
            p.setCollisionFilterGroupMask(prey_id, -1, collisionFilterGroup=2, collisionFilterMask=0)

            prey_vol, _ = p.getBaseVelocity(prey_id)

            self.agent_ids.append(prey_id)
            self.agent_roles.append(PREY)
            self.agent_positions.append(prey_pos)
            self.agent_velocities.append([prey_vol[0], prey_vol[1]])


        self.agent_positions = np.array(self.agent_positions, dtype=np.float32)
        self.agent_velocities = np.array(self.agent_velocities, dtype=np.float32)


        return self.get_obs()

    def close(self):
        p.disconnect()


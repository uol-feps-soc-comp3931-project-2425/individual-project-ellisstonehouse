import numpy as np
import pybullet as p
import pybullet_data
import time
import random


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
        self.predator_ids = []
        self.prey_ids = []
        self.predator_positions = []
        self.prey_positions = []
        self.predator_velocities = []
        self.prey_velocities = []
        

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

        # for each predator adds its coords, then concatenates on all prey coords
        for pred in range(len(self.predator_ids)):
            templist = []

            templist.extend(self.predator_positions[pred])
            templist.extend(self.predator_velocities[pred])
            
            for prey in range(len(self.prey_ids)):
                templist.extend(self.prey_positions[prey])
                templist.extend(self.prey_velocities[prey])


            obs.append(np.array(templist, dtype=np.float32))
            # obs.append(templist)

        
        # for each prey adds its coords, then concatenates on all predator coords
        for prey in range(len(self.prey_ids)):
            templist = []

            templist.extend(self.prey_positions[prey])
            templist.extend(self.prey_velocities[prey])
            
            for pred in range(len(self.predator_ids)):
                templist.extend(self.predator_positions[pred])
                templist.extend(self.predator_velocities[pred])

            obs.append(np.array(templist, dtype=np.float32))
            # obs.append(templist)
        
        
        return obs


    def calculate_rewards(self):

        rewards = np.zeros(self.num_predators + self.num_prey, dtype=np.float32)

        # Rewards for Predators
        for i, predator_pos in enumerate(self.predator_positions):
            reward = -0.5  # Time penalty to encourage faster hunting

            for prey_pos in self.prey_positions:
                distance = np.linalg.norm(np.array(prey_pos) - np.array(predator_pos))
                if distance < 1.0:  # Catch condition
                    reward += 100.0
                else:
                    reward += 2.0 / distance

                # # Prey Reward for reaching home
                # for home_pos in self.home_finish:
                #     if np.linalg.norm(np.array(prey_pos) - np.array(home_pos)) < 1.0:
                #         reward -= 100.0
                #         break

            # for barrier_pos in self.barriers:
            #     if np.linalg.norm(np.array(predator_pos) - np.array(barrier_pos)) < 1.0:
            #         reward -= 1.0
            #         break

            rewards[i] = reward  # Assign reward to predator

        # Rewards for Prey
        for i, prey_pos in enumerate(self.prey_positions):
            reward = 0.5

            # Reward for reaching home
            for home_pos in self.home_finish:
                if np.linalg.norm(np.array(prey_pos) - np.array(home_pos)) < 1.0:
                    reward += 100.0
                    break

            # Penalty for getting caught
            for predator_pos in self.predator_positions:
                distance = np.linalg.norm(np.array(prey_pos) - np.array(predator_pos))
                if distance < 1.0:  # Catch condition
                    reward -= 100.0
                else:
                    reward -= 2.0 / distance

            # for barrier_pos in self.barriers:
            #     if np.linalg.norm(np.array(prey_pos) - np.array(barrier_pos)) < 1.0:
            #         reward -= 1.0
            #         break
            
            rewards[self.num_predators + i] = reward

        return rewards  # Return as numpy array


    def done(self):

        dones = [False]*(self.num_predators + self.num_prey)

        # Check if predator catches a prey
        for idx, predator_pos in enumerate(self.predator_positions):
            for prey_pos in self.prey_positions:
                if np.linalg.norm(np.array(prey_pos) - np.array(predator_pos)) < 1.0:  # Catch condition
                    dones[idx] = True
                    break

        # Check if prey is caught
        for idx, prey_pos in enumerate(self.prey_positions):
            for predator_pos in self.predator_positions:
                if np.linalg.norm(np.array(prey_pos) - np.array(predator_pos)) < 1.0:  # Catch condition
                    dones[self.num_predators + idx] = True
                    break
        
            # Check if prey is home
            for home_pos in self.home_finish:
                if np.linalg.norm(np.array(prey_pos) - np.array(home_pos)) < 1.0:  # Home base condition
                    dones[self.num_predators + idx] = True
                    break

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
        for i in range(self.num_predators):
            self.change_velocity(self.predator_ids[i], actions[i])
            self.predator_positions[i] = p.getBasePositionAndOrientation(self.predator_ids[i])[0][0:2]

            predator_vol, _ = p.getBaseVelocity(self.predator_ids[i])
            self.predator_velocities[i] = [predator_vol[0], predator_vol[1]]

        # Apply actions to prey
        for i in range(self.num_prey):
            self.change_velocity(self.prey_ids[i], actions[i + self.num_predators])
            self.prey_positions[i] = p.getBasePositionAndOrientation(self.prey_ids[i])[0][0:2]

            prey_vol, _ = p.getBaseVelocity(self.prey_ids[i])
            self.prey_velocities[i] = [prey_vol[0], prey_vol[1]]

        # Advance simulation
        p.stepSimulation()

        return self.get_obs(), self.calculate_rewards(), self.done()

    def render(self):
        pass
        

    def reset(self):

        for agent in self.predator_ids + self.prey_ids:
            p.removeBody(agent)
        
        # Set initial positions
        self.predator_ids = []
        self.prey_ids = []
        self.predator_positions = []
        self.prey_positions = []
        self.predator_velocities = []
        self.prey_velocities = []



        sphereOrientation = p.getQuaternionFromEuler([0, 0, 0])

        for _ in range(self.num_predators):
            predator_pos = [5.5, 2.5]
            predator_id = p.loadURDF("sphere2red.urdf", [predator_pos[0], predator_pos[1], 0], sphereOrientation, globalScaling=1, useFixedBase=False)
            
            # Disable collisions with other objects
            p.setCollisionFilterGroupMask(predator_id, -1, collisionFilterGroup=1, collisionFilterMask=0)

            predator_vol, _ = p.getBaseVelocity(predator_id)

            self.predator_ids.append(predator_id)
            self.predator_positions.append(predator_pos)
            self.predator_velocities.append([predator_vol[0], predator_vol[1]])


        # for prey_pos in [[0, 1], [0, 4]]:
        for _ in range(self.num_prey):
            prey_pos = random.choice(self.home_start)

            prey_id = p.loadURDF("sphere2red.urdf", [prey_pos[0], prey_pos[1], 0], sphereOrientation, globalScaling=1, useFixedBase=False)
            p.changeVisualShape(prey_id, -1, rgbaColor=[0.5, 0.5, 0.5, 1])

            # Disable collisions with other objects
            p.setCollisionFilterGroupMask(prey_id, -1, collisionFilterGroup=2, collisionFilterMask=0)

            prey_vol, _ = p.getBaseVelocity(prey_id)

            self.prey_ids.append(prey_id)
            self.prey_positions.append(prey_pos)
            self.prey_velocities.append([prey_vol[0], prey_vol[1]])


        self.predator_positions = np.array(self.predator_positions, dtype=np.float32)
        self.prey_positions = np.array(self.prey_positions, dtype=np.float32)
        self.predator_velocities = np.array(self.predator_positions, dtype=np.float32)
        self.prey_velocities = np.array(self.prey_positions, dtype=np.float32)


        return self.get_obs()

    def close(self):
        p.disconnect()


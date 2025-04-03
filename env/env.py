import numpy as np
import pybullet as p
import pybullet_data
import time
import random

BULLDOG = 1
RUNNER = 0

COL_GROUP_WALLS = 1
COL_GROUP_AREA = 2
COL_GROUP_RUNNER = 4
COL_GROUP_BULLDOG = 8


class BritishBulldogEnv():

    def __init__(self, num_bulldogs=1, num_runner=1, GUI=False):


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
        self.num_bulldogs = num_bulldogs
        self.num_runner = num_runner

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

        self.observation_space = [5*(num_bulldogs+num_runner)]*(num_bulldogs+num_runner)
        self.action_space = [2]*(num_bulldogs+num_runner)

        # self.reset()



    

    def create_arena(self):

        walls_id = p.loadURDF("env/walls.urdf", [0, 0, 0], useFixedBase=True)
        p.setCollisionFilterGroupMask(walls_id, -1, COL_GROUP_WALLS, COL_GROUP_RUNNER | COL_GROUP_BULLDOG)

        hb_start_id = p.loadURDF("env/homebase_start.urdf", [0, 0, 0], useFixedBase=True)
        p.setCollisionFilterGroupMask(hb_start_id, -1, COL_GROUP_AREA, COL_GROUP_BULLDOG)

        hb_finish_id = p.loadURDF("env/homebase_finish.urdf", [0, 0, 0], useFixedBase=True)
        p.setCollisionFilterGroupMask(hb_finish_id, -1, COL_GROUP_AREA, COL_GROUP_BULLDOG)

        
    def get_obs(self):

        obs = []

        templist = []

        # for each bulldog adds its coords, then concatenates on all runner coords
        for agent_id in range(len(self.agent_ids)):
            templist.append(self.agent_roles[agent_id])
            templist.extend(self.agent_positions[agent_id])
            templist.extend(self.agent_velocities[agent_id])
        
        obs = [np.array(templist, dtype=np.float32)] * 3
        
        return obs


    def calculate_rewards(self):

        rewards = np.zeros(self.num_bulldogs + self.num_runner, dtype=np.float32)

        for agent_id, role in enumerate(self.agent_roles):
            
            # Rewards for Predators
            if role == BULLDOG:
                # reward = 0.0
                reward = -0.5  # Time penalty to encourage faster hunting

                bulldog_pos = self.agent_positions[agent_id]
                
                for runner_id, role2 in enumerate(self.agent_roles):
                    if role2 == RUNNER:
                        distance = np.linalg.norm(np.array(self.agent_positions[runner_id]) - np.array(bulldog_pos))
                        if distance < 1.0:  # Catch condition
                            reward += 100.0
                        else:
                            reward += 2.0 / distance

                rewards[agent_id] = reward  # Assign reward to bulldog

            # # Rewards for runner
            else:
                # reward = 0.0
                reward = 0.5

                runner_pos = self.agent_positions[agent_id]

                # Reward for reaching home
                for home_pos in self.home_finish:
                    if np.linalg.norm(np.array(runner_pos) - np.array(home_pos)) < 1.0:
                        reward += 10.0
                        break

                
                for bulldog_id, role2 in enumerate(self.agent_roles):
                    if role2 == BULLDOG:
                        distance = np.linalg.norm(np.array(self.agent_positions[bulldog_id]) - np.array(runner_pos))
                        if distance < 1.0:  # Catch condition
                            reward -= 100.0
                        else:
                            reward -= 2.0 / distance
                
                rewards[agent_id] = reward

        return rewards  # Return as numpy array


    def done(self):

        dones = [False]*(self.num_bulldogs + self.num_runner)

        # all runner are caught
        if all(self.agent_roles):
            dones = [True]*(self.num_bulldogs + self.num_runner)

        runner_home = 0
        
        for agent_id, role in enumerate(self.agent_roles):
            if role == RUNNER:
                runner_pos = self.agent_positions[agent_id]

                for home_pos in self.home_finish:
                    if np.linalg.norm(np.array(runner_pos) - np.array(home_pos)) < 1.0:  # Home base condition
                        runner_home+=1
                        break
                        # return [True]*(self.num_bulldogs + self.num_runner)

        if runner_home == (self.num_bulldogs + self.num_runner - sum(self.agent_roles)):
            dones = [True]*(self.num_bulldogs + self.num_runner)


        

        # # Check if bulldog catches a runner
        # for idx, bulldog_pos in enumerate(self.bulldog_positions):
        #     for runner_pos in self.runner_positions:
        #         if np.linalg.norm(np.array(runner_pos) - np.array(bulldog_pos)) < 1.0:  # Catch condition
        #             dones[idx] = True
        #             break

        # # Check if runner is caught
        # for idx, runner_pos in enumerate(self.runner_positions):
        #     for bulldog_pos in self.bulldog_positions:
        #         if np.linalg.norm(np.array(runner_pos) - np.array(bulldog_pos)) < 1.0:  # Catch condition
        #             dones[self.num_bulldogs + idx] = True
        #             break
        
        #     # Check if runner is home
        #     for home_pos in self.home_finish:
        #         if np.linalg.norm(np.array(runner_pos) - np.array(home_pos)) < 1.0:  # Home base condition
        #             dones[self.num_bulldogs + idx] = True
        #             break

        for agent_id, role in enumerate(self.agent_roles):
            
            # Rewards for Bulldogs
            if role == 0:

                runner_pos = self.agent_positions[agent_id]

                for bulldog_id, role2 in enumerate(self.agent_roles):
                    if role2 == 1:
                        distance = np.linalg.norm(np.array(self.agent_positions[bulldog_id]) - np.array(runner_pos))
                        if distance < 1.0:  # Catch condition
                            p.changeVisualShape(self.agent_ids[agent_id], -1, rgbaColor=[1., 0.5, 0.5, 1])
                            p.setCollisionFilterGroupMask(self.agent_ids[agent_id], -1, COL_GROUP_BULLDOG, COL_GROUP_WALLS | COL_GROUP_AREA)
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

        for agent_id, agent in enumerate(self.agent_ids):

            # apply action to velocity
            linVel, angVel = p.getBaseVelocity(agent)

            x = linVel[0] + actions[agent_id][0]
            y = linVel[1] + actions[agent_id][1]
            max_vel = 50
            
            if((x**2 + y**2)**.5 > max_vel):
                diff = (x**2 + y**2)**.5 / max_vel
                x /= diff
                y /= diff

            linVel = (x,y,0)

            # lock the z axis to 0 due to collisions
            pos, ori = p.getBasePositionAndOrientation(agent)
            self.agent_positions[agent_id] = pos[0:2]
            p.resetBasePositionAndOrientation(agent, [pos[0], pos[1], 0], ori)

            p.resetBaseVelocity(agent, linVel, angVel)

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

        bulldog_id =0

        for _ in range(self.num_bulldogs):
            bulldog_pos = [5.5, 2.5]
            bulldog_id = p.loadURDF("sphere2red.urdf", [bulldog_pos[0], bulldog_pos[1], 0], sphereOrientation, globalScaling=1, useFixedBase=False)
            
            # Disable collisions with other objects
            p.setCollisionFilterGroupMask(bulldog_id, -1, COL_GROUP_BULLDOG, COL_GROUP_WALLS | COL_GROUP_AREA)

            # p.setCollisionFilterPair(bulldog_id, self.homebases_id, -1, -1, enableCollision=True)

            bulldog_vol, _ = p.getBaseVelocity(bulldog_id)

            self.agent_ids.append(bulldog_id)
            self.agent_roles.append(BULLDOG)
            self.agent_positions.append(bulldog_pos)
            self.agent_velocities.append([bulldog_vol[0], bulldog_vol[1]])


        # for runner_pos in [[0, 1], [0, 4]]:
        for _ in range(self.num_runner):
            runner_pos = random.choice(self.home_start)

            runner_id = p.loadURDF("sphere2red.urdf", [runner_pos[0], runner_pos[1], 0], sphereOrientation, globalScaling=1, useFixedBase=False)
            p.changeVisualShape(runner_id, -1, rgbaColor=[0.5, 0.5, 0.5, 1])

            # Disable collisions with other objects
            p.setCollisionFilterGroupMask(runner_id, -1, COL_GROUP_RUNNER, COL_GROUP_WALLS)

            runner_vol, _ = p.getBaseVelocity(runner_id)

            self.agent_ids.append(runner_id)
            self.agent_roles.append(RUNNER)
            self.agent_positions.append(runner_pos)
            self.agent_velocities.append([runner_vol[0], runner_vol[1]])


        self.agent_positions = np.array(self.agent_positions, dtype=np.float32)
        self.agent_velocities = np.array(self.agent_velocities, dtype=np.float32)


        return self.get_obs()

    def close(self):
        p.disconnect()


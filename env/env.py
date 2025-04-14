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

    def __init__(self, init_bulldogs, init_runners, GUI=False):

        self.home_start = [[x, y] for x in range(0, 2) for y in range(0, 6)]
        self.home_finish = [[x, y] for x in range(10, 12) for y in range(0, 6)]

        # Connect to PyBullet
        if GUI:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        
        #p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(cameraDistance=6, cameraYaw=0, cameraPitch=-89, cameraTargetPosition = (6, 2, 6))
        p.setGravity(0,0,0)

        # Agent configuration
        self.init_bulldogs = init_bulldogs
        self.init_runners = init_runners
        self.n_agents = init_bulldogs + init_runners

        self.runners_home = [False]*self.n_agents

        # Agent variables
        self.agent_ids = []
        self.agent_roles = []
        self.agent_positions = []
        self.agent_velocities = []

        # for each agent: x, y, vx, vy, role
        self.observation_space = [5*(self.n_agents)]*(self.n_agents)
        # for each agent: vx, vy
        self.action_space = [2]*(self.n_agents)


        self.create_arena()


    

    def create_arena(self):

        walls_id = p.loadURDF("env/walls.urdf", [0, 0, 0], useFixedBase=True)
        p.setCollisionFilterGroupMask(walls_id, -1, COL_GROUP_WALLS, COL_GROUP_RUNNER | COL_GROUP_BULLDOG)

        hb_start_id = p.loadURDF("env/homebase_start.urdf", [0, 0, 0], useFixedBase=True)
        p.setCollisionFilterGroupMask(hb_start_id, -1, COL_GROUP_AREA, COL_GROUP_BULLDOG)

        hb_finish_id = p.loadURDF("env/homebase_finish.urdf", [0, 0, 0], useFixedBase=True)
        p.setCollisionFilterGroupMask(hb_finish_id, -1, COL_GROUP_AREA, COL_GROUP_BULLDOG)

        
    def get_obs(self):

        obs = np.concatenate([self.agent_roles,
                                self.agent_positions.flatten(),
                                self.agent_velocities.flatten()]).astype(np.float32)
        
        obs = [obs] * self.n_agents
        
        return obs


    def calculate_rewards(self):

        rewards = np.zeros(self.n_agents, dtype=np.float32)
        # rewards = np.array([-0.01, -0.01, -0.01], dtype=np.float32)

        role_changes = []

        for runner_id, role in enumerate(self.agent_roles):
            if role == RUNNER and not self.runners_home[runner_id]:

                runner_pos = self.agent_positions[runner_id]
                
                for bulldog_id, role2 in enumerate(self.agent_roles):
                    if role2 == BULLDOG:

                        bulldog_pos = self.agent_positions[bulldog_id]

                        # Rewards for runner reaching home
                        for home_pos in self.home_finish:
                            if np.linalg.norm(np.array(runner_pos) - np.array(home_pos)) < 1.0:
                                rewards[bulldog_id] -= 100.0 / self.agent_roles.count(BULLDOG)
                                rewards[runner_id] += 100.0 / self.agent_roles.count(BULLDOG)

                                self.runners_home[runner_id] = True
                                break

                        distance = np.linalg.norm(np.array(bulldog_pos) - np.array(runner_pos))
                        
                        # Rewards for runner getting caught
                        if distance < 0.75:
                            rewards[bulldog_id] += 100.0
                            rewards[runner_id] -= 100.0

                            role_changes.append(runner_id)

                        else:
                            rewards[bulldog_id] += (5 - distance) / 10
                            rewards[runner_id] += (distance - 5) / 10

        for runner_id in role_changes:
            p.changeVisualShape(self.agent_ids[runner_id], -1, rgbaColor=[1., 0.5, 0.5, 1])
            p.setCollisionFilterGroupMask(self.agent_ids[runner_id], -1, COL_GROUP_BULLDOG, COL_GROUP_WALLS | COL_GROUP_AREA)
            self.agent_roles[runner_id] = BULLDOG

        return rewards


    def done(self):

        dones = [False]*(self.n_agents)

        # all runners are caught
        if all(self.agent_roles):
            dones = [True]*(self.n_agents)
            return dones

        # all runners are home
        if sum(self.runners_home) == self.agent_roles.count(RUNNER):
            dones = [True]*(self.n_agents)

        return dones


    
    def step(self, actions):

        for agent_id, agent in enumerate(self.agent_ids):

            # apply action to velocity
            linVel, angVel = p.getBaseVelocity(agent)

            x = linVel[0] + actions[agent_id][0]
            y = linVel[1] + actions[agent_id][1]

            max_vel = 20
            
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

        return self.agent_roles, self.get_obs(), self.calculate_rewards(), self.done()


    def reset(self):

        for agent in self.agent_ids:
            p.removeBody(agent)
        
        # Set initial positions
        self.agent_ids = []
        self.agent_roles = []
        self.agent_positions = []
        self.agent_velocities = []

        self.runners_home = [False]*self.n_agents

        sphereOrientation = p.getQuaternionFromEuler([0, 0, 0])

        for _ in range(self.init_bulldogs):
            bulldog_pos = [5.5, 2.5]
            bulldog_id = p.loadURDF("sphere2red.urdf", [bulldog_pos[0], bulldog_pos[1], 0], sphereOrientation, globalScaling=1, useFixedBase=False)
            
            # Sets collisions with walls and homebases
            p.setCollisionFilterGroupMask(bulldog_id, -1, COL_GROUP_BULLDOG, COL_GROUP_WALLS | COL_GROUP_AREA)

            bulldog_vol, _ = p.getBaseVelocity(bulldog_id)

            self.agent_ids.append(bulldog_id)
            self.agent_roles.append(BULLDOG)
            self.agent_positions.append(bulldog_pos)
            self.agent_velocities.append([bulldog_vol[0], bulldog_vol[1]])

        for _ in range(self.init_runners):
            runner_pos = random.choice(self.home_start)

            runner_id = p.loadURDF("sphere2red.urdf", [runner_pos[0], runner_pos[1], 0], sphereOrientation, globalScaling=1, useFixedBase=False)
            p.changeVisualShape(runner_id, -1, rgbaColor=[0.5, 0.5, 0.5, 1])

            # Sets collisions with walls
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


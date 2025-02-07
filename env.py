import numpy as np
import pybullet as p
import pybullet_data
import time
import random


class PredPreyEnv():

    def __init__(self, arena_shape=(16, 16), num_predators=1, num_prey=1, GUI=False):

        # Arena configuration
        self.arena_shape = arena_shape
        self.w, self.h = arena_shape

        self.home_start = np.array([(x, y) for x in range(1, 3) for y in range(1, self.h-1)])
        self.home_finish = np.array([(x, y) for x in range(self.w-3, self.w-1) for y in range(1, self.h-1)])

        
        # Connect to PyBullet
        if GUI:
            self.physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
        else:
            self.physicsClient = p.connect(p.DIRECT)
        
        #p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.resetDebugVisualizerCamera(cameraDistance=6, cameraYaw=0, cameraPitch=-89, cameraTargetPosition = (self.w/2, self.h/2, self.w/2))

        p.setGravity(0,0,0)

        # Agent configuration
        self.starting_num_predators = num_predators
        self.starting_num_prey = num_prey

        self.num_predators = num_predators
        self.num_prey = num_prey

        # Agent variables
        self.predator_ids = []
        self.prey_ids = []
        self.predator_positions = []
        self.prey_positions = []
        

        # Episode settings
        self.max_steps = 500
        self.current_step = 0
        self.reward_goal = 100
        self.reward_caught = -100
        self.reward_time_penalty = -1



    

    def create_arena(self):

        p.loadURDF("plane_bulldog.urdf", [0, 0, 0], useFixedBase=True)

        
    
    def _get_state(self):
        # Return the positions of all agents as the state

        # print ("get state", self.predator_positions)
        # print ("now np ", np.array(self.predator_positions))

        state = {
            "predator": np.array(self.predator_positions),
            "prey": np.array(self.prey_positions),
        }
        return state

    def calculate_rewards(self):

        predator_rewards = 0
        prey_rewards = 0

        # Check if predator catches prey
        for predator_pos in self.predator_positions:

            for prey_pos in self.prey_positions:
                if np.linalg.norm(prey_pos - predator_pos) < 1.0:  # Catch condition
                    predator_rewards += 10
            
            predator_rewards -= 0.01


            # ADD IN A DEDUCTION USING THE TOTAL ELPASED TIME TO URGE PRED TO CATCH FASTER

        
        # Check if prey is caught by any predator or is home
        for prey_pos in self.prey_positions:

            for home_pos in self.home_finish:
                if np.linalg.norm(prey_pos - home_pos) < 1.0:  # Home base condition
                    prey_rewards += 10
                    break
            
            for predator_pos in self.predator_positions:
                if np.linalg.norm(prey_pos - predator_pos) < 1.0:  # Catch condition
                    prey_rewards -= 10
            
            for home_pos in self.home_start:
                if np.linalg.norm(prey_pos - home_pos) < 1.0:  # Home base condition
                    prey_rewards -= 0.1
                    break
            
            prey_rewards += 0.01
            

        rewards = {
            "predator": np.array(predator_rewards),
            "prey": np.array(prey_rewards),
        }

        return rewards


    def done(self):

        prey_caught = 0

        # Check if predator catches all prey
        for prey_pos in self.prey_positions:
            for predator_pos in self.predator_positions:
                if np.linalg.norm(prey_pos - predator_pos) < 1.0:  # Catch condition
                    prey_caught += 1
                break
            
        if prey_caught == self.num_prey:
            return True
        

        prey_home = 0

        # Check if predator catches all prey
        for prey_pos in self.prey_positions:
            for home_pos in self.home_finish:
                if np.linalg.norm(prey_pos - home_pos) < 1.0:  # Home base condition
                    prey_home += 1
                    break

            
        if prey_home == self.num_prey:
            return True
        

        return False




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

    
    def step(self, predator_actions, prey_actions):
        # # Define moves
        # moves = np.array([
        #     [0, 1],   # Up
        #     [0, -1],  # Down
        #     [-1, 0],  # Left
        #     [1, 0],   # Right
        #     [-1, 1],  # Up-Left
        #     [1, 1],   # Up-Right
        #     [-1, -1], # Down-Left
        #     [1, -1],  # Down-Right
        # ])


        # Apply actions to predators
        for i in range(self.num_predators):
            self.change_velocity(self.predator_ids[i], predator_actions[0])
            self.predator_positions[i] = np.array(p.getBasePositionAndOrientation(self.predator_ids[i])[0][0:2])

        # Apply actions to prey
        for i in range(self.num_prey):
            self.change_velocity(self.prey_ids[i], prey_actions[0])
            self.prey_positions[i] = np.array(p.getBasePositionAndOrientation(self.prey_ids[i])[0][0:2])


        # Advance simulation
        p.stepSimulation()
        time.sleep(1 / 240)


        return self._get_state(), self.calculate_rewards(), self.done()


        

    def reset(self):

        for agent in self.predator_ids + self.prey_ids:
            p.removeBody(agent)
        
        # Set initial positions
        self.predator_ids = []
        self.prey_ids = []
        self.predator_positions = []
        self.prey_positions = []

        sphereOrientation = p.getQuaternionFromEuler([0, 0, 0])

        for _ in range(self.starting_num_predators):
            predator_pos = np.array([self.w/2, self.h/2], dtype=np.int32)
            predator_id = p.loadURDF("sphere2red.urdf", [predator_pos[0], predator_pos[1], 0], sphereOrientation, globalScaling=1, useFixedBase=False)
            p.changeVisualShape(predator_id, -1, rgbaColor=[1, 0, 0, 1])
            self.predator_ids.append(predator_id)
            self.predator_positions.append(predator_pos)

        for _ in range(self.starting_num_prey):
            prey_pos = random.choice(self.home_start).flatten()
            prey_id = p.loadURDF("sphere2red.urdf", [prey_pos[0], prey_pos[1], 0], sphereOrientation, globalScaling=1, useFixedBase=False)
            p.changeVisualShape(prey_id, -1, rgbaColor=[0.5, 0.5, 0.5, 1])
            self.prey_ids.append(prey_id)
            self.prey_positions.append(prey_pos)


        # # Create prey and predator objects in PyBullet
        # self.prey_id = p.loadURDF("sphere2red.urdf", [self.prey_position[0], self.prey_position[1], 0.5])
        # self.predator_id = p.loadURDF("sphere2red.urdf", [self.predator_position[0], self.predator_position[1], 0.5])

        return self._get_state()

    def close(self):
        p.disconnect()


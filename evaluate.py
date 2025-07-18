import numpy as np
from ddpg.agent import Agent as DDPG_Agent
from maddpg.agent import Agent as MADDPG_Agent
from env.env import BritishBulldogEnv

import os


BULLDOG = 1
RUNNER = 0

MADDPG = 0
DDPG = 1
RANDOM = 2

EPISODES = 1000
MAX_STEPS = 500
PRINT_INTERVAL = 100

ALPHA = 1
BETA = 1
GAMMA = 1
TAU = 1

bulldog_algo = MADDPG
runner_algo = DDPG

bulldog_model = 'model_1'
runner_model = 'model_1'


file = ''

if bulldog_algo == RANDOM:
    file += 'RANDOM_vs_'
elif bulldog_algo == DDPG:
    file += 'DDPG_'+bulldog_model+'_vs_'
else:
    file += 'MADDPG_'+bulldog_model+'_vs_'

if runner_algo == RANDOM:
    file += 'RANDOM'
elif runner_algo == DDPG:
    file += 'DDPG_'+runner_model
else:
    file += 'MADDPG_'+runner_model

# os.makedirs('results/evaluate/'+file, exist_ok=True)


def run():
    
    env = BritishBulldogEnv(init_bulldogs=1, init_runners=2, GUI=True)

    total_steps = 0
    bulldog_score_history = []
    runner_score_history = []
    episodes = []

    bulldog_agents = []
    if bulldog_algo == MADDPG:
        actor_dims = []
        n_actions = []
        for i in range(env.n_agents):
            actor_dims.append(env.observation_space[i])
            n_actions.append(env.action_space[i])

        critic_dims = actor_dims[0] + sum(n_actions)


        for agent_idx in range(env.n_agents):
            bulldog_agents.append(MADDPG_Agent(actor_dims[agent_idx], critic_dims,
                                n_actions[agent_idx], env.n_agents, agent_idx,
                                alpha=ALPHA, beta=BETA, gamma=GAMMA, tau=TAU, 
                                fc1=64, fc2=64, model=bulldog_model))

        for agent in bulldog_agents:
            agent.load_models()

    elif bulldog_algo == DDPG:
        for agent_idx in range(env.n_agents):
            input_dims = env.observation_space[agent_idx]
            n_actions = env.action_space[agent_idx]

            bulldog_agents.append(DDPG_Agent(agent_idx, alpha=ALPHA, beta=BETA,
                        input_dims=input_dims, tau=TAU, gamma=GAMMA,
                        batch_size=1024, fc1_dims=64, fc2_dims=64,
                        n_actions=n_actions, model=bulldog_model))

        for agent in bulldog_agents:
            agent.load_models()


    runner_agents = []
    if runner_algo == MADDPG:
        actor_dims = []
        n_actions = []
        for i in range(env.n_agents):
            actor_dims.append(env.observation_space[i])
            n_actions.append(env.action_space[i])

        critic_dims = actor_dims[0] + sum(n_actions)

        for agent_idx in range(env.n_agents):
            runner_agents.append(MADDPG_Agent(actor_dims[agent_idx], critic_dims,
                                n_actions[agent_idx], env.n_agents, agent_idx,
                                alpha=ALPHA, beta=BETA, gamma=GAMMA, tau=TAU, 
                                fc1=64, fc2=64, model=runner_model))

        for agent in runner_agents:
            agent.load_models()

    elif runner_algo == DDPG:
        for agent_idx in range(env.n_agents):
            input_dims = env.observation_space[agent_idx]
            n_actions = env.action_space[agent_idx]

            runner_agents.append(DDPG_Agent(agent_idx, alpha=ALPHA, beta=BETA,
                        input_dims=input_dims, tau=TAU, gamma=GAMMA,
                        batch_size=1024, fc1_dims=64, fc2_dims=64,
                        n_actions=n_actions, model=runner_model))

        for agent in runner_agents:
            agent.load_models()



    for episode in range(EPISODES):
        roles = [BULLDOG]*env.init_bulldogs + [RUNNER]*env.init_runners
        observation = env.reset()
        done = [False]*env.n_agents
        episode_step = 0
        bulldog_score = 0
        runner_score = 0

        while not all(done):

            actions = [0]*env.n_agents

            # action mask depending on chosen algorithm
            for agent_idx, role in enumerate(roles):
                if role == BULLDOG:
                    if bulldog_algo == MADDPG or bulldog_algo == DDPG:
                        actions[agent_idx] = bulldog_agents[agent_idx].choose_action(observation[agent_idx])
                    else: # RANDOM
                        actions[agent_idx] = np.random.uniform(-1.0, 1.0, size=2)
                        
                else: #RUNNER
                    if bulldog_algo == MADDPG or bulldog_algo == DDPG:
                        actions[agent_idx] = bulldog_agents[agent_idx].choose_action(observation[agent_idx])
                    else: # RANDOM
                        actions[agent_idx] = np.random.uniform(-1.0, 1.0, size=2)

            roles_, observation_, rewards, done = env.step(actions)

            # end episode if max steps reached
            if episode_step >= MAX_STEPS:
                done = [True]*env.n_agents

            observation = observation_

            for idx, role in enumerate(roles):
                if role == BULLDOG:
                    bulldog_score += rewards[idx]
                else: # Runner
                    runner_score += rewards[idx]

            roles = roles_.copy()
            
            total_steps += 1
            episode_step += 1
        
        bulldog_score_history.append(bulldog_score)
        runner_score_history.append(runner_score)
        episodes.append(episode)

        if episode % PRINT_INTERVAL == 0:
            bulldog_avg_score = np.mean(bulldog_score_history[-PRINT_INTERVAL:])
            runner_avg_score = np.mean(runner_score_history[-PRINT_INTERVAL:])
            print(f'Episode {episode}, last {PRINT_INTERVAL} avg, bd score {bulldog_avg_score:.1f}, r score {runner_avg_score:.1f}')

    env.close()


if __name__ == '__main__':
    
    run()
        
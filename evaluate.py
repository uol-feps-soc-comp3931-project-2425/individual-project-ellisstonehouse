import numpy as np
from maddpg.maddpg import MADDPG
from ddpg.agent import Agent
from env.env import BritishBulldogEnv


BULLDOG = 1
RUNNER = 0

MADDPG_ = 0
DDPG_ = 1
RANDOM_ = 2

EPISODES = 10_000
MAX_STEPS = 500
PRINT_INTERVAL = 100

bulldog_algo = MADDPG_
runner_algo = MADDPG_

maddpg_model = 'model_1'
ddpg_model = 'model_1'


def run():
    
    env = BritishBulldogEnv(init_bulldogs=1, init_runners=2, GUI=True)

    total_steps = 0
    bulldog_score_history = []
    runner_score_history = []
    episodes = []


    if bulldog_algo == MADDPG_ or runner_algo == MADDPG_:
        actor_dims = []
        n_actions = []
        for i in range(env.n_agents):
            actor_dims.append(env.observation_space[i])
            n_actions.append(env.action_space[i])

        critic_dims = sum(actor_dims) + sum(n_actions)


        maddpg_agents = MADDPG(actor_dims=actor_dims, critic_dims=critic_dims,
                               n_agents=env.n_agents, n_actions=n_actions,
                               model=maddpg_model)

        maddpg_agents.load_checkpoint()

    ddpg_agents = []
    if bulldog_algo == DDPG_ or runner_algo == DDPG_:
        for agent_idx in range(env.n_agents):
            input_dims = env.observation_space[0]
            n_actions = env.action_space[0]

            ddpg_agents.append(Agent(agent_idx, alpha=1e-3, beta=1e-3,
                        input_dims=input_dims, tau=0.01, gamma=0.95,
                        batch_size=1024, fc1_dims=64, fc2_dims=64,
                        n_actions=n_actions, model=ddpg_model))

        for agent in ddpg_agents:
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

            if bulldog_algo == MADDPG_ or runner_algo == MADDPG_:
                maddpg_actions = maddpg_agents.choose_action(observation, evaluate=True)

            # action mask depending on chosen algorithm
            for agent_idx, role in enumerate(roles):
                if role == BULLDOG:
                    if bulldog_algo == MADDPG_:
                        actions[agent_idx] = maddpg_actions[agent_idx]
                    elif bulldog_algo == DDPG_:
                        actions[agent_idx] = ddpg_agents[agent_idx].choose_action(observation[agent_idx])
                    else: # RANDOM
                        actions[agent_idx] = np.random.uniform(-1.0, 1.0, size=2)
                        
                else: #RUNNER
                    if runner_algo == MADDPG_:
                        actions[agent_idx] = maddpg_actions[agent_idx]
                    elif runner_algo == DDPG_:
                        actions[agent_idx] = ddpg_agents[agent_idx].choose_action(observation[agent_idx])
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
            bulldog_avg_score = np.mean(bulldog_score_history[-100:])
            runner_avg_score = np.mean(runner_score_history[-100:])
            episodes.append(episode)
            print(f'Episode {episode}, last 100 avg, bd score {bulldog_avg_score:.1f}, r score {runner_avg_score:.1f}')

    np.save('results/evaluate/'+maddpg_model+'.npy', np.array(bulldog_score_history))
    np.save('results/evaluate/'+ddpg_model+'.npy', np.array(runner_score_history))
    np.save('results/evaluate/episodes.npy', np.array(episodes))

    env.close()


if __name__ == '__main__':
    run()
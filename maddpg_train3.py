import numpy as np
from maddpg.buffer import MultiAgentReplayBuffer
from maddpg.agent import Agent
from env.env import BritishBulldogEnv
import os

BULLDOG = 1
RUNNER = 0
MADDPG = 0
RANDOM = 1

EPISODES = 10000
MAX_STEPS = 500
PRINT_INTERVAL = 100

ALPHA = 1e-3
BETA = 1e-3
GAMMA = 0.95
TAU = 0.01

bulldog_algo = MADDPG
runner_algo = MADDPG

model = 'model_3'
os.makedirs('results/MADDPG/'+model, exist_ok=True)


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

def run():

    env = BritishBulldogEnv(init_bulldogs=1, init_runners=2, GUI=False)

    total_steps = 0
    scores_history = []
    bulldog_score_history = []
    runner_score_history = []
    episodes = []


    actor_dims = []
    n_actions = []
    for i in range(env.n_agents):
        actor_dims.append(env.observation_space[i])
        n_actions.append(env.action_space[i])
    # critic_dims = sum(actor_dims) + sum(n_actions)
    critic_dims = env.observation_space[i] + sum(n_actions)
    

    agents = []

    for agent_idx in range(env.n_agents):
        agents.append(Agent(actor_dims[agent_idx], critic_dims,
                            n_actions[agent_idx], env.n_agents, agent_idx,
                            alpha=ALPHA, beta=BETA, gamma=GAMMA, tau=TAU, 
                            fc1=64, fc2=64, model=model))
    

    # critic_dims = sum(actor_dims)
    critic_dims = env.observation_space[i]

    memory = MultiAgentReplayBuffer(1_000_000, critic_dims, actor_dims,
                                    n_actions, env.n_agents, batch_size=1024)
    

    for episode in range(EPISODES+1):
        roles = [BULLDOG]*env.init_bulldogs + [RUNNER]*env.init_runners
        observation = env.reset()
        done = [False]*env.n_agents
        episode_step = 0
        scores = np.zeros(env.n_agents)
        bulldog_score = 0
        runner_score = 0

        while not all(done):

            # eval false includes noise for better exploration
            actions = [agent.choose_action(observation[idx], evaluate=False)
                       for idx, agent in enumerate(agents)]
            
            
            # for training against random roles
            for idx in range(len(actions)):
                if roles[idx] == BULLDOG and bulldog_algo == RANDOM:
                    actions[idx] = np.random.uniform(-1.0, 1.0, size=2)
                elif roles[idx] == RUNNER and runner_algo == RANDOM:
                    actions[idx] = np.random.uniform(-1.0, 1.0, size=2)

            roles_, observation_, rewards, done = env.step(actions)

            # state = obs_list_to_state_vector(observation)
            # state_ = obs_list_to_state_vector(observation_)

            # set full state to single agent observation, since each agent views full state,
            # they are all the same
            state = observation[0]
            state_ = observation_[0]

            # store experience
            memory.store_transition(observation, state, actions, rewards,
                        observation_, state_, done)

            # end episode if max steps reached
            if episode_step >= MAX_STEPS:
                done = [True]*env.n_agents

            # every 100 steps learn
            if total_steps % 100 == 0:
                for agent in agents:
                    agent.learn(memory, agents)

            observation = observation_

            scores += rewards
            
            for idx, role in enumerate(roles):
                if role == BULLDOG:
                    bulldog_score += rewards[idx]
                else: # Runner
                    runner_score += rewards[idx]
            
            roles = roles_.copy()

            total_steps += 1
            episode_step += 1
        
        scores_history.append(scores)

        # average agent scores for last 100 episodes
        avg_scores = np.mean(scores_history, axis=0)
        avg_last_100_scores = np.mean(scores_history[-100:], axis=0)

        for idx, agent in enumerate(agents):
            if avg_last_100_scores[idx] > avg_scores[idx]:
                agent.save_models()

        bulldog_score_history.append(bulldog_score)
        runner_score_history.append(runner_score)
        episodes.append(episode)

        if episode % PRINT_INTERVAL == 0:
            bulldog_avg_score = np.mean(bulldog_score_history[-100:])
            runner_avg_score = np.mean(runner_score_history[-100:])
            print(f'Episode {episode}, last 100 avg, bd score {bulldog_avg_score:.1f}, r score {runner_avg_score:.1f}')

    if bulldog_algo == MADDPG:
        np.save('results/MADDPG/'+model+'/bulldogs.npy', np.array(bulldog_score_history))
    if runner_algo == MADDPG:
        np.save('results/MADDPG/'+model+'/runners.npy', np.array(runner_score_history))
    np.save('results/MADDPG/'+model+'/eps.npy', np.array(episodes))

    env.close()


if __name__ == '__main__':
    run()

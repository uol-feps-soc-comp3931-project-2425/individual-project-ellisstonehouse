import numpy as np
from maddpg.maddpg import MADDPG
from maddpg.buffer import MultiAgentReplayBuffer
from env.env import BritishBulldogEnv
import os


BULLDOG = 1
RUNNER = 0
MADDPG_ = 0
RANDOM_ = 1

EPISODES = 10_000
MAX_STEPS = 500
PRINT_INTERVAL = 100

GAMMA = 0.95
ALPHA = 1e-4
BETA = 1e-3
TAU = 0.01

bulldog_algo = MADDPG_
runner_algo = MADDPG_

model = 'model_4'
os.makedirs('results/MADDPG/'+model, exist_ok=True)

print(model)

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

def run():

    env = BritishBulldogEnv(init_bulldogs=1, init_runners=2, GUI=False)

    total_steps = 0
    scores_history = []
    best_agent_scores = np.zeros(env.n_agents)
    bulldog_score_history = []
    runner_score_history = []
    episodes = []


    actor_dims = []
    n_actions = []
    for i in range(env.n_agents):
        actor_dims.append(env.observation_space[i])
        n_actions.append(env.action_space[i])
    critic_dims = sum(actor_dims) + sum(n_actions)


    maddpg_agents = MADDPG(actor_dims, critic_dims, env.n_agents, n_actions,
                           gamma=GAMMA, alpha=ALPHA, beta=BETA, tau=TAU,
                           model=model)

    critic_dims = sum(actor_dims)
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
            actions = maddpg_agents.choose_action(observation, evaluate=False)
            
            # for training against random roles
            for idx in range(len(actions)):
                if roles[idx] == BULLDOG and bulldog_algo == RANDOM_:
                    actions[idx] = np.random.uniform(-1.0, 1.0, size=2)
                elif roles[idx] == RUNNER and runner_algo == RANDOM_:
                    actions[idx] = np.random.uniform(-1.0, 1.0, size=2)

            roles, observation_, rewards, done = env.step(actions)

            state = obs_list_to_state_vector(observation)
            state_ = obs_list_to_state_vector(observation_)

            # store experience
            memory.store_transition(observation, state, actions, rewards,
                        observation_, state_, done)

            # end episode if max steps reached
            if episode_step >= MAX_STEPS:
                done = [True]*env.n_agents

            # every 100 steps learn
            if total_steps % 100 == 0:
                maddpg_agents.learn(memory)

            observation = observation_

            scores += rewards
            
            for idx, role in enumerate(roles):
                if role == BULLDOG:
                    bulldog_score += rewards[idx]
                else: # Runner
                    runner_score += rewards[idx]
            
            total_steps += 1
            episode_step += 1
        
        scores_history.append(scores)

        # average agent scores for last 100 episodes
        avg_agent_scores = np.mean(scores_history[-100:], axis=0)

        for idx, agent in enumerate(maddpg_agents.agents):
            if avg_agent_scores[idx] > best_agent_scores[idx]:
                agent.save_models()
                best_agent_scores[idx] = avg_agent_scores[idx]

        bulldog_score_history.append(bulldog_score)
        runner_score_history.append(runner_score)
        episodes.append(episode)

        if episode % PRINT_INTERVAL == 0:
            bulldog_avg_score = np.mean(bulldog_score_history[-100:])
            runner_avg_score = np.mean(runner_score_history[-100:])
            print(f'Episode {episode}, last 100 avg, bd score {bulldog_avg_score:.1f}, r score {runner_avg_score:.1f}')

    if bulldog_algo == MADDPG_:
        np.save('results/MADDPG/'+model+'/bulldogs.npy', np.array(bulldog_score_history))
    if runner_algo == MADDPG_:
        np.save('results/MADDPG/'+model+'/runners.npy', np.array(runner_score_history))
    np.save('results/MADDPG/'+model+'/eps.npy', np.array(episodes))

    env.close()


if __name__ == '__main__':
    run()

import numpy as np
from ddpg.agent import Agent
from env.env import BritishBulldogEnv
import os


# BULLDOG = 1
# RUNNER = 0
# DDPG = 0
# RANDOM = 1

# EPISODES = 10_000
# MAX_STEPS = 500
# PRINT_INTERVAL = 100

# ALPHA = 1e-3
# BETA = 1e-3
# GAMMA = 0.99
# TAU = 0.01

# bulldog_algo = DDPG
# runner_algo = DDPG

# model = 'model_1'
# os.makedirs('results/DDPG/'+model, exist_ok=True)

def run():

    env = BritishBulldogEnv(init_bulldogs=1, init_runners=2, GUI=False)

    total_steps = 0
    scores_history = []
    best_agent_scores = np.zeros(env.n_agents)
    bulldog_score_history = []
    runner_score_history = []
    episodes = []


    agents = []

    for agent_idx in range(env.n_agents):
        input_dims = env.observation_space[0]
        n_actions = env.action_space[0]

        agents.append(Agent(agent_idx, alpha=ALPHA, beta=BETA,
                      input_dims=input_dims, tau=TAU, gamma=GAMMA,
                      batch_size=1024, fc1_dims=64, fc2_dims=64,
                      n_actions=n_actions, model=model))
        

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

            # randomise role actions for optimal policy
            for idx in range(len(actions)):
                if roles[idx] == BULLDOG and bulldog_algo == RANDOM:
                    actions[idx] = np.random.uniform(-1.0, 1.0, size=2)
                elif roles[idx] == RUNNER and runner_algo == RANDOM:
                    actions[idx] = np.random.uniform(-1.0, 1.0, size=2)
            
            roles_, observation_, rewards, done = env.step(actions)

            # store experience
            for idx, agent in enumerate(agents):
                agent.remember(observation[idx], actions[idx],
                               rewards[idx], observation_[idx], done[idx])
            
            # end episode if max steps reached
            if episode_step >= MAX_STEPS:
                done = [True]*env.n_agents
            
            # every 100 steps learn
            if total_steps % 100 == 0:
                for agent in agents:
                    agent.learn()

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
        # avg_scores = np.mean(scores_history, axis=0)
        # avg_last_100_scores = np.mean(scores_history[-100:], axis=0)

        # for idx, agent in enumerate(agents):
        #     if avg_last_100_scores[idx] > avg_scores[idx]:
        #         agent.save_models()

        bulldog_score_history.append(bulldog_score)
        runner_score_history.append(runner_score)
        episodes.append(episode)

        if episode % PRINT_INTERVAL == 0:
            bulldog_avg_score = np.mean(bulldog_score_history[-100:])
            runner_avg_score = np.mean(runner_score_history[-100:])
            print(f'Episode {episode}, last 100 avg, bd score {bulldog_avg_score:.1f}, r score {runner_avg_score:.1f}')
        
    if bulldog_algo == DDPG:
        np.save(file+'/bulldogs.npy', np.array(bulldog_score_history))
    if runner_algo == DDPG:
        np.save(file+'/runners.npy', np.array(runner_score_history))
    np.save(file+'/eps.npy', np.array(episodes))

    env.close()


if __name__ == '__main__':

    BULLDOG = 1
    RUNNER = 0
    DDPG = 0
    RANDOM = 1

    EPISODES = 5000
    MAX_STEPS = 500
    PRINT_INTERVAL = 100

    ALPHA = 1e-4
    BETA = 1e-4
    GAMMA = 0.95
    TAU = 0.01

    for bulldog_algo in [DDPG, RANDOM]:
        runner_algo = not(bulldog_algo)

        model = 'BD_'*(not bulldog_algo)+'R_'*(not runner_algo)+'a='+str(ALPHA)+'_b='+str(BETA)+'_g='+str(GAMMA)+'_t='+str(TAU)

        for folder in ['results1/', 'results2/', 'results3/']:

            file = folder + 'DDPG/' + model

            os.makedirs(file, exist_ok=True)
            print(model)

            run()

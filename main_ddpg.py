import numpy as np
from ddpg.agent import Agent
from env.env import BritishBulldogEnv


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


if __name__ == '__main__':

    env = BritishBulldogEnv(num_bulldogs=1, num_runner=2, GUI=True)
    env.create_arena()
    
    scenario = 'simple_speaker_listener'

    n_agents = env.num_bulldogs + env.num_runner

    agents = []

    for i in range(n_agents):
        input_dims = env.observation_space[0]
        n_actions = env.action_space[0]

        agents.append(Agent(alpha=1e-3, beta=1e-3,
                      input_dims=input_dims, tau=0.01, gamma=0.95,
                      batch_size=1024, fc1_dims=64, fc2_dims=64,
                      n_actions=n_actions))

    N_GAMES = 25_000
    PRINT_INTERVAL = 1
    total_steps = 0
    score_history = []
    evaluate = True
    best_score = 0

    if evaluate:
        for agent in agents:
            agent.load_models()

    total_steps = 0

    for i in range(N_GAMES):
        observation = env.reset()
        done = [False] * n_agents
        score = 0
        # observation = list(observation.values())


        while not any(done):
            action = [agent.choose_action(observation[idx])
                      for idx, agent in enumerate(agents)]
            
            observation_, reward, done = env.step(action)

            for idx, agent in enumerate(agents):
                agent.remember(observation[idx], action[idx],
                               reward[idx], observation_[idx], done[idx])
                
            if total_steps % 100 == 0 and not evaluate:
                for agent in agents:
                    agent.learn()
            score += sum(reward)
            observation = observation_
            total_steps += 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        if i % PRINT_INTERVAL == 0 and i > 0:
            print(f'episode {i} avg score {avg_score:.1f}')

import numpy as np
from maddpg.maddpg import MADDPG
from maddpg.buffer import MultiAgentReplayBuffer
from env.env import BritishBulldogEnv

import time


def observation_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


def run():
    
    evaluate = True

    scenario = 'test1'


    env = BritishBulldogEnv(num_bulldogs=1, num_runner=2, GUI=True)
    env.create_arena()

    n_agents = env.num_bulldogs + env.num_runner

    print(env.observation_space)


    actor_dims = []
    n_actions = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i])
        n_actions.append(env.action_space[i])

    critic_dims = sum(actor_dims) + sum(n_actions)


    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           gamma=0.95, alpha=1e-4, beta=1e-3,
                           scenario=scenario)
    critic_dims = sum(actor_dims)
    memory = MultiAgentReplayBuffer(1_000_000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=1024)


    PRINT_INTERVAL = 100
    N_GAMES = 10000
    MAX_STEPS = 500
    total_steps = 0
    score_history = []
    best_score = 0

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        observation = env.reset()
        score = [0, 0, 0]
        done = [False]*n_agents
        episode_step = 0

        while not any(done):
            if eval:
                time.sleep(1 / 2400)

            actions = maddpg_agents.choose_action(observation, evaluate=evaluate)

            actions = list(actions.values())

            # WE MUST MAKE IT SO THAT THE AGENT KNOWS THAT IT IS DONE SINCE IT IS STORED IN THE EXPERIENCES
            # SO IF RUNNER REACHES HOEM BASE IT IS DONE, EPISODE COMMENCE WHEN ALL DONE
            observation_, reward, done = env.step(actions)





            state = observation_list_to_state_vector(observation)
            state_ = observation_list_to_state_vector(observation_)

            if episode_step >= MAX_STEPS:
                done = [True]*n_agents
            
            memory.store_transition(observation, state, actions, reward,
                                    observation_, state_, done)

            # every 100 steps learn
            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            observation = observation_
            
            # newly added
            for agent_id in range(len(score)):
                score[agent_id] += reward[agent_id]

            total_steps += 1
            episode_step += 1

        score_history.append(score)

        avg_score_agent = np.mean(score_history[-100:], axis=0)
        avg_score = np.mean(avg_score_agent)

        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            # print("Bulldog:", avg_score_bulldog, "Runner:", avg_score_runner)
            print("agent 1:", avg_score_agent[0], "agent 2:", avg_score_agent[1], "agent 3:", avg_score_agent[2])
            print('episode', i, 'average score {:.1f}'.format(avg_score))



if __name__ == '__main__':
    run()
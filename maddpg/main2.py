import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from env import PredPreyEnv

import time


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


def run():
    
    eval = True

    scenario = 'test1'


    env = PredPreyEnv(num_predators=1, num_prey=2, GUI=True)
    env.create_arena()

    n_agents = env.num_predators + env.num_prey


    actor_dims = []
    n_actions = []
    for i in range(env.num_predators + env.num_prey):
        actor_dims.append(5*n_agents)
        n_actions.append(2)

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

    if eval:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        done = [False]*n_agents
        episode_step = 0

        while not any(done):
            if eval:
                time.sleep(1 / 2400)

            actions = maddpg_agents.choose_action(obs, evaluate=eval)

            obs_, reward, done = env.step(actions)


            list_actions = list(actions.values())

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                done = [True]*n_agents
            
            memory.store_transition(obs, state, list_actions, reward,
                                    obs_, state_, done)

            if total_steps % 100 == 0 and not eval:
                maddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward)

            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if not eval:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            # print("Pred:", avg_score_pred, "Prey:", avg_score_prey)
            # print("agent 1:", avg_score_pred, "Prey:", avg_score_prey)
            print('episode', i, 'average score {:.1f}'.format(avg_score))





def evaluate(maddpg_agents, env, ep, step, n_eval=3):
    n_agents = env.num_predators + env.num_prey
    score_history = []

    for i in range(n_eval):
        obs = env.reset()
        score = 0
        done = [False] * n_agents


        while not any(done):
            actions = maddpg_agents.choose_action(obs, evaluate=True)
            obs_, reward, done  = env.step(actions)

            obs = obs_
            score += sum(reward)

        score_history.append(score)
    avg_score = np.mean(score_history)
    print(f'Evaluation episode {ep} train steps {step}'
          f' average score {avg_score:.1f}')
    return avg_score


if __name__ == '__main__':
    run()
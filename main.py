import numpy as np
from maddpg.maddpg import MADDPG
from maddpg.buffer import MultiAgentReplayBuffer
from env.env import BritishBulldogEnv


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


def run():
    
    eval = False
    scenario = 'test2'

    env = BritishBulldogEnv(num_predators=1, num_prey=2, GUI=False)
    env.create_arena()

    n_agents = env.num_predators + env.num_prey


    actor_dims = []
    n_actions = []
    for i in range(env.num_predators):
        actor_dims.append(5 + 5*env.num_prey)
        n_actions.append(2)
    for i in range(env.num_prey):
        actor_dims.append(5 + 5*env.num_predators)
        n_actions.append(2)

    critic_dims = sum(actor_dims) + sum(n_actions)


    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           gamma=0.95, alpha=1e-4, beta=1e-3,
                           scenario=scenario)
    critic_dims = sum(actor_dims)
    memory = MultiAgentReplayBuffer(1_000_000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=1024)


    EVAL_INTERVAL = 250
    MAX_STEPS = 10_000

    total_steps = 0
    episode = 0
    eval_scores = []
    eval_steps = []

    score = evaluate(maddpg_agents, env, episode, total_steps)
    eval_scores.append(score)
    eval_steps.append(total_steps)


    while total_steps < MAX_STEPS:

        obs = env.reset()
        done = [False] * n_agents

        while not any(done):
            actions = maddpg_agents.choose_action(obs, evaluate=False)

            obs_, reward, done = env.step(actions)

            list_actions = list(actions.values())

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)
            
            memory.store_transition(obs, state, list_actions, reward,
                                    obs_, state_, done)

            if total_steps % 100 == 0:
                maddpg_agents.learn(memory)
            obs = obs_
            total_steps += 1

        if total_steps % EVAL_INTERVAL == 0:
            score = evaluate(maddpg_agents, env, episode, total_steps)
            eval_scores.append(score)
            eval_steps.append(total_steps)

        episode += 1

    np.save('data/maddpg_scores.npy', np.array(eval_scores))
    np.save('data/maddpg_steps.npy', np.array(eval_steps))




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
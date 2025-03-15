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

if __name__ == '__main__':

    evaluate = True

    scenario = 'test1'
    
    env = PredPreyEnv(num_predators=1, num_prey=1, GUI=True)
    env.create_arena()

    n_agents = env.num_predators + env.num_prey



    actor_dims = []
    for i in range(env.num_predators):
        actor_dims.append(2 + 2*env.num_prey)
    for i in range(env.num_prey):
        actor_dims.append(2 + 2*env.num_predators)

    critic_dims = sum(actor_dims)

    n_actions = 2

    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)

    PRINT_INTERVAL = 100
    N_GAMES = 10000
    MAX_STEPS = 250
    total_steps = 0
    score_history_pred = []
    score_history_prey = []
    best_score_pred = 0
    best_score_prey = 0

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        obs = env.reset()
        pred_score = 0
        prey_score = 0
        done = [False]*n_agents
        episode_step = 0

        while not any(done):
            if evaluate:
                time.sleep(1 / 2400)
            
            actions = maddpg_agents.choose_action(obs)

            print(actions)


            obs_, reward, done = env.step(actions)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                done = [True]*n_agents

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_

            pred_score += sum(reward[:env.num_predators])
            prey_score += sum(reward[env.num_prey:])

            total_steps += 1
            episode_step += 1
        
        # print("Pred:", pred_score, "Prey:", prey_score)

        score_history_pred.append(pred_score)
        score_history_prey.append(prey_score)
        avg_score_pred = np.mean(score_history_pred[-100:])
        avg_score_prey = np.mean(score_history_prey[-100:])

        if not evaluate:
            if avg_score_pred > best_score_pred:
                maddpg_agents.save_checkpoint()
                best_score_pred = avg_score_pred
            if avg_score_prey > best_score_prey:
                maddpg_agents.save_checkpoint()
                best_score_prey = avg_score_prey
        if i % PRINT_INTERVAL == 0 and i > 0:
            print("Pred:", avg_score_pred, "Prey:", avg_score_prey)
            print('episode', i, 'average score {:.1f}'.format((avg_score_pred + avg_score_prey)/n_agents))

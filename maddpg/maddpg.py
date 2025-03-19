from agent import Agent


class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 alpha=1e-4, beta=1e-3, fc1=64, fc2=64, gamma=0.95, tau=0.01,
                 chkpt_dir='tmp/maddpg/', scenario='sample'):
        self.agents = []
        chkpt_dir += scenario
        for agent_idx in range(n_agents):

            min_action = -1.0
            max_action = 1.0

            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                               n_actions[agent_idx], n_agents, agent_idx,
                               alpha=alpha, beta=beta, tau=tau, fc1=fc1,
                               fc2=fc2, chkpt_dir=chkpt_dir,
                               gamma=gamma, min_action=min_action,
                               max_action=max_action))

    def save_checkpoint(self):
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs, evaluate=False):
        actions = {}
        for agent_id, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_id], evaluate)
            actions[agent_id] = action
        return actions

    def learn(self, memory):
        for agent in self.agents:
            agent.learn(memory, self.agents)

import numpy as np
import matplotlib.pyplot as plt


def plot_eval_curve(x, scores, filename, lines=None):
    maddpg_scores, ddpg_scores = scores

    fig, ax = plt.subplots()

    # Compute running averages
    N = len(maddpg_scores)
    maddpg_avg = np.empty(N)
    for t in range(N):
        maddpg_avg[t] = np.mean(maddpg_scores[max(0, t-100):(t+1)])

    N = len(ddpg_scores)
    ddpg_avg = np.empty(N)
    for t in range(N):
        ddpg_avg[t] = np.mean(ddpg_scores[max(0, t-100):(t+1)])

    # Plot both curves on the same axis
    ax.plot(x, maddpg_avg, label='MADDPG', color='C0')
    ax.plot(x, ddpg_avg, label='DDPG', color='C1')

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Average Score")
    ax.legend()

    if lines is not None:
        for line in lines:
            plt.axvline(x=line, color='grey', linestyle='--')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    # plt.savefig(filename)


def plot_training_curve(algo, model, lines=None):


    bulldog_scores = np.load('results/'+algo+'/'+model+'_bulldogs.npy')
    runner_scores = np.load('results/'+algo+'/'+model+'_runners.npy')
    episodes = np.load('results/'+algo+'/'+model+'_eps.npy')
    
    fig, ax = plt.subplots()

    # Plot both curves on the same axis
    ax.plot(episodes, bulldog_scores, label='MADDPG', color='C0')
    ax.plot(episodes, runner_scores, label='DDPG', color='C1')

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Average Score")
    ax.legend()

    plt.tight_layout()
    plt.savefig('plots/'+algo+'_'+model+'.png')
    print('Plotted: '+algo+'_'+model)
    plt.close()




if __name__ == '__main__':

    for algo in ['MADDPG']:
        for model in ['model_1']:

            # bulldog_scores = np.load('results/'+algo+'/'+model+'_bulldogs.npy')
            # runner_scores = np.load('results/'+algo+'/'+model+'_runners.npy')
            # episodes = np.load('results/'+algo+'/'+model+'_eps.npy')

            plot_training_curve(algo, model)


            # plot_eval_curve(scores=(bulldog_scores, runner_scores), 
            #                 episodes=episodes,
            #                 filename='plots/'+algo+'.png')
    
    # model = 'test3'
    # algo = 'MADDPG'

    # plot_training_curve(algo, model)


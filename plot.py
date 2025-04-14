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
    plt.savefig('plots/maddpg_bd_vs_ddpg_r.png')
    plt.close()




if __name__ == '__main__':

    # bulldog_scores = np.load('results/evaluate/test1.npy')
    # runner_scores = np.load('results/evaluate/simple_speaker_listener.npy')

    # steps = np.load('results/evaluate/eval_steps.npy')

    # plot_eval_curve(x=steps,
    #                     scores=(bulldog_scores, runner_scores),
    #                     filename='plots/maddpg_bd_vs_ddpg_r.png')
    
    model = 'test3'
    algo = 'MADDPG'

    plot_training_curve(algo, model)


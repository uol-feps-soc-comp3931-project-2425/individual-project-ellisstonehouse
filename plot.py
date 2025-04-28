import numpy as np
import matplotlib.pyplot as plt
import os


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


    bulldog_scores = np.load('results/'+algo+'/'+model+'/bulldogs.npy')
    runner_scores = np.load('results/'+algo+'/'+model+'/runners.npy')
    episodes = np.load('results/'+algo+'/'+model+'/eps.npy')
    
    _, ax = plt.subplots()

    # Plot both curves on the same axis
    ax.plot(episodes, bulldog_scores, label='Bulldog', color='C0', alpha=0.7)
    ax.plot(episodes, runner_scores, label='Runner', color='C1', alpha=0.7)

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(f"{algo} - {model} Cumulative Reward per Episode")
    ax.legend()

    plt.tight_layout()
    os.makedirs('plots/'+algo, exist_ok=True)

    plt.savefig('plots/'+algo+'/'+model+'.png')
    print('Plotted: '+algo+'/'+model)
    plt.close()


def plot_training_curve_all(algo, lines=None):

    bulldog_scores_1 = np.load('results/'+algo+'/model_1/bulldogs.npy')
    bulldog_scores_2 = np.load('results/'+algo+'/model_2/bulldogs.npy')
    bulldog_scores_3 = np.load('results/'+algo+'/model_3/bulldogs.npy')
    bulldog_scores_4 = np.load('results/'+algo+'/model_4/bulldogs.npy')
    bulldog_scores_5 = np.load('results/'+algo+'/model_5/bulldogs.npy')
    runner_scores_1 = np.load('results/'+algo+'/model_1/runners.npy')
    runner_scores_2 = np.load('results/'+algo+'/model_2/runners.npy')
    runner_scores_3 = np.load('results/'+algo+'/model_3/runners.npy')
    runner_scores_4 = np.load('results/'+algo+'/model_4/runners.npy')
    runner_scores_5 = np.load('results/'+algo+'/model_5/runners.npy')

    episodes = np.load('results/'+algo+'/model_1/eps.npy')

    bulldog_scores = np.mean([bulldog_scores_1, bulldog_scores_2, bulldog_scores_3, bulldog_scores_4, bulldog_scores_5], axis=0)
    runner_scores = np.mean([runner_scores_1, runner_scores_2, runner_scores_3, runner_scores_4, runner_scores_5], axis=0)

    _, ax = plt.subplots()

    # Plot both curves on the same axis
    ax.plot(episodes, bulldog_scores, label='Bulldog', color='C0', alpha=0.7)
    ax.plot(episodes, runner_scores, label='Runner', color='C1', alpha=0.7)

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(f"{algo} - Average Cumulative Reward per Episode")
    ax.legend()

    plt.tight_layout()
    os.makedirs('plots/'+algo, exist_ok=True)

    plt.savefig('plots/'+algo+'/all.png')
    print('Plotted: '+algo+'/all')
    plt.close()


if __name__ == '__main__':

    for algo in ['DDPG', 'MADDPG']:
        for model in ['model_1', 'model_2', 'model_3', 'model_4', 'model_5']:
            plot_training_curve(algo, model)


            # plot_eval_curve(scores=(bulldog_scores, runner_scores), 
            #                 episodes=episodes,
            #                 filename='plots/'+algo+'.png')
        plot_training_curve_all(algo)
    


import numpy as np
import matplotlib.pyplot as plt


def plot_opti_param(bulldog_algo, algo, model, lines=None):

    if bulldog_algo == MADDPG_:
        scores = np.load('results/'+algo+'/'+model+'/bulldogs.npy')
    else:
        scores = np.load('results/'+algo+'/'+model+'/runners.npy')
        
    episodes = np.load('results/'+algo+'/'+model+'/eps.npy')
    
    fig, ax = plt.subplots()

    # Plot both curves on the same axis
    ax.plot(episodes, scores, label=algo, color='C0')

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Average Score")
    ax.legend()

    plt.tight_layout()
    plt.savefig('plots/'+algo+'/'+model+'.png')
    print('Plotted: '+algo+'/'+model)
    plt.close()


if __name__ == '__main__':

    MADDPG_ = 0
    RANDOM_ = 1

    ALPHA = 1e-3
    BETA = 1e-3
    GAMMA = 0.95
    TAU = 0.01


    # plot_training_curve(algo, model)
    for algo in ['MADDPG']:
        for bulldog_algo in [MADDPG_, RANDOM_]:
            runner_algo = not(bulldog_algo)

            model = 'BD_'*(not bulldog_algo)+'R_'*(not runner_algo)+'a='+str(ALPHA)+'_b='+str(BETA)+'_g='+str(GAMMA)+'_t='+str(TAU)

            plot_opti_param(bulldog_algo, algo, model)

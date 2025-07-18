import numpy as np
import matplotlib.pyplot as plt
import os


def plot_opti_param(bulldog_algo, algo, model, lines=None):

    if bulldog_algo == ALGO:
        scores1 = np.load('opti_param_results/results1/'+algo+'/'+model+'/bulldogs.npy')
        scores2 = np.load('opti_param_results/results2/'+algo+'/'+model+'/bulldogs.npy')
        scores3 = np.load('opti_param_results/results3/'+algo+'/'+model+'/bulldogs.npy')
    else:
        scores1 = np.load('opti_param_results/results1/'+algo+'/'+model+'/runners.npy')
        scores2 = np.load('opti_param_results/results2/'+algo+'/'+model+'/runners.npy')
        scores3 = np.load('opti_param_results/results3/'+algo+'/'+model+'/runners.npy')
        
    episodes = np.load('opti_param_results/results1/'+algo+'/'+model+'/eps.npy')

    scores = np.mean([scores1, scores2, scores3], axis=0)
    
    fig, ax = plt.subplots()

    # Plot both curves on the same axis
    ax.plot(episodes, scores, label=algo, color='C0')

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Average Score")
    ax.legend()

    plt.tight_layout()

    # os.makedirs('opti_param_plots/'+algo+'/'+model, exist_ok=True)

    
    plt.savefig('opti_param_plots/'+algo+'/'+model+'.png')
    print('Plotted: '+algo+'/'+model)


    
    
    plt.close()


if __name__ == '__main__':

    ALGO = 0
    RANDOM = 1

    TAU = 0.01

    for algorithm in ['MADDPG']:
        for bulldog_algo in [ALGO, RANDOM]:
            runner_algo = not(bulldog_algo)
            for GAMMA in [0.95, 0.99]:
                for BETA in [1e-3, 1e-4]:
                    for ALPHA in [1e-3, 1e-4]:
                        model = 'BD_'*(not bulldog_algo)+'R_'*(not runner_algo)+'a='+str(ALPHA)+'_b='+str(BETA)+'_g='+str(GAMMA)+'_t='+str(TAU)

                        plot_opti_param(bulldog_algo, algorithm, model)



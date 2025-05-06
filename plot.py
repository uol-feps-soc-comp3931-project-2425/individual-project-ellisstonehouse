import numpy as np
import matplotlib.pyplot as plt
import os


def plot_eval_curve(bulldog_algo, runner_algo):


    bulldog_scores_list = []
    runner_scores_list = []
    
    episodes = None


    if bulldog_algo == 'RANDOM':
        for i in range(1, 6):
            bulldog_scores_list.append(np.load(f'results/evaluate/RANDOM_vs_{runner_algo}_model_{i}/bulldogs.npy'))
            runner_scores_list.append(np.load(f'results/evaluate/RANDOM_vs_{runner_algo}_model_{i}/runners.npy'))
        episodes = np.load(f'results/evaluate/RANDOM_vs_{runner_algo}_model_1/episodes.npy')
    elif runner_algo == 'RANDOM':
        for i in range(1, 6):
            bulldog_scores_list.append(np.load(f'results/evaluate/{bulldog_algo}_model_{i}_vs_RANDOM/bulldogs.npy'))
            runner_scores_list.append(np.load(f'results/evaluate/{bulldog_algo}_model_{i}_vs_RANDOM/runners.npy'))
        episodes = np.load(f'results/evaluate/{bulldog_algo}_model_1_vs_RANDOM/episodes.npy')
    else:
        for i in range(1, 6):
            for j in range(1, 6):
                bulldog_scores_list.append(np.load(f'results/evaluate/{bulldog_algo}_model_{i}_vs_{runner_algo}_model_{j}/bulldogs.npy'))
                runner_scores_list.append(np.load(f'results/evaluate/{bulldog_algo}_model_{i}_vs_{runner_algo}_model_{j}/runners.npy'))
        episodes = np.load(f'results/evaluate/{bulldog_algo}_model_1_vs_{runner_algo}_model_1/episodes.npy')

        
    bulldog_scores = np.mean(bulldog_scores_list, axis=0)
    runner_scores = np.mean(runner_scores_list, axis=0)



    _, ax = plt.subplots()

    # Plot both curves on the same axis
    ax.plot(episodes, bulldog_scores, label='Bulldog', color='C0', alpha=0.7)
    ax.plot(episodes, runner_scores, label='Runner', color='C1', alpha=0.7)

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(f"{bulldog_algo} vs. {runner_algo} - Average Cumulative Reward per Episode")
    ax.legend()

    plt.tight_layout()
    os.makedirs('plots/evaluate', exist_ok=True)

    plt.savefig(f'plots/evaluate/{bulldog_algo} vs. {runner_algo}.png')
    print(f'plots/evaluate/{bulldog_algo} vs. {runner_algo}')
    plt.close()




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
    ax.set_title(f"{algo} - {model} Cumulative Reward per Episode, during Training")
    ax.legend()

    plt.tight_layout()
    os.makedirs('plots/'+algo, exist_ok=True)

    plt.savefig('plots/'+algo+'/'+model+'.png')
    print('Plotted: '+algo+'/'+model)
    plt.close()


def plot_training_curve_all(algo, lines=None):

    bulldog_scores_list = []
    runner_scores_list = []

    for i in range(1, 6):  # Loop over models 1 to 5
        bulldog_scores_list.append(np.load(f'results/{algo}/model_{i}/bulldogs.npy'))
        runner_scores_list.append(np.load(f'results/{algo}/model_{i}/runners.npy'))

    bulldog_scores = np.mean(bulldog_scores_list, axis=0)
    runner_scores = np.mean(runner_scores_list, axis=0)
    
    episodes = np.load('results/'+algo+'/model_1/eps.npy')

    _, ax = plt.subplots()

    # Plot both curves on the same axis
    ax.plot(episodes, bulldog_scores, label='Bulldog', color='C0', alpha=0.7)
    ax.plot(episodes, runner_scores, label='Runner', color='C1', alpha=0.7)

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(f"{algo} - Average Cumulative Reward per Episode, during Training")
    ax.legend()

    plt.tight_layout()
    os.makedirs('plots/'+algo, exist_ok=True)

    plt.savefig('plots/'+algo+'/all.png')
    print('Plotted: '+algo+'/all')
    plt.close()


def plot_eval_wins():

    bd_algo = []
    r_algo = []
    bulldog_wins = []


    for bulldog_algo in ['DDPG', 'MADDPG']:
        for runner_algo in ['DDPG', 'MADDPG']:
            if bulldog_algo == 'RANDOM' and runner_algo == 'RANDOM':
                continue

            bulldog_scores_list = []

            if bulldog_algo == 'RANDOM':
                for i in range(1, 6):
                    bulldog_scores_list.append(np.load(f'results/evaluate/RANDOM_vs_{runner_algo}_model_{i}/bulldogs.npy'))
            elif runner_algo == 'RANDOM':
                for i in range(1, 6):
                    bulldog_scores_list.append(np.load(f'results/evaluate/{bulldog_algo}_model_{i}_vs_RANDOM/bulldogs.npy'))
            else:
                for i in range(1, 6):
                    for j in range(1, 6):
                        bulldog_scores_list.append(np.load(f'results/evaluate/{bulldog_algo}_model_{i}_vs_{runner_algo}_model_{j}/bulldogs.npy'))


            bulldog_scores = np.mean(bulldog_scores_list, axis=0)
            
            bd_algo.append(bulldog_algo)
            r_algo.append(runner_algo)
            bulldog_wins.append(np.sum(bulldog_scores > 0))

    for bulldog_algo in ['DDPG', 'MADDPG']:
        runner_algo = 'RANDOM'

        bulldog_scores_list = []

        if bulldog_algo == 'RANDOM':
            for i in range(1, 6):
                bulldog_scores_list.append(np.load(f'results/evaluate/RANDOM_vs_{runner_algo}_model_{i}/bulldogs.npy'))
        elif runner_algo == 'RANDOM':
            for i in range(1, 6):
                bulldog_scores_list.append(np.load(f'results/evaluate/{bulldog_algo}_model_{i}_vs_RANDOM/bulldogs.npy'))
        else:
            for i in range(1, 6):
                for j in range(1, 6):
                    bulldog_scores_list.append(np.load(f'results/evaluate/{bulldog_algo}_model_{i}_vs_{runner_algo}_model_{j}/bulldogs.npy'))


        bulldog_scores = np.mean(bulldog_scores_list, axis=0)

        bd_algo.append(bulldog_algo)
        r_algo.append(runner_algo)
        bulldog_wins.append(np.sum(bulldog_scores > 0))

    
    for runner_algo in ['DDPG', 'MADDPG']:
        bulldog_algo = 'RANDOM'

        bulldog_scores_list = []

        if bulldog_algo == 'RANDOM':
            for i in range(1, 6):
                bulldog_scores_list.append(np.load(f'results/evaluate/RANDOM_vs_{runner_algo}_model_{i}/bulldogs.npy'))
        elif runner_algo == 'RANDOM':
            for i in range(1, 6):
                bulldog_scores_list.append(np.load(f'results/evaluate/{bulldog_algo}_model_{i}_vs_RANDOM/bulldogs.npy'))
        else:
            for i in range(1, 6):
                for j in range(1, 6):
                    bulldog_scores_list.append(np.load(f'results/evaluate/{bulldog_algo}_model_{i}_vs_{runner_algo}_model_{j}/bulldogs.npy'))


        bulldog_scores = np.mean(bulldog_scores_list, axis=0)

        bd_algo.append(bulldog_algo)
        r_algo.append(runner_algo)
        bulldog_wins.append(np.sum(bulldog_scores > 0))


    total_games = 1000

    bulldog_wins = bulldog_wins[::-1]
    runner_wins = [total_games - win for win in bulldog_wins]
   
    bd_algo = bd_algo[::-1]
    r_algo = r_algo[::-1]

    y = np.arange(len(bd_algo))

    _, ax = plt.subplots(figsize=(10, 6))

    ax.barh(y, bulldog_wins, label='Bulldog Wins', alpha=0.7)
    ax.barh(y, runner_wins, left=bulldog_wins, label='Runner Wins', alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels([''] * len(bd_algo))

    for i in range(len(bd_algo)):
        
        ax.text(5, y[i], bd_algo[i], va='center', ha='left', color='black', fontsize=9, weight='bold')
        
        ax.text(bulldog_wins[i] + runner_wins[i] - 5, y[i], r_algo[i], va='center', ha='right', color='black', fontsize=9, weight='bold')

    ax.set_xlabel('Number of Wins')
    ax.set_title('Win Distribution per Match up')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)


    plt.tight_layout()
    # plt.show()

    plt.savefig('plots/bar.png')
    print('Plotted: bar.png')
    plt.close()


    



if __name__ == '__main__':

    for algo in ['DDPG', 'MADDPG']:
        for model in ['model_1', 'model_2', 'model_3', 'model_4', 'model_5']:
            plot_training_curve(algo, model)
        plot_training_curve_all(algo)
    

    for bulldog_algo in ['DDPG', 'MADDPG']:
        for runner_algo in ['DDPG', 'MADDPG']:
                if bulldog_algo == 'RANDOM' and runner_algo == 'RANDOM':
                    continue
                
                plot_eval_curve(bulldog_algo, runner_algo)

    for bulldog_algo in ['DDPG', 'MADDPG']:
        plot_eval_curve(bulldog_algo, 'RANDOM')
    
    for runner_algo in ['DDPG', 'MADDPG']:
        plot_eval_curve('RANDOM', runner_algo)


    plot_eval_wins()



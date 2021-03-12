from modules.game_module import game_module
import numpy as np
import random
import csv
import pygame
import os


def generate_dataset(obstacle_dataset_file,goal_dataset_file,obstacles,trials):

    csv_writer = csv.writer(open(obstacle_dataset_file, 'w'))
    csv_writer_goal = csv.writer(open(goal_dataset_file, 'w'))

    # Generate a set of worlds to test on, and save them to a file
    for trial in enumerate(range(trials)):
        a = game_module('game_data_test_random.out', obstacles)
        num_block = a.world_size[0] * a.world_size[1]
        obs_idx = random.sample(list(range(num_block)), a.num_obs + 1)
        csv_writer.writerow(obs_idx)

        for i in range(a.num_obs):
            row_pos = obs_idx[i] // a.world_size[0]
            col_pos = obs_idx[i] % a.world_size[1]
            a.obs.append((row_pos, col_pos))
            a.obs_mat[row_pos, col_pos] = 1

        a.pos = [obs_idx[-1] // a.world_size[0], obs_idx[-1] % a.world_size[1]]
        a.random_goal_location()
        csv_writer_goal.writerow(a.goal_pos)

def play_game_files(train_file,test_num,obstacle_dataset_file,goal_dataset_file, inspect_envs):

    successes=[0]*test_num
    crashes=[0]*test_num
    stucks=[0]*test_num
    mean_steps_success=[0]*test_num

    for i in range(test_num):
        print('Trial number ', i)
        a = game_module('./data/tests.out',obstacles)
        a.train_from_file(train_file)
        out=a.play_game_from_file(obstacle_dataset_file,goal_dataset_file)
        # success, crash, stuck, step_count, crash_count, stuck_count
        successes[i]=float(out[0]/len(out[3]))
        crashes[i]=float(out[1]/len(out[3]))
        stucks[i]=float(out[2]/len(out[3]))
        mean_steps_success[i]=np.mean(np.array([num for inum,num in enumerate(out[3]) if out[4][inum]<1]))

        if inspect_envs:
            #Environments that threw a timeout (stuck)
            stuck_envs=[]
            for i, ot in enumerate(out[-1]):
                if ot[2]==1:
                    stuck_envs.append(i)

            #Read trainig environments from file
            test_reader = csv.reader(open(obstacle_dataset_file, 'r'))
            goal_reader = csv.reader(open(goal_dataset_file, 'r'))
            test_ids=list(test_reader)
            goal_ids=list(goal_reader)

            #Save prints of the enironments that
            if not os.path.exists('./inspect_figs'):
                os.makedirs('inspect_figs')
            gm = game_module('game_data_test_random.out', obstacles)
            for sen,stuck_env in enumerate(stuck_envs):
                if not os.path.exists('./inspect_figs/obs_env_' + str(obstacles) + '_sn_'+ str(stuck_env) + ".jpg"):
                    gm.game_inspect_env('1',[int(num) for num in test_ids[stuck_env]], [int(num) for num in goal_ids[stuck_env]], filename='./inspect_figs/obs_env_' + str(obstacles) + '_sn_'+ str(stuck_env) + ".jpg")

            # print('Time out ', sum([ot[2] for ot in out[-1]]) )
            # print('Stuck ', sum([ot[3] for ot in out[-1]]) )
            # print('xStuck ', sum([ot[4] for ot in out[-1]]) )
            # print('yStuck ', sum([ot[5] for ot in out[-1]]) )

    return successes,crashes,stucks,mean_steps_success


trials=100

obstacle_list=[5,10,15,20]

#The 'avoid' training is done on worlds with 20 obstacles -- the assumption is that it willbe good at both obstacle avoidance and goal reaching
#The 'goal' training is done on worlds with 5 obstacles -- the assumption is that it may reach the goal fast when it can (but it does crash often)
# types_train=['avoid','goal']
types_train=['avoid']

successes_dict=dict((key, [0]*len(obstacle_list)) for key in types_train)
crashes_dict=dict((key, [0]*len(obstacle_list)) for key in types_train)
stucks_dict=dict((key, [0]*len(obstacle_list)) for key in types_train)
steps_success_dict=dict((key, [0]*len(obstacle_list)) for key in types_train)


for on,obstacles in enumerate(obstacle_list):

    obstacle_dataset_file = './data/obstacles_' + str(obstacles) + '.csv'
    goal_dataset_file = './data/goals_' + str(obstacles) + '.csv'

    generate_datasets=0

    if generate_datasets:

        generate_dataset(obstacle_dataset_file, goal_dataset_file, obstacles, trials)

    else:

        # I was looping through these two "types of training". They are essentially two training files, one was generated
        # by playing on a 5 obstacle environment (goal focus), and the other one of a 20 obstacle environment (obstacle
        # avoidance focus).

        for type_train in types_train:

            print('\n\nObstacles :', obstacles, ', type: ', type_train)

            # train_file='./data/game_data_random_'+type_train+'.out'

            #For these experiments I'm just using the datasets generated for the two state case
            train_file='./data/game_data_2state_based.out'
            test_num=10
            inspect_test_environments=1

            successes,crashes,stucks,mean_steps_success=play_game_files(train_file, test_num, obstacle_dataset_file, goal_dataset_file, inspect_test_environments)
            mean_successes=np.mean(np.array(successes))
            mean_crashes=np.mean(np.array(crashes))
            mean_stucks=np.mean(np.array(stucks))
            mean_steps_success=np.mean(np.array(mean_steps_success))

            successes_dict[type_train][on]=mean_successes
            crashes_dict[type_train][on]=mean_crashes
            stucks_dict[type_train][on]=mean_stucks
            steps_success_dict[type_train][on]=mean_steps_success

            print('Mean successes ', mean_successes)
            print('Mean crashes ', mean_crashes)
            print('Mean stucks ', mean_stucks)
            print('Mean steps success ', mean_steps_success)

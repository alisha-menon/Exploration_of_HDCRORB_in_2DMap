from modules.game_module import game_module
import numpy as np
import random
import csv


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

def play_game_files(train_file,test_num,obstacle_dataset_file,goal_dataset_file):

    successes=[0]*test_num
    crashes=[0]*test_num
    stucks=[0]*test_num
    mean_steps_success=[0]*test_num
    triggered_x_stuck = [0]*test_num
    triggered_y_stuck = [0]*test_num
    stuck_after_stuck = [0]*test_num
    crash_after_stuck = [0]*test_num
    stuck_count = [0]*test_num

    for i in range(test_num):

        a = game_module('./data/tests.out',obstacles)
        a.train_from_file(train_file)
        out=a.play_game_from_file(obstacle_dataset_file,goal_dataset_file)
        # success, crash, stuck, step_count, crash_count, stuck_count
        successes[i]=float(out[0]/len(out[3]))
        crashes[i]=float(out[1]/len(out[3]))
        stucks[i]=float(out[2]/len(out[3]))
        mean_steps_success[i]=np.mean(np.array([num for inum,num in enumerate(out[3]) if out[4][inum]<1]))
        triggered_x_stuck[i] = out[6]
        triggered_y_stuck[i] = out[7]
        stuck_after_stuck[i] = out[8]
        #stuck_after_ystuck = out[9]
        crash_after_stuck[i] = out[9]
        stuck_count[i] = out[10]
        #crash_after_ystuck = out[11]
        #print('triggered_x_stuck: ', triggered_x_stuck)
        #print('triggered_y_stuck: ', triggered_y_stuck)

    return successes,crashes,stucks,mean_steps_success,triggered_x_stuck,triggered_y_stuck, stuck_after_stuck,crash_after_stuck


trials=100

obstacle_list=[15]

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
            test_num=100
            successes,crashes,stucks,mean_steps_success,triggered_x_stuck,triggered_y_stuck,stuck_after_stuck,crash_after_stuck,stuck_count=play_game_files(train_file, test_num, obstacle_dataset_file, goal_dataset_file)
            mean_successes=np.mean(np.array(successes))
            mean_crashes=np.mean(np.array(crashes))
            mean_stucks=np.mean(np.array(stucks))
            mean_steps_success=np.mean(np.array(mean_steps_success))
            mean_triggered_x_stuck = np.mean(np.array(triggered_x_stuck))
            mean_triggered_y_stuck = np.mean(np.array(triggered_y_stuck))
            mean_stuck_after_stuck = np.mean(np.array(stuck_after_stuck))
            #mean_stuck_after_ystuck = np.mean(np.array(stuck_after_ystuck))
            mean_crash_after_stuck = np.mean(np.array(crash_after_stuck))
            #mean_crash_after_ystuck = np.mean(np.array(crash_after_ystuck))
            mean_stuck_count = np.mean(np.array(stuck_count))
            successes_dict[type_train][on]=mean_successes
            crashes_dict[type_train][on]=mean_crashes
            stucks_dict[type_train][on]=mean_stucks
            steps_success_dict[type_train][on]=mean_steps_success

            print('Mean successes ', mean_successes)
            print('Mean crashes ', mean_crashes)
            print('Mean stucks ', mean_stucks)
            print('Mean steps success ', mean_steps_success)
            print('mean total triggered x stuck across the 100 environments',mean_triggered_x_stuck)
            print('mean total triggered y stuck across the 100 environments',mean_triggered_y_stuck)
            print('mean total stuck count across the 100 environments: ',mean_stuck_count)
            print('mean total stuck after stuck triggered across 100 environments',mean_stuck_after_stuck)
            #print('mean total stuck after y stuck triggered across 100 environments',mean_stuck_after_ystuck)
            print('mean total crash after stuck triggered across 100 environments',mean_crash_after_stuck)
            #print('mean total crash after y stuck triggered across 100 environments',mean_crash_after_ystuck)


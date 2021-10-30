from modules.game_module import game_module
import numpy as np
train_file = './data/game_data_2state_based.out'
# train_file='./data/game_data_random_goal.out'


live = int(input("Enter 0 to play the game live, 1 to autoplay using HDC recall, 2 to autoplay using NN recall, 3 to test using HDC recall, 4 to test using NN recall\n"))
gametype = int(input("Enter 1 to play with goals, 0 without\n"))

num = '20'
a = game_module('game_data_random_goal.out', int(num))

if live==0:
    a.play_game(gametype)
elif live==1:
    a.train_from_file(train_file)
    a.autoplay_game(gametype, False)
elif live==2:
    a.train_from_file_NN(train_file)
    a.autoplay_game(gametype, True)
elif live==3:
    # count = int(input("Enter the number of testing loops\n"))
    # num = ['5', '10', '15', '20']
    # a.train_from_file(train_file)
    # for n in num:
    #     a.num_obs = int(n)
    #     a.play_game_from_file('./data/obstacles_' + n + '.csv', './data/goals_' + n + '.csv', False, count)

    count = int(input("Enter the number of testing loops\n"))
    a.num_obs = 15
    dims = [50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    a.set_softmax_param(7.79)

    for d in dims:
        a.set_dim(d)
        a.train_from_file(train_file)
        a.play_game_from_file('./data/obstacles_' + '15' + '.csv', './data/goals_' + '15' + '.csv', False, count)
        print('STD: ', np.std(a.final_accuracy))
elif live==4:
    # num = ['5', '10', '15', '20']
    # count = int(input("Enter the number of testing loops\n"))
    # a.train_from_file_NN(train_file)
    # a.set_softmax_param(7.5)
    # for n in num:
    #     a.num_obs = int(n)
    #     a.play_game_from_file('./data/obstacles_' + n + '.csv', './data/goals_' + n + '.csv', True, count)
    count = int(input("Enter the number of testing loops\n"))
    a.set_softmax_param(7.5)
    a.num_obs = 15
    # dims = [50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    # dims = [10000]

    for d in dims:
        a.set_dim(d)
        a.train_from_file_NN(train_file)
        a.play_game_from_file('./data/obstacles_' + '15' + '.csv', './data/goals_' + '15' + '.csv', True, count)
        print('STD: ', np.std(a.final_accuracy))





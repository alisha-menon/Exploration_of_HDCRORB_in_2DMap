from modules.game_module import game_module
import csv

train_file = './data/game_data_2state_based.out'
# train_file='./data/game_data_random_goal.out'

gametype = 1
obstacles= 15

#Obstacle number (line number in the goal/env file) -- lines start with 0
number=45

obstacle_dataset_file = './data/obstacles_' + str(obstacles) + '.csv'
goal_dataset_file = './data/goals_' + str(obstacles) + '.csv'

test_reader = csv.reader(open(obstacle_dataset_file, 'r'))
goal_reader = csv.reader(open(goal_dataset_file, 'r'))

test_ids=list(test_reader)
goal_ids=list(goal_reader)

print(test_ids)

a = game_module('./data/testout.out',obstacles)
a.train_from_file(train_file)
a.play_display_game(gametype,[int(num) for num in test_ids[number]], [int(num) for num in goal_ids[number]])
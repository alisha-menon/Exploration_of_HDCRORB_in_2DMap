from modules.game_module import game_module
import numpy as np
import matplotlib.pyplot as plt

train_file = './data/game_data_2state_based.out'
#train_file = './data/game_dat_test4.out'
game_data = np.loadtxt(train_file, dtype = np.int8, delimiter=',')
n_samples = game_data.shape[0]
print (n_samples)

#threshold = 0.25
#a = game_module()
#a.train_from_file(train_file, threshold)
#success, crash, stuck = a.test_game(1000)

trials = 100
#%%
#thresholds = [0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29]
#thresholds = [0.29]

thresholds = np.linspace(0,1,101)
threshold_nostate = 0.05
threshold_state1 = 0.16
threshold_state2 = 0.04
softmax_param = 7.79
softmax_param_state1 = 10.12
softmax_param_state2 = 10.12
#print(thresholds)

# success = 0
# crash = 0
# stuck = 0
# num_cond = 0
# num_thrown = 0
# a = game_module()
# a.set_threshold_known(threshold)
# a.set_threshold_known_state1(threshold_state1)
# a.set_threshold_known_state2(threshold_state2)
# a.set_softmax_param(softmax_param)
# a.set_softmax_param_state1(softmax_param_state1)
# a.set_softmax_param_state2(softmax_param_state2)
# print(threshold)
# a.train_from_file(train_file)
# print("state 1 HDprogHV: ",a.hd_module.hd_program_vec_state1)
# print("state 2 HDprogHV: ",a.hd_module.hd_program_vec_state2)
# num_thrown_state1 = a.num_thrown_state1
# num_cond_state1 = a.num_cond_state1
# num_thrown_state2 = a.num_thrown_state2
# num_cond_state2 = a.num_cond_state2
# print("ratio of thrown samples: ",(num_thrown_state1+num_thrown_state2)/n_samples)
# print("ratio of trained samples: ",(num_cond_state1+num_thrown_state2)/n_samples)
# success, crash, stuck = a.test_game(trials)
# print("ratio of crashed trials: ",crash/trials)
# print("ratio of stuck trials: ", stuck/trials)
# print("ratio of successful trials: ", success/trials)



success=np.empty((len(thresholds),))
crash=np.empty((len(thresholds),))
stuck=np.empty((len(thresholds),))
steps = np.empty((len(thresholds),))
num_cond_list = np.empty((len(thresholds),))
num_thrown_list = np.empty((len(thresholds),))
sum_success = 0
best_success = 0
best_threshold = 0
crashes = 0
stucks = 0
average_steps = 0
for i,threshold in enumerate(thresholds):
    a = game_module()
    #a.set_sensor_weight(weight)
    a.set_threshold_known(threshold_nostate)
    a.set_threshold_known_state1(threshold_state1)
    a.set_threshold_known_state2(threshold_state2)
    a.set_softmax_param(softmax_param)
    a.set_softmax_param_state1(softmax_param_state1)
    a.set_softmax_param_state2(softmax_param_state2)
    #print(i)
    print(threshold,"\n")
    a.train_from_file(train_file)
    num_thrown_list[i] = a.num_thrown
    num_cond_list[i] = a.num_cond
    #steps[i] = a.average_steps
    #print(num_thrown_list[i]/n_samples)
    #print(num_cond_list[i]/n_samples)
    success[i], crash[i], stuck[i],steps[i] = a.test_game(trials)
    #print(crash[i])
    #print(stuck[i])
    sum_success += success[i]/trials
    if success[i]/trials > best_success:
        best_success = success[i]/trials
        best_threshold = threshold
    crashes += crash[i]
    stucks += stuck[i]
    average_steps += steps[i]
print('done with tests')
print("average success rate: ", sum_success/len(thresholds))
print("average crashes: ", crashes/len(thresholds))
print("average stuck trials: ", stucks/len(thresholds))
print("average steps per trials:",average_steps/len(thresholds))
print("best_threshold: ",best_threshold)
print("best_success: ",best_success)
plt.plot(thresholds,success/trials, label='Success')
plt.plot(thresholds,crash/trials,label='Crash')
plt.plot(thresholds,stuck/trials,label='Stuck')
#plt.plot(thresholds,num_cond_list/n_samples,label='Trained conditions')
#plt.plot(thresholds,num_thrown_list/n_samples,label='Rejected conditions')
plt.legend()
#plt.ylim(0,1)
plt.title('Success, crash, and stuck rate vs. new condition threshold')
plt.xlabel('New condition threshold')
plt.ylabel('Percentage (%)')
plt.show()

# softmax_params = np.linspace(1,25,101)
# #threshold = 0.08
# #threshold_state1 = 0.2
# #threshold_state2 = 0.08
# print(softmax_params)
# success=np.empty((len(softmax_params),))
# crash=np.empty((len(softmax_params),))
# stuck=np.empty((len(softmax_params),))
# num_cond_list = np.empty((len(softmax_params),))
# num_thrown_list = np.empty((len(softmax_params),))
# for i,softmax_param in enumerate(softmax_params):
#     a = game_module()
#     softmax_param_state1 = softmax_param
#     softmax_param_state2 = softmax_param
#     #a.set_sensor_weight(weight)
#     a.set_threshold_known(threshold_nostate)
#     a.set_threshold_known_state1(threshold_state1)
#     a.set_threshold_known_state2(threshold_state2)
#     a.set_softmax_param(softmax_param)
#     a.set_softmax_param_state1(softmax_param_state1)
#     a.set_softmax_param_state2(softmax_param_state2)
#     #print(i)
#     print(softmax_param)
#     a.train_from_file(train_file)
#     num_thrown_list[i] = a.num_thrown_state2 + a.num_thrown_state1
#     num_cond_list[i] = a.num_cond_state2 + a.num_cond_state1
#     print(num_thrown_list[i]/n_samples)
#     print(num_cond_list[i]/n_samples)
#     success[i], crash[i], stuck[i] = a.test_game(trials)
#     print(crash[i])
#     print(stuck[i])
# print('done with tests')
# plt.plot(softmax_params,success/trials, label='Success')
# plt.plot(softmax_params,crash/trials,label='Crash')
# plt.plot(softmax_params,stuck/trials,label='Stuck')
# #plt.plot(thresholds,num_cond_list/n_samples,label='Trained conditions')
# #plt.plot(thresholds,num_thrown_list/n_samples,label='Rejected conditions')
# plt.legend()
# plt.title('Success rate, crashes, and stuck samples vs. softmax scaling parameter')
# plt.xlabel('softmax scaling parameter')
# plt.ylabel('Percentage (%)')
# plt.show()







import random
import numpy as np
import os
import pickle
import math
from scipy.special import softmax

# from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
# from sklearn import neighbors
# from sklearn.linear_model import SGDClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split
#
import tensorflow as tf

class hd_module:
    def __init__(self, d=10000):
        # HD dimension used
        self.dim = d

        self.num_sensors = 7
        self.num_actuators = 4
        self.sensor_weight = 1

        self.threshold_known = 0.05

        self.threshold_known_state1 = 0.16
        self.threshold_known_state2 = 0.04

        self.threshold_known_only_x = 0.16
        self.threshold_known_only_y = 0.04

        self.softmax_param = 7.79
        self.softmax_param_state1 = 10.12
        self.softmax_param_state2 = 10.12

        self.softmax_param_only_x = 10.12
        self.softmax_param_only_y = 10.12
        # method 1(0), 2(1), 3(2)
        # want to modify so that given global information prioritize obstacle avoidance, or goal. Specifically see issue where the direction of goal is through an
        # obstacle, and going around the obstacle means going away from the goal temporarily. The agent is not able to make that choice to go away from goal
        # temporarily in order to get around an obstacle. Could define 3 states
        # 1. no obstacle, prioritize goal
        # 2. There are obstacles, but still want to generally head in direction of goal
        # 3. Have been stuck in a cycle for the past some_threshold moves, need to set aside goal completely and instead just focus on going away from obstacles

        #goal method alone = method 4
        self.encoding_method = self.encoding_method_helper(3)
        #set to 1 to use softmax, set to 0 to not
        self.activation_function = 1
        # set to 1 to activate the two state method
        self.two_states = 0

        # set to 1 to activate the stuck identification
        self.stuck_id = 0
        self.stuck_state_machine = 1


        self.output_vectors = []
        self.output_actuators = []

        self.outdir = './data/'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        sensor_ids_fname = self.outdir + 'hd_sensor_ids_dim_' + str(self.dim) + '.npy'
        sensor_vals_fname = self.outdir + 'hd_sensor_vals_dim_' + str(self.dim) + '.npy'
        sensor_dist_fname = self.outdir + 'hd_sensor_dist_' + str(self.dim) + '.npy'
        sensor_last_fname= self.outdir + 'hd_sensor_last_' + str(self.dim) + '.npy'
        actuator_vals_fname = self.outdir + 'hd_actuator_vals_dim_' + str(self.dim) + '.npy'

        # Load/create HD items
        if os.path.exists(sensor_ids_fname):
            self.hd_sensor_ids = np.load(sensor_ids_fname)
        else:
            print("creating sensor id mem")
            self.hd_sensor_ids = self.create_bipolar_mem(self.num_sensors,self.dim)
            np.save(sensor_ids_fname, self.hd_sensor_ids)

        if os.path.exists(sensor_vals_fname):
            self.hd_sensor_vals = np.load(sensor_vals_fname)
        else:
            print("creating sensor val mem")
            self.hd_sensor_vals = self.create_bipolar_mem(2,self.dim)
            np.save(sensor_vals_fname, self.hd_sensor_vals)

        if os.path.exists(sensor_dist_fname):
            self.hd_sensor_dist = np.load(sensor_dist_fname)
        else:
            print("creating sensor dist mem")
            #self.hd_sensor_dist = self.create_bipolar_CIM(19,self.dim)
            self.hd_sensor_dist = self.create_bipolar_mem(3,self.dim)
            np.save(sensor_dist_fname, self.hd_sensor_dist)

        if os.path.exists(sensor_last_fname):
            self.hd_sensor_last = np.load(sensor_last_fname)
        else:
            print("creating sensor last mem")
            self.hd_sensor_last = self.create_bipolar_mem(4,self.dim)
            np.save(sensor_last_fname, self.hd_sensor_last)

        if os.path.exists(actuator_vals_fname):
            self.hd_actuator_vals = np.load(actuator_vals_fname)
        else:
            print("creating actuator val mem")
            self.hd_actuator_vals = self.create_bipolar_mem(self.num_actuators,self.dim)
            np.save(actuator_vals_fname, self.hd_actuator_vals)


        self.hd_x_act_vals = self.hd_actuator_vals[0:2,:] #first two
        self.hd_y_act_vals = self.hd_actuator_vals[2:,:]

        # Initialize program vector
        self.hd_program_vec = np.zeros((self.dim,), dtype = np.int)
        self.hd_program_vec_state1 = np.zeros((self.dim,), dtype = np.int)
        self.hd_program_vec_state2 = np.zeros((self.dim,), dtype = np.int)
        self.hd_program_vec_goalx = np.zeros((self.dim,), dtype = np.int)
        self.hd_program_vec_goaly = np.zeros((self.dim,), dtype = np.int)
        self.hd_program_vec_only_x = np.zeros((self.dim,), dtype = np.int)
        self.hd_program_vec_only_y = np.zeros((self.dim,), dtype = np.int)

        # Initialize condition vector
        self.hd_cond_vec = np.zeros((self.dim,), dtype = np.int)
        self.hd_cond_vec_state1 = np.zeros((self.dim,), dtype = np.int)
        self.hd_cond_vec_state2 = np.zeros((self.dim,), dtype = np.int)
        self.hd_cond_vec_goalx = np.zeros((self.dim,), dtype = np.int)
        self.hd_cond_vec_goaly = np.zeros((self.dim,), dtype = np.int)
        self.hd_cond_vec_only_x = np.zeros((self.dim,), dtype = np.int)
        self.hd_cond_vec_only_y = np.zeros((self.dim,), dtype = np.int)

        self.num_cond = 0
        self.num_cond_state1 = 0
        self.num_cond_state2 = 0
        self.num_cond_goalx = 0
        self.num_cond_goaly = 0
        self.num_cond_only_x = 0
        self.num_cond_only_y = 0

        self.num_thrown = 0
        self.num_thrown_state1 = 0
        self.num_thrown_state2 = 0
        self.num_thrown_goalx = 0
        self.num_thrown_goaly = 0
        self.num_thrown_only_x = 0
        self.num_thrown_only_y = 0

        # self.clf = svm.SVC()
        # self.clf = RandomForestClassifier()
        # self.clf = ExtraTreesClassifier()
        # self.clf = DecisionTreeClassifier()
        # self.clf = GradientBoostingClassifier()
        # self.clf = BaggingClassifier()
        # self.clf = BaggingClassifier(RandomForestClassifier())
        # self.clf = neighbors.KNeighborsClassifier()
        # self.clf = SGDClassifier()
        # self.clf = MLPClassifier()
        # self.clf = GaussianNB()
        # self.clf = LogisticRegression()
        self.model = ''

        self.ang_CiM = self.create_bipolar_CIM(3, self.dim)
        self.mag_CiM = self.create_bipolar_CIM(5, self.dim)

    def set_softmax_param(self, s_p):
        self.softmax_param = s_p

    def create_bipolar_mem(self, numitem, dim):
        # Creates random bipolar memory of given size

        rand_arr = np.rint(np.random.rand(numitem, dim)).astype(np.int8)
        return (rand_arr*2 - 1)

    def create_bipolar_CIM(self, numitem, dim):
        # Creates random bipolar memory of given size
        rand_arr = np.rint(np.random.rand(dim,)).astype(np.int8)
        bipolar_arr = rand_arr*2 - 1
        neg_arr = -bipolar_arr

        CIM = np.zeros((numitem,dim)).astype(np.int8)
        block = dim//numitem
        for i in range(numitem):
            CIM[i,:i*block] = bipolar_arr[:i*block]
            CIM[i,i*block:] = neg_arr[i*block:]
        '''
        rand_arr = np.rint(np.random.rand(2, dim)).astype(np.int8)
        bipolar_arr = rand_arr*2 - 1

        CIM = np.zeros((numitem,dim)).astype(np.int8)
        block = dim//numitem
        for i in range(numitem):
            CIM[i,:i*block] = bipolar_arr[0,:i*block]
            CIM[i,i*block:] = bipolar_arr[1,i*block:]
        '''

        return CIM

    def hd_mul(self, A, B):
        # Return element-wise multiplication between bipolar HD vectors
        # inputs:
        #   - A: bipolar HD vector
        #   - B: bipolar HD vector
        # outputs:
        #   - A*B: bipolar HD vector
        return np.multiply(A,B,dtype = np.int8)

    def hd_perm(self, A):
        # Return right cyclic shift of input bipolar vector
        # inputs:
        #   - A: bipiolar HD vector
        # outputs:
        #   - rho(A): bipolar HD vector
        return np.roll(A,1)

    def hd_threshold(self, A):
        # Given integer vector, threshold at zero to bipolar
        # inputs:
        #   - A: bipolar HD vector
        # outputs:
        #   - [A]: bipolar HD vector
        #return (np.greater_equal(A,0, dtype=np.int8)*2-1)
        return (np.int8(np.greater_equal(A,0))*2-1)

    def search_actuator_vals(self, A):
        # Find the nearest item in 'hd_actuator_vals' according to Inner product distance
        # inputs:
        #   - A: bipolar HD vector
        # outputs:
        #   - i: integer index of closest item in 'hd_actuator_vals'
        dists = np.matmul(self.hd_actuator_vals, A, dtype = np.int)
        return np.argmax(dists)

    def search_x_actuator_vals(self, A):
        # Find the nearest item in 'hd_actuator_vals' according to Inner Product distance
        # inputs:
        #   - A: bipolar HD vector
        # outputs:
        #   - i: integer index of closest item in 'hd_actuator_vals'
        dists = np.matmul(self.hd_x_act_vals, A, dtype = np.int)
        return np.argmax(dists)

    def search_y_actuator_vals(self, A):
        # Find the nearest item in 'hd_actuator_vals' according to Inner Product distance
        # inputs:
        #   - A: bipolar HD vector
        # outputs:
        #   - i: integer index of closest item in 'hd_actuator_vals'
        dists = np.matmul(self.hd_y_act_vals, A, dtype = np.int)
        return np.argmax(dists)

    def softmax_actuator_vals(self, A, softmax_param):
        # Probabilisticly chooses an hd vector from 'hd_actuator_vals' according 
        # to probabilities defined by the Inner Product distance
        # inputs:
        #   - A: bipolar HD vector
        # outputs:
        #   - i: integer index of closest item in 'hd_actuator_vals'
        dists = np.matmul(self.hd_actuator_vals, A, dtype = np.int)
        probs = softmax(dists/np.max(dists)*softmax_param)
        #print(dists)
        #print(probs)

        return np.random.choice(4, p = probs)

    def encoding_method_helper(self, method):
        if method==1:
            return self.encode_sensors
        elif method==4:
            return self.encode_sensors_goal
        elif method==5:
            return self.encode_sensors_extrainfo
        return lambda sensor_in, train: self.encode_sensors_directional(sensor_in, train, method)
        # return lambda sensor_in, train: self.encode_sensors_x_direction(sensor_in, train, method)

        #method 1
    def encode_sensors(self, sensor_in, train):
        # Encode sensory data into HD space
        # Currently binds together all sensor inputs
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - sensor_vec: bipolar HD vector
        sensor_vec = np.ones((self.dim,), dtype = np.int8) #bind
        #sensor_vec = np.zeros((self.dim,), dtype = np.int8) #bundle
        for i,sensor_val in enumerate(sensor_in[:4]):
            permuted_vec = self.hd_sensor_vals[sensor_val,:]
            for j in range(i):
                # permute hd_sensor_val based on the corresponding sensor id
                permuted_vec = self.hd_perm(permuted_vec)
            binded_sensor = self.hd_mul(self.hd_sensor_ids[i,:],permuted_vec)
            sensor_vec = self.hd_mul(sensor_vec, binded_sensor) #bind
            #sensor_vec = sensor_vec + binded_sensor #bundle
        #sensor_vec = self.hd_threshold(sensor_vec) #bundle

        #xdist_vec = self.hd_mul(self.hd_sensor_ids[4,:], self.hd_sensor_dist[sensor_in[4] + 9,:])
        #ydist_vec = self.hd_mul(self.hd_sensor_ids[5,:], self.hd_perm(self.hd_sensor_dist[sensor_in[5] + 9,:]))
        if sensor_in[4] > 0:
            xval = self.hd_sensor_dist[2]
        elif sensor_in[4] < 0:
            xval = self.hd_sensor_dist[0]
        else:
            xval = self.hd_sensor_dist[1]

        if sensor_in[5] > 0:
            yval = self.hd_sensor_dist[2]
        elif sensor_in[5] < 0:
            yval = self.hd_sensor_dist[0]
        else:
            yval = self.hd_sensor_dist[1]
        yval = self.hd_perm(yval)

        xdist_vec = self.hd_mul(self.hd_sensor_ids[4,:], xval)
        ydist_vec = self.hd_mul(self.hd_sensor_ids[5,:], yval)
        dist_vec = self.hd_mul(xdist_vec, ydist_vec)

        last_vec = self.hd_sensor_last[sensor_in[6],:]
        #last_vec = self.hd_mul(dist_vec, last_vec)

        #return self.hd_mul(sensor_vec, last_vec)
#        if train:
#            return self.hd_threshold(last_vec + self.sensor_weight*sensor_vec + dist_vec)
#        else:
#            return self.hd_threshold(last_vec + sensor_vec + dist_vec)
        #random_vec = np.squeeze(self.create_bipolar_mem(1,self.dim))


        return self.hd_threshold(last_vec + sensor_vec + dist_vec)

    def encode_sensors_goal(self, sensor_in, train):
        # Encode sensory data into HD space
        # Currently binds together all sensor inputs
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - sensor_vec: bipolar HD vector
        sensor_vec = np.empty((self.dim,4), dtype = np.int8) #bind
        for i,sensor_val in enumerate(sensor_in[:4]):
            permuted_vec = self.hd_sensor_vals[sensor_val,:]
            for j in range(i):
                # permute hd_sensor_val based on the corresponding sensor id
                permuted_vec = self.hd_perm(permuted_vec)

            binded_sensor = self.hd_mul(self.hd_sensor_ids[i,:],permuted_vec)
            sensor_vec[:,i] = binded_sensor

        xsensors = self.hd_mul(sensor_vec[:,0],sensor_vec[:,1])
        ysensors = self.hd_mul(sensor_vec[:,2],sensor_vec[:,3])

        #xdist_vec = self.hd_mul(self.hd_sensor_ids[4,:], self.hd_sensor_dist[sensor_in[4] + 9,:])
        #ydist_vec = self.hd_mul(self.hd_sensor_ids[5,:], self.hd_perm(self.hd_sensor_dist[sensor_in[5] + 9,:]))
        if sensor_in[4] > 0:
            xval = self.hd_sensor_dist[2]
        elif sensor_in[4] < 0:
            xval = self.hd_sensor_dist[0]
        else:
            xval = self.hd_sensor_dist[1]

        if sensor_in[5] > 0:
            yval = self.hd_sensor_dist[2]
        elif sensor_in[5] < 0:
            yval = self.hd_sensor_dist[0]
        else:
            yval = self.hd_sensor_dist[1]
        yval = self.hd_perm(yval)

        xdist_vec = self.hd_mul(self.hd_sensor_ids[4,:], xval)
        ydist_vec = self.hd_mul(self.hd_sensor_ids[5,:], yval)
        dist_vec = self.hd_mul(xdist_vec, ydist_vec)

        last_vec = self.hd_sensor_last[sensor_in[6],:]

        other_vec = self.hd_threshold(sensor_vec[:,0] + sensor_vec[:,1] + sensor_vec[:,2] + sensor_vec[:,3] + last_vec)
        # other_vec = self.hd_threshold(self.hd_mul(xsensors, ysensors) + last_vec)

        return self.hd_mul(other_vec, dist_vec)

    def encode_sensors_extrainfo(self, sensor_in, train):
        # Encode sensory data into HD space
        # Currently binds together all sensor inputs
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - sensor_vec: bipolar HD vector
        sensor_vec = np.empty((self.dim,4), dtype = np.int8) #bind
        for i,sensor_val in enumerate(sensor_in[:4]):
            permuted_vec = self.hd_sensor_vals[int(sensor_val),:]
            for j in range(i):
                # permute hd_sensor_val based on the corresponding sensor id
                permuted_vec = self.hd_perm(permuted_vec)

            binded_sensor = self.hd_mul(self.hd_sensor_ids[i,:],permuted_vec)
            sensor_vec[:,i] = binded_sensor

        xsensors = self.hd_mul(sensor_vec[:,0],sensor_vec[:,1])
        ysensors = self.hd_mul(sensor_vec[:,2],sensor_vec[:,3])


        if sensor_in[4] > 0:
            xval = self.hd_sensor_dist[2]
        elif sensor_in[4] < 0:
            xval = self.hd_sensor_dist[0]
        else:
            xval = self.hd_sensor_dist[1]

        if sensor_in[5] > 0:
            yval = self.hd_sensor_dist[2]
        elif sensor_in[5] < 0:
            yval = self.hd_sensor_dist[0]
        else:
            yval = self.hd_sensor_dist[1]
        yval = self.hd_perm(yval)

        xdist_vec = self.hd_mul(self.hd_sensor_ids[4,:], xval)
        ydist_vec = self.hd_mul(self.hd_sensor_ids[5,:], yval)
        dist_vec = self.hd_mul(xdist_vec, ydist_vec)

        xsense = self.hd_mul(xsensors,xdist_vec)
        ysense = self.hd_mul(ysensors,ydist_vec)

        last_vec = self.hd_sensor_last[int(sensor_in[8]),:]

        ang = sensor_in[6]
        ang_ind = (ang + np.pi) / (2 * np.pi)
        ang_ind = np.floor(ang_ind * len(self.ang_CiM))
        if ang_ind == len(self.ang_CiM):
            ang_ind = len(self.ang_CiM) - 1

        mag = sensor_in[7]
        mag_ind = np.floor(mag-1)

        ang_vec = self.ang_CiM[int(ang_ind)]
        mag_vec = self.mag_CiM[int(mag_ind)]


        #last_vec = self.hd_mul(dist_vec, last_vec)

        #return self.hd_mul(sensor_vec, last_vec)
#        if train:
#            return self.hd_threshold(last_vec + self.sensor_weight*sensor_vec + dist_vec)
#        else:
#            return self.hd_threshold(last_vec + sensor_vec + dist_vec)
        #random_vec = np.squeeze(self.create_bipolar_mem(1,self.dim))

        return self.hd_mul(self.hd_mul(xsensors,ysensors), self.hd_threshold(xdist_vec + ydist_vec + last_vec + ang_vec + mag_vec))


    #methods 2 and 3
    def encode_sensors_directional(self, sensor_in, train, method):
        # Encode sensory data into HD space
        # Currently binds together all sensor inputs
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - sensor_vec: bipolar HD vector
        sensor_vec = np.empty((self.dim,4), dtype = np.int8) #bind
        for i,sensor_val in enumerate(sensor_in[:4]):
            permuted_vec = self.hd_sensor_vals[int(sensor_val),:]
            for j in range(i):
                # permute hd_sensor_val based on the corresponding sensor id
                permuted_vec = self.hd_perm(permuted_vec)

            binded_sensor = self.hd_mul(self.hd_sensor_ids[i,:],permuted_vec)
            sensor_vec[:,i] = binded_sensor

        xsensors = self.hd_mul(sensor_vec[:,0],sensor_vec[:,1])
        ysensors = self.hd_mul(sensor_vec[:,2],sensor_vec[:,3])


        if sensor_in[4] > 0:
            xval = self.hd_sensor_dist[2]
        elif sensor_in[4] < 0:
            xval = self.hd_sensor_dist[0]
        else:
            xval = self.hd_sensor_dist[1]

        if sensor_in[5] > 0:
            yval = self.hd_sensor_dist[2]
        elif sensor_in[5] < 0:
            yval = self.hd_sensor_dist[0]
        else:
            yval = self.hd_sensor_dist[1]
        yval = self.hd_perm(yval)

        xdist_vec = self.hd_mul(self.hd_sensor_ids[4,:], xval)
        ydist_vec = self.hd_mul(self.hd_sensor_ids[5,:], yval)
        dist_vec = self.hd_mul(xdist_vec, ydist_vec)

        xsense = self.hd_mul(xsensors,xdist_vec)
        ysense = self.hd_mul(ysensors,ydist_vec)

        # ang = sensor_in[4]
        # ang_ind = 1
        # if ang < 0:
        #     ang_ind = 0
        # elif ang > 0:
        #     ang_ind = 2
        #
        # mag = sensor_in[5]
        # mag_ind = 0
        # if mag < 3:
        #     mag_ind = 4
        # elif mag < 6:
        #     mag_ind = 3
        # elif mag < 9:
        #     mag_ind = 2
        # elif mag < 12:
        #     mag_ind = 1
        #
        # xdist_vec = self.ang_CiM[int(ang_ind)]
        # ydist_vec = self.mag_CiM[int(mag_ind)]

        last_vec = self.hd_sensor_last[int(sensor_in[6]),:]
        #last_vec = self.hd_mul(dist_vec, last_vec)

        #return self.hd_mul(sensor_vec, last_vec)
#        if train:
#            return self.hd_threshold(last_vec + self.sensor_weight*sensor_vec + dist_vec)
#        else:
#            return self.hd_threshold(last_vec + sensor_vec + dist_vec)
        #random_vec = np.squeeze(self.create_bipolar_mem(1,self.dim))

        if method==2:
            ## method 2
            return self.hd_threshold(xsense + ysense + last_vec)
        elif method==3:
            ## method 3
            return self.hd_mul(self.hd_mul(xsensors,ysensors), self.hd_threshold(xdist_vec + ydist_vec + last_vec))

    #Consider only x direction
    def encode_sensors_x_direction(self, sensor_in, train, method):
        # Encode sensory data into HD space
        # Currently binds together all sensor inputs
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - sensor_vec: bipolar HD vector
        sensor_vec = np.empty((self.dim, 4), dtype=np.int8)  # bind
        for i, sensor_val in enumerate(sensor_in[:4]):
            permuted_vec = self.hd_sensor_vals[sensor_val, :]
            for j in range(i):
                # permute hd_sensor_val based on the corresponding sensor id
                permuted_vec = self.hd_perm(permuted_vec)

            binded_sensor = self.hd_mul(self.hd_sensor_ids[i, :], permuted_vec)
            sensor_vec[:, i] = binded_sensor

        xsensors = self.hd_mul(sensor_vec[:, 0], sensor_vec[:, 1])
        ysensors = self.hd_mul(sensor_vec[:, 2], sensor_vec[:, 3])

        if sensor_in[4] > 0:
            xval = self.hd_sensor_dist[2]
        elif sensor_in[4] < 0:
            xval = self.hd_sensor_dist[0]
        else:
            xval = self.hd_sensor_dist[1]

        xdist_vec = self.hd_mul(self.hd_sensor_ids[4, :], xval)
        last_vec = self.hd_sensor_last[sensor_in[6], :]

        return self.hd_mul(self.hd_mul(xsensors, ysensors), self.hd_threshold(xdist_vec + last_vec))

    #Consider only y direction
    def encode_sensors_y_direction(self, sensor_in, train, method):
        # Encode sensory data into HD space
        # Currently binds together all sensor inputs
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - sensor_vec: bipolar HD vector
        sensor_vec = np.empty((self.dim, 4), dtype=np.int8)  # bind
        for i, sensor_val in enumerate(sensor_in[:4]):
            permuted_vec = self.hd_sensor_vals[sensor_val, :]
            for j in range(i):
                # permute hd_sensor_val based on the corresponding sensor id
                permuted_vec = self.hd_perm(permuted_vec)

            binded_sensor = self.hd_mul(self.hd_sensor_ids[i, :], permuted_vec)
            sensor_vec[:, i] = binded_sensor

        xsensors = self.hd_mul(sensor_vec[:, 0], sensor_vec[:, 1])
        ysensors = self.hd_mul(sensor_vec[:, 2], sensor_vec[:, 3])

        if sensor_in[5] > 0:
            yval = self.hd_sensor_dist[2]
        elif sensor_in[5] < 0:
            yval = self.hd_sensor_dist[0]
        else:
            yval = self.hd_sensor_dist[1]
        yval = self.hd_perm(yval)

        ydist_vec = self.hd_mul(self.hd_sensor_ids[5, :], yval)

        last_vec = self.hd_sensor_last[sensor_in[6], :]

        return self.hd_mul(self.hd_mul(xsensors, ysensors), self.hd_threshold(ydist_vec + last_vec))


    def new_condition(self, condition_vec, threshold, cond_vec_input):
        dist = np.matmul(condition_vec, self.hd_threshold(cond_vec_input), dtype = np.int)
        pct = dist/self.dim
        if (pct > threshold):
            return 0
        else:
            return 1

    def train_sample(self, sensor_in, act_in):
        # Multiply encoded sensor vector with actuator vector
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        #   - act_in: integer representing actuator action
        # outputs:
        #   - sample_vec: bipolar HD vector

        #for method 1, use encode_sensors
        #for method 2 and 3, use encoder_sensors_directional and change the last line of that function depending on the method
        sensor_vec = self.encoding_method(sensor_in,True)
        if self.new_condition(sensor_vec, self.threshold_known, self.hd_cond_vec):
            act_vec = self.hd_actuator_vals[act_in,:]
            sample_vec = self.hd_mul(sensor_vec,act_vec)
            self.hd_cond_vec += sensor_vec
            self.num_cond += 1
        else:
            sample_vec = np.zeros((self.dim), dtype=np.int8)
            self.num_thrown += 1

        return sample_vec

    def train_sample_state1(self, sensor_in, act_in):
        # Multiply encoded sensor vector with actuator vector
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        #   - act_in: integer representing actuator action
        # outputs:
        #   - sample_vec: bipolar HD vector

        #for method 1, use encode_sensors
        #for method 2 and 3, use encoder_sensors_directional and either 2 or 3
        #for goal method use encode_sensors_goal
        sensor_vec = self.encode_sensors_goal(sensor_in,True)
        if self.new_condition(sensor_vec, self.threshold_known_state1, self.hd_cond_vec_state1):
            act_vec = self.hd_actuator_vals[act_in,:]
            sample_vec = self.hd_mul(sensor_vec,act_vec)
            self.hd_cond_vec_state1 += sensor_vec
            self.num_cond_state1 += 1
        else:
            sample_vec = np.zeros((self.dim), dtype=np.int8)
            self.num_thrown_state1 += 1

        return sample_vec

    def train_sample_state2(self, sensor_in, act_in):
        # Multiply encoded sensor vector with actuator vector
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        #   - act_in: integer representing actuator action
        # outputs:
        #   - sample_vec: bipolar HD vector

        #for method 1, use encode_sensors
        #for method 2 and 3, use encoder_sensors_directional and either 2 or 3
        sensor_vec = self.encode_sensors_directional(sensor_in,True,3)
        if self.new_condition(sensor_vec, self.threshold_known_state2, self.hd_cond_vec_state2):
            act_vec = self.hd_actuator_vals[act_in,:]
            sample_vec = self.hd_mul(sensor_vec,act_vec)
            self.hd_cond_vec_state2 += sensor_vec
            self.num_cond_state2 += 1
        else:
            sample_vec = np.zeros((self.dim), dtype=np.int8)
            self.num_thrown_state2 += 1

        return sample_vec

    #Model that considers only movement in x
    def train_sample_goalx(self, sensor_in, act_in):
        # Multiply encoded sensor vector with actuator vector
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        #   - act_in: integer representing actuator action
        # outputs:
        #   - sample_vec: bipolar HD vector

        sensor_vec = self.encode_sensors_x_direction(sensor_in,True,3)
        if self.new_condition(sensor_vec, self.threshold_known, self.hd_cond_vec):
            act_vec = self.hd_actuator_vals[act_in,:]
            sample_vec = self.hd_mul(sensor_vec,act_vec)
            self.hd_cond_vec_state1 += sensor_vec
            self.num_cond_state1 += 1
        else:
            sample_vec = np.zeros((self.dim), dtype=np.int8)
            self.num_thrown_state1 += 1

        return sample_vec

    #Model that considers only movement in y
    def train_sample_goaly(self, sensor_in, act_in):
        # Multiply encoded sensor vector with actuator vector
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        #   - act_in: integer representing actuator action
        # outputs:
        #   - sample_vec: bipolar HD vector

        sensor_vec = self.encode_sensors_y_direction(sensor_in,True,3)
        if self.new_condition(sensor_vec, self.threshold_known, self.hd_cond_vec):
            act_vec = self.hd_actuator_vals[act_in,:]
            sample_vec = self.hd_mul(sensor_vec,act_vec)
            self.hd_cond_vec_state1 += sensor_vec
            self.num_cond_state1 += 1
        else:
            sample_vec = np.zeros((self.dim), dtype=np.int8)
            self.num_thrown_state1 += 1

        return sample_vec

    def train_sample_only_x(self, sensor_in, act_in):
        # Multiply encoded sensor vector with actuator vector
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        #   - act_in: integer representing actuator action
        # outputs:
        #   - sample_vec: bipolar HD vector

        sensor_vec = self.encoding_method(sensor_in,True)
        if self.new_condition(sensor_vec, self.threshold_known_only_x, self.hd_cond_vec_only_x):
            act_vec = self.hd_x_act_vals[act_in,:]
            #print(act_vec)
            sample_vec = self.hd_mul(sensor_vec,act_vec)
            self.hd_cond_vec_only_x += sensor_vec
            self.num_cond_only_x += 1
        else:
            sample_vec = np.zeros((self.dim), dtype=np.int8)
            self.num_thrown_only_x += 1

        return sample_vec

    #Model that considers only movement in y
    def train_sample_only_y(self, sensor_in, act_in):
        # Multiply encoded sensor vector with actuator vector
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        #   - act_in: integer representing actuator action
        # outputs:
        #   - sample_vec: bipolar HD vector

        sensor_vec = self.encoding_method(sensor_in,True)
        if self.new_condition(sensor_vec, self.threshold_known_only_y, self.hd_cond_vec_only_y):
            act_in = act_in - 2 #move index down to the two y vector possibilities
            act_vec = self.hd_y_act_vals[act_in,:]
            #print(act_vec)
            sample_vec = self.hd_mul(sensor_vec,act_vec)
            self.hd_cond_vec_only_y += sensor_vec
            self.num_cond_only_y += 1
        else:
            sample_vec = np.zeros((self.dim), dtype=np.int8)
            self.num_thrown_only_y += 1

        return sample_vec

    def test_sample(self, sensor_in):
        # Determine actuator action given sensory data
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - act_out: integer representing decided actuator action

        #for method 1, use encode_sensors
        #for method 2 and 3, use encoder_sensors_directional and change the last line of that function depending on the method
        #print("testing!\n")
        sensor_vec = self.encoding_method(sensor_in,False)
        #unbind_vec = self.hd_mul(sensor_vec,self.hd_threshold(self.hd_program_vec))
        unbind_vec = self.hd_mul(sensor_vec,self.hd_program_vec)
        if self.activation_function:
            act_out = self.softmax_actuator_vals(unbind_vec, self.softmax_param)
        else:
            act_out = self.search_actuator_vals(unbind_vec)

        return act_out

    def test_sample_xgoal(self, sensor_in):
        # Determine actuator action given sensory data
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - act_out: integer representing decided actuator action

        #for method 1, use encode_sensors
        #for method 2 and 3, use encoder_sensors_directional and change the last line of that function depending on the method
        #print("testing!\n")
        sensor_vec = self.encoding_method(sensor_in,False)
        #unbind_vec = self.hd_mul(sensor_vec,self.hd_threshold(self.hd_program_vec))
        unbind_vec = self.hd_mul(sensor_vec,self.hd_program_vec_goalx)
        if self.activation_function:
            act_out = self.softmax_actuator_vals(unbind_vec, self.softmax_param)
        else:
            act_out = self.search_actuator_vals(unbind_vec)

        return act_out


    def test_sample_ygoal(self, sensor_in):
        # Determine actuator action given sensory data
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - act_out: integer representing decided actuator action

        #for method 1, use encode_sensors
        #for method 2 and 3, use encoder_sensors_directional and change the last line of that function depending on the method
        #print("testing!\n")
        sensor_vec = self.encoding_method(sensor_in,False)
        #unbind_vec = self.hd_mul(sensor_vec,self.hd_threshold(self.hd_program_vec))
        unbind_vec = self.hd_mul(sensor_vec,self.hd_program_vec_goaly)
        if self.activation_function:
            act_out = self.softmax_actuator_vals(unbind_vec, self.softmax_param)
        else:
            act_out = self.search_actuator_vals(unbind_vec)

        return act_out

    def test_sample_state1(self, sensor_in):
        # Determine actuator action given sensory data
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - act_out: integer representing decided actuator action

        #for method 1, use encode_sensors
        #for method 2 and 3, use encoder_sensors_directional and change the last line of that function depending on the method
        #print("testing!\n")
        sensor_vec = self.encode_sensors_goal(sensor_in,True)
        #unbind_vec = self.hd_mul(sensor_vec,self.hd_threshold(self.hd_program_vec))
        unbind_vec = self.hd_mul(sensor_vec,self.hd_program_vec_state1)
        #if self.activation_function:
        #    act_out = self.softmax_actuator_vals(unbind_vec, self.softmax_param_state1)
        #else:
        act_out = self.search_actuator_vals(unbind_vec)

        return act_out

    def test_sample_state2(self, sensor_in):
        # Determine actuator action given sensory data
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - act_out: integer representing decided actuator action

        #for method 1, use encode_sensors
        #for method 2 and 3, use encoder_sensors_directional and change the last line of that function depending on the method
        #print("testing!\n")
        sensor_vec = self.encode_sensors_directional(sensor_in,True,3)
        #unbind_vec = self.hd_mul(sensor_vec,self.hd_threshold(self.hd_program_vec))
        unbind_vec = self.hd_mul(sensor_vec,self.hd_program_vec_state2)
        if self.activation_function:
            act_out = self.softmax_actuator_vals(unbind_vec, self.softmax_param_state2)
        else:
            act_out = self.search_actuator_vals(unbind_vec)

        return act_out

    def test_sample_only_x(self, sensor_in):
        # Determine actuator action given sensory data
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - act_out: integer representing decided actuator action

        #for method 1, use encode_sensors
        #for method 2 and 3, use encoder_sensors_directional and change the last line of that function depending on the method
        #print("testing!\n")
        sensor_vec = self.encoding_method(sensor_in,True)
        #unbind_vec = self.hd_mul(sensor_vec,self.hd_threshold(self.hd_program_vec))
        unbind_vec = self.hd_mul(sensor_vec,self.hd_program_vec_only_x)
        #if self.activation_function:
        #    act_out = self.softmax_actuator_vals(unbind_vec, self.softmax_param_state1)
        #else:
        act_out = self.search_x_actuator_vals(unbind_vec)

        return act_out

    def test_sample_only_y(self, sensor_in):
        # Determine actuator action given sensory data
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - act_out: integer representing decided actuator action

        #for method 1, use encode_sensors
        #for method 2 and 3, use encoder_sensors_directional and change the last line of that function depending on the method
        #print("testing!\n")
        sensor_vec = self.encoding_method(sensor_in,True)
        #unbind_vec = self.hd_mul(sensor_vec,self.hd_threshold(self.hd_program_vec))
        unbind_vec = self.hd_mul(sensor_vec,self.hd_program_vec_only_y)
        #if self.activation_function:
        #    act_out = self.softmax_actuator_vals(unbind_vec, self.softmax_param_state2)
        #else:
        act_out = self.search_y_actuator_vals(unbind_vec) + 2

        return act_out

    def train_from_file(self, file_in):
        # Build the program HV from a text file of recorded moves
        # inputs:
        #   -file_in: filename for the recorded moves
        # Currently, the hd_program_vec is not being thresholded
        print("training with ", file_in ,"\n")
        game_data = np.loadtxt(file_in, dtype = np.int8, delimiter=',')
        sensor_vals = game_data[:,:-1]
        actuator_vals = game_data[:,-1]
        n_samples = game_data.shape[0]

        state1_count = 0
        state2_count = 0
        only_x_count = 0
        only_y_count = 0

        # for i in range(len(sensor_vals)):
        #     delta_x = sensor_vals[i][4]
        #     delta_y = sensor_vals[i][5]
        #     sensor_vals[i][4] = math.atan2(delta_y, delta_x)
        #     sensor_vals[i][5] = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))


#        program_vec_b4thresh = np.zeros((self.dim,),dtype=np.int8)
#        for sample in range(n_samples):
#            sample_vec = self.train_sample(sensor_vals[sample,:],actuator_vals[sample])
#            program_vec_b4thresh = program_vec_b4thresh + sample_vec
#        if n_samples%2 == 0:
#            random_vec = np.squeeze(self.create_bipolar_mem(1,self.dim))
#            program_vec_b4thresh = program_vec_b4thresh + random_vec
#        self.hd_program_vec = self.hd_threshold(program_vec_b4thresh)
        if self.two_states:
            for sample in range(n_samples):
                # figure out which state we are in, and train into the correct program vector and condition vector list accordingly
                state = sensor_vals[sample,7]
                #state2_count += 1
                #sample_vec = self.train_sample_state2(sensor_vals[sample,:],actuator_vals[sample])
                #self.hd_program_vec_state2 = self.hd_program_vec_state2 + sample_vec
                if state == 1:
                    state1_count += 1
                    sample_vec = self.train_sample_state1(sensor_vals[sample,:],actuator_vals[sample])
                    self.hd_program_vec_state1 = self.hd_program_vec_state1 + sample_vec
                else:
                    state2_count += 1
                    sample_vec = self.train_sample_state2(sensor_vals[sample,:],actuator_vals[sample])
                    self.hd_program_vec_state2 = self.hd_program_vec_state2 + sample_vec
        elif self.stuck_id:
            for sample in range(n_samples):
                sample_vec = self.train_sample(sensor_vals[sample,:],actuator_vals[sample])
                #if any(sample_vec):
                #    self.output_vectors.append(sample_vec)
                #   self.output_actuators.append(actuator_vals[sample])
                self.hd_program_vec = self.hd_program_vec + sample_vec

            if self.stuck_id:
                for sample in range(n_samples):
                    sample_vecx = self.train_sample_goalx(sensor_vals[sample, :], actuator_vals[sample])
                    self.hd_program_vec_goalx = self.hd_program_vec_goalx + sample_vecx
                    sample_vecy = self.train_sample_goaly(sensor_vals[sample, :], actuator_vals[sample])
                    self.hd_program_vec_goaly = self.hd_program_vec_goaly + sample_vecy
        elif self.stuck_state_machine:
            for sample in range(n_samples):
                sample_vec = self.train_sample(sensor_vals[sample,:],actuator_vals[sample])
                self.hd_program_vec = self.hd_program_vec + sample_vec
                # if actuation is in the x direction then add to only_x program vector
                if (actuator_vals[sample] == 0) or (actuator_vals[sample] == 1):
                    sample_vecx = self.train_sample_only_x(sensor_vals[sample, :], actuator_vals[sample])
                    self.hd_program_vec_only_x = self.hd_program_vec_only_x + sample_vecx
                    only_x_count += 1
                else: #actuation is in the y direction, add to only_y program vector
                    sample_vecy = self.train_sample_only_y(sensor_vals[sample, :], actuator_vals[sample])
                    self.hd_program_vec_only_y = self.hd_program_vec_only_y + sample_vecy
                    only_y_count += 1
        else:
            for sample in range(n_samples):
                sample_vec = self.train_sample(sensor_vals[sample,:],actuator_vals[sample])
                self.hd_program_vec = self.hd_program_vec + sample_vec

        print("state1 moves\n",state1_count)
        print("state2 moves\n",state2_count)

        #pickle_vectors = open('output_vectors_original.pckl', 'wb')
        #pickle_actuators = open('output_actuators_original.pckl', 'wb')
        #pickle.dump(self.output_vectors, pickle_vectors)
        #pickle.dump(self.output_actuators, pickle_actuators)
        #pickle_vectors.close()
        #pickle_actuators.close()

        return

    def test_from_file(self, file_in, NN):
        # Prints out input sensor data and resulting output
        # inputs:
        #   -file_in: filename for the recorded moves
        # Currently, the hd_program_vec is not being thresholded
        game_data = np.loadtxt(file_in, dtype = np.int, delimiter=',')
        sensor_vals = game_data[:,:-1]
        actuator_vals = game_data[:,-1]
        n_samples = game_data.shape[0]
        correct = 0
        valid = 0

        for sample in range(n_samples):
            #print("sensor inputs: {}".format(game_data[sample]))
            if (NN):
                act_out = self.test_sample_NN(sensor_vals[sample,:])
            else:
                act_out = self.test_sample(sensor_vals[sample,:])
            #print("guessed output: {} \t correct output: {}".format(act_out, actuator_vals[sample]))
            if (act_out == actuator_vals[sample]):
                correct += 1
            if self.is_valid_move(sensor_vals[sample,:],act_out):
                valid += 1
        print("Accuracy: {}".format(correct/n_samples))
        print("Valid: {}".format(valid/n_samples))
        return

    def is_valid_move(self, sensor_in, move_in):
        if sensor_in[move_in]==1:
            valid = False
        else:
            valid = True

        return valid

    def loss(self, model, x, y, training, loss_object):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_ = model(x, training=training)
        return loss_object(y_true=y, y_pred=y_)

    def grad(self, model, inputs, targets, loss_object):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, inputs, targets, True, loss_object)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train_from_file_NN(self, file_in):
        # Build the program HV from a text file of recorded moves
        # inputs:
        #   -file_in: filename for the recorded moves
        # Currently, the hd_program_vec is not being thresholded
        game_data = np.loadtxt(file_in, dtype = 'int', delimiter=',')
        sensor_vals = game_data[:,:-2]
        actuator_vals = game_data[:,-1]
        n_samples = game_data.shape[0]

        # for i in range(len(sensor_vals)):
        #     delta_x = sensor_vals[i][4]
        #     delta_y = sensor_vals[i][5]
        #     sensor_vals[i][4] = math.atan2(delta_y, delta_x)
        #     sensor_vals[i][5] = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))

        X_train = []
        y_train = []
        for sample in range(n_samples):
            sample_vec, act = self.train_sample_NN(sensor_vals[sample,:],actuator_vals[sample])
            if any(sample_vec):
                X_train.append(sample_vec)
                y_train.append(act)

        # self.clf.fit(X_train, y_train)
        # self.clf.fit(sensor_vals, actuator_vals)

        features = tf.convert_to_tensor(X_train)
        labels = tf.convert_to_tensor(y_train)
        # features = tf.convert_to_tensor(sensor_vals)
        # labels = tf.convert_to_tensor(actuator_vals)

        # input_size = features.shape[1]
        # output_size = len(np.unique(labels))
        #
        # self.model = tf.keras.Sequential([
        #     tf.keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(input_size,)),
        #     tf.keras.layers.Dense(8, activation=tf.nn.relu),
        #     tf.keras.layers.Dense(output_size)
        # ])
        #
        # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        #
        # # Keep results for plotting
        # train_loss_results = []
        # train_accuracy_results = []
        #
        # num_epochs = 301
        #
        # for epoch in range(num_epochs):
        #     epoch_loss_avg = tf.keras.metrics.Mean()
        #     epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        #
        #     # Optimize the model
        #     loss_value, grads = self.grad(self.model, features, labels, loss_object)
        #     optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        #
        #     # Track progress
        #     epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        #     # Compare predicted label to actual label
        #     # training=True is needed only if there are layers with different
        #     # behavior during training versus inference (e.g. Dropout).
        #     epoch_accuracy.update_state(labels, self.model(features, training=True))
        #
        #     # End epoch
        #     train_loss_results.append(epoch_loss_avg.result())
        #     train_accuracy_results.append(epoch_accuracy.result())
        #
        #
        #     if epoch % 50 == 0:
        #         print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
        #                                                                 epoch_loss_avg.result(),
        #                                                                 epoch_accuracy.result()))


        X_train = []
        y_train = []
        for sample in range(n_samples):
            sample_vec, act = self.train_sample_NN(sensor_vals[sample,:],actuator_vals[sample])
            if any(sample_vec):
                X_train.append(sample_vec)
                y_train.append(self.hd_actuator_vals[act])

        pickle_vectors = open('train_vectors.pckl', 'wb')
        pickle_actuators = open('train_actuators.pckl', 'wb')
        pickle.dump(X_train, pickle_vectors)
        pickle.dump(y_train, pickle_actuators)
        pickle_vectors.close()
        pickle_actuators.close()

        features = tf.convert_to_tensor(X_train, dtype='float32')
        labels = tf.convert_to_tensor(y_train, dtype='float32')

        input_size = features.shape[1]
        output_size = labels.shape[1]

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(50, activation=tf.nn.tanh, input_shape=(input_size,)),
            tf.keras.layers.Dense(25, activation=tf.nn.tanh),
            tf.keras.layers.Dense(output_size, activation=tf.nn.tanh)
        ])

        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.cosine_similarity,
                      metrics=['accuracy'])

        self.model.fit(features, labels, epochs = 3)

        return

    def test_sample_NN(self, sensor_in):
        # lm = sensor_in[-1]
        # sensor_in = sensor_in[:-1]
        # sensor_in = np.append(sensor_in, [0,0,0,0])
        # sensor_in[6 + lm] = 1
        # print(sensor_in)
        sensor_in = self.encoding_method(sensor_in, False)
        # sensor_in = sensor_in[:50]
        # return self.clf.predict([sensor_in])[0]

        # sensor_in = np.array(sensor_in)
        # pred = self.model.predict(sensor_in[:, None].T)
        # probs = softmax(pred/np.max(pred) * self.softmax_param)
        # return np.random.choice(4, p = probs.flatten())

        pred = self.model.predict(np.array([sensor_in,]))
        dists = np.matmul(self.hd_actuator_vals, pred.T, dtype='float32')
        probs = softmax(dists/np.max(dists) * self.softmax_param)
        return np.random.choice(4, p = probs.flatten())

    def train_sample_NN(self, sensor_in, act_in):
        # Multiply encoded sensor vector with actuator vector
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        #   - act_in: integer representing actuator action
        # outputs:
        #   - sample_vec: bipolar HD vector

        # lm = sensor_in[-1]
        # sensor_in = sensor_in[:-1]
        # sensor_in = np.append(sensor_in, [0,0,0,0])
        # sensor_in[6 + lm] = 1

        #for method 1, use encode_sensors
        #for method 2 and 3, use encoder_sensors_directional and change the last line of that function depending on the method
        sensor_vec = self.encoding_method(sensor_in, True)
        # return (sensor_vec, act_in)
        if self.new_condition(sensor_vec, self.threshold_known, self.hd_cond_vec):
            self.num_cond += 1
            # return (sensor_in, act_in)
            return (sensor_vec, act_in)
            # return (sensor_vec[:50], act_in)
        else:
            sample_vec = np.zeros((len(sensor_in)), dtype=np.int8)
            self.num_thrown += 1
            return (sample_vec, act_in)





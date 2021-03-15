import pygame
import random
import numpy as np
import os
import csv
import time

from modules.hd_module import hd_module

class game_module:
    def __init__(self, outfile_name, num_obs):
        self.world_size = (10, 10)
        self.grid_size = (self.world_size[0] + 2, self.world_size[1] + 2)
        self.scale = 50
        self.pixel_dim = (self.grid_size[0] * self.scale, self.grid_size[1] * self.scale)

        # self.num_obs = 15
        self.num_obs = num_obs
        self.timeout = 100

        self.white = (255, 255, 255)
        self.blue = (0, 0, 225)
        self.green = (0, 255, 0)
        self.black = (0, 0, 0)

        self.pos = [0, 0]
        self.goal_pos = [0, 0]
        self.obs = []
        self.obs_mat = np.zeros(self.world_size)

        self.steps = 0
        self.average_steps = 0

        self.hd_module = hd_module()
        self.num_cond = self.hd_module.num_cond
        self.num_thrown = self.hd_module.num_thrown

        self.outdir = './data/'
        # self.outfile = self.outdir + 'game_data_random.out'
        self.outfile = self.outdir + outfile_name
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def setup_game(self, obs_id=None, goal_id=None):
        self.obs = []
        self.obs_mat = np.zeros(self.world_size)
        num_block = self.world_size[0] * self.world_size[1]
        if not obs_id:
            obs_idx = random.sample(list(range(num_block)), self.num_obs + 1)
        else:
            obs_idx = obs_id
        for i in range(self.num_obs):
            row_pos = obs_idx[i] // self.world_size[0]
            col_pos = obs_idx[i] % self.world_size[1]
            self.obs.append((row_pos, col_pos))
            self.obs_mat[row_pos, col_pos] = 1

        self.pos = [obs_idx[-1] // self.world_size[0], obs_idx[-1] % self.world_size[1]]
        if not goal_id:
            self.random_goal_location()
        else:
            self.goal_pos = goal_id
        self.steps = 0
        return

    def train_from_file(self, filename):
        self.hd_module.train_from_file(filename) #need to put state information into new training data file
        self.num_cond = self.hd_module.num_cond
        self.num_thrown = self.hd_module.num_thrown
        self.num_cond_state1 = self.hd_module.num_cond_state1
        self.num_thrown_state1 = self.hd_module.num_thrown_state1       
        self.num_cond_state2 = self.hd_module.num_cond_state2
        self.num_thrown_state2 = self.hd_module.num_thrown_state2

    def play_game(self, gametype):
        pygame.init()
        screen = pygame.display.set_mode(self.pixel_dim)

        f = open(self.outfile, 'w')

        running = True
        not_crash = True

        actuator = 0
        state1_count = 0
        state2_count = 0
        while running:
            self.setup_game()
            delta_x_buffer=[1]*4
            delta_y_buffer=[1]*4
            stuck_buffer=[0]*4
            while not_crash:
                self.game_step(gametype, screen)
                pygame.display.update()

                event = pygame.event.wait()
                if event.type == pygame.QUIT:
                    not_crash = False
                    running = False
                elif event.type == pygame.KEYDOWN:
                    current_sensor = self.get_sensor()
                    current_sensor.append(actuator)
                    print('Current sensor reading is ', current_sensor)
                    delta_x_buffer.pop(0)
                    delta_y_buffer.pop(0)
                    delta_x_buffer.append(current_sensor[4])
                    delta_y_buffer.append(current_sensor[5])

                    print('x buffer ', delta_x_buffer)
                    print('y buffer ', delta_y_buffer)

                    self.check_behaviour(delta_x_buffer,delta_y_buffer)

                    #add state information
                    current_sensor.append(self.check_state(self.pos[0],self.pos[1]))
                    #print(self.check_state(self.pos[0],self.pos[1]))
                    if event.key == pygame.K_LEFT:
                        self.pos[0] -= 1
                        actuator = 0
                    elif event.key == pygame.K_RIGHT:
                        self.pos[0] += 1
                        actuator = 1
                    elif event.key == pygame.K_UP:
                        self.pos[1] -= 1
                        actuator = 2
                    elif event.key == pygame.K_DOWN:
                        self.pos[1] += 1
                        actuator = 3
                    elif event.key == pygame.K_RETURN:
                        self.setup_game()


                    if (self.check_collision(self.pos[0], self.pos[1])):
                        not_crash = False
                    else:
# *********************** CHANGE BASED ON SENSOR DATA *************************
                        sensor_str = "{}, {}, {}, {}, {}, {}, {}, {}".format(*current_sensor)
                        f.write(sensor_str + ", " + str(actuator) + "\n")
# *****************************************************************************
                        if current_sensor[7]==1:
                            print("state 1 move was made\n")
                            state1_count += 1
                        else:
                            print("state 2 move was made\n")
                            state2_count += 1

                self.game_step(gametype, screen)
                pygame.display.update()

            event2 = pygame.event.wait()
            if event.type == pygame.QUIT:
                running = False
                print("state1 moves\n",state1_count)
                print("state2 moves\n",state2_count)
            elif event.type == pygame.KEYDOWN:
                not_crash = True


        pygame.display.quit()
        pygame.quit()
        f.close()
        return

    def set_sensor_weight(self,sensor_weight):
        self.hd_module.sensor_weight = sensor_weight
        return

    def set_threshold_known(self,threshold_known):
        self.hd_module.threshold_known = threshold_known
        return

    def set_threshold_known_state1(self,threshold_known):
        self.hd_module.threshold_known_state1 = threshold_known
        return

    def set_threshold_known_state2(self,threshold_known):
        self.hd_module.threshold_known_state2 = threshold_known
        return     

    def set_threshold_known_only_x(self,threshold_known):
        self.hd_module.threshold_known_only_x = threshold_known
        return

    def set_threshold_known_only_y(self,threshold_known):
        self.hd_module.threshold_known_only_y = threshold_known
        return 

    def set_softmax_param(self,softmax_param):
        self.hd_module.softmax_param = softmax_param
        return

    def set_softmax_param_state1(self,softmax_param):
        self.hd_module.softmax_param_state1 = softmax_param
        return

    def set_softmax_param_state2(self,softmax_param):
        self.hd_module.softmax_param_state2 = softmax_param
        return

    def autoplay_game(self, gametype):
        pygame.init()
        screen = pygame.display.set_mode(self.pixel_dim)
        clock = pygame.time.Clock()
        running = True
        not_crash = True

        last_act = 0
        state = 1
        switch_goal = 5
        while running:
            self.setup_game()
            self.steps = 0
            buffer_x_delta=[1]*4
            buffer_y_delta=[1]*4
            stuck_bufferx=[0]*4
            stuck_buffery=[0]*4
            while not_crash:
                x_stuck_alert = 0
                y_stuck_alert = 0
                self.game_step(gametype, screen)
                pygame.display.update()

                clock.tick(3)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        not_crash = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            self.setup_game()

                current_sensor = self.get_sensor()
                current_sensor.append(last_act)

                buffer_x_delta.pop(0)
                buffer_y_delta.pop(0)
                buffer_x_delta.append(current_sensor[4])
                buffer_y_delta.append(current_sensor[5])
                stuck_bufferx.pop(0)
                stuck_buffery.pop(0)

                is_stuck=self.check_behaviour(buffer_x_delta,buffer_y_delta)

                stuck_bufferx.append(is_stuck[0])
                stuck_buffery.append(is_stuck[1])

                if sum(stuck_bufferx) == len(stuck_bufferx):
                    print("!!!!!!!!")
                    print("STUCK ALERT in X")
                    print("!!!!!!!!")
                    x_stuck_alert = 1

                if sum(stuck_buffery) == len(stuck_buffery):
                    print("!!!!!!!!")
                    print("STUCK ALERT in Y")
                    print("!!!!!!!!")
                    y_stuck_alert = 1
                print(self.pos[0],self.pos[1])
                if self.hd_module.two_states: 
                    if (self.check_state(self.pos[0],self.pos[1]) == 1):
                        act_out = self.hd_module.test_sample_state1(current_sensor)
                    #elif (self.check_state(self.pos[0],self.pos[1])==2):
                    else:
                        act_out = self.hd_module.test_sample_state2(current_sensor)
                elif self.hd_module.stuck_id:
                    if x_stuck_alert:
                        act_out = self.hd_module.test_sample_ygoal(current_sensor)
                    elif y_stuck_alert:
                        act_out = self.hd_module.test_sample_xgoal(current_sensor)
                    else:
                        act_out = self.hd_module.test_sample(current_sensor)
                elif self.hd_module.stuck_state_machine:
                    # decide state for the next move based on current sensor information and state
                    xpos = self.pos[0]
                    ypos = self.pos[1]
                    obs_to_goalx = 0
                    obs_to_goaly = 0
                    right_xpos = xpos+1
                    right_ypos = ypos
                    left_xpos = xpos-1
                    left_ypos = ypos
                    up_xpos = xpos
                    down_xpos = xpos
                    up_ypos = ypos-1 #for some reason up means decrement
                    down_ypos = ypos + 1
                    delta_x = self.goal_pos[0] - self.pos[0] # if >0 then goal is to right, else left
                    delta_y = self.goal_pos[1] - self.pos[1] # if >0 then goal is down, else up

                    #check if there is an obstacle in the way to the goal for both the y and x directions
                    if delta_x > 0:
                        obs_to_goalx = self.check_collision(right_xpos, right_ypos)
                    else:
                        obs_to_goalx = self.check_collision(left_xpos,left_ypos)
                    if delta_y > 0:
                        obs_to_goaly = self.check_collision(down_xpos, down_ypos)
                    else:
                        obs_to_goalt = self.check_collision(up_xpos, up_ypos)

                    if state == 1:
                        if x_stuck_alert:
                            avoidance_steps = 1
                            switch_goal = 5
                            if self.check_collision(right_xpos, right_ypos):
                                state = 2 # go left
                            else:
                                state = 3 # go right
                        elif y_stuck_alert:
                            avoidance_steps = 1
                            switch_goal = 5
                            if self.check_collision(up_xpos,up_ypos):
                                state = 5 # go down
                            else:
                                state = 4 # go up
                        else:
                            state = 1 # keep using main method
                    elif state == 2:
                        if obs_to_goaly: # if still an obstacle in the way of the y goal direction
                            # if timed out going left or hit an obstacle, switch directions
                            if (avoidance_steps > switch_goal) or self.check_collision(left_xpos,left_ypos):
                                state = 3 # switch and go right
                                avoidance_steps = 1
                                switch_goal += 5
                            # else keep going left
                            else:
                                state = 2 # keep going left
                                avoidance_steps += 1
                        else:
                            state = 6 # only y direction recall
                    elif state == 3:
                        if obs_to_goaly: # if still an obstacle in the way of the y goal direction
                            # if timed out going left or hit an obstacle, switch directions
                            if (avoidance_steps > switch_goal) or self.check_collision(right_xpos,right_ypos):
                                state = 2 # switch and go right
                                avoidance_steps = 1
                                switch_goal += 5
                            # else keep going left
                            else:
                                state = 3 # keep going left
                                avoidance_steps += 1
                        else:
                            state = 6 # only y direction recall
                    elif state == 4:
                        if obs_to_goalx: # if still an obstacle in the way of the x goal direction
                            # if timed out going left or hit an obstacle, switch directions
                            if (avoidance_steps > switch_goal) or self.check_collision(up_xpos,up_ypos):
                                state = 5 # switch and go down
                                avoidance_steps = 1
                                switch_goal += 5
                            # else keep going up
                            else:
                                state = 4 # keep going up
                                avoidance_steps += 1
                        else:
                            state = 7 # only x direction recall
                    elif state == 5:
                        if obs_to_goalx: # if still an obstacle in the way of the x goal direction
                            # if timed out going left or hit an obstacle, switch directions
                            if (avoidance_steps > switch_goal) or self.check_collision(down_xpos,down_ypos):
                                state = 4 # switch and go up
                                avoidance_steps = 1
                                switch_goal += 5
                            # else keep going left
                            else:
                                state = 5 # keep going down
                                avoidance_steps += 1
                        else:
                            state = 7 # only x direction recall
                    elif state == 6:
                        #if hit an obstacle or reached goal in y coordinate, switch back to main mode
                        if self.check_collision(down_xpos,down_ypos) or self.check_collision(up_xpos,up_ypos):
                            state = 1
                        else:
                            state = 6
                    elif state == 7:
                        #if hit an obstacle or reached goal in y coordinate, switch back to main mode
                        if self.check_collision(left_xpos,left_ypos) or self.check_collision(right_xpos,right_ypos):
                            state = 1
                        else:
                            state = 7                 
                    else:
                        print("unknown state")
                        state = 1

                    print("state is: ",state)
                    # decide actuation depending on state
                    if state == 1:
                        act_out = self.hd_module.test_sample(current_sensor)
                    elif state == 2:
                        act_out = 0
                        print(act_out)
                        #print(xpos,ypos)
                    elif state == 3:
                        act_out = 1
                        print(act_out)
                        #print(xpos,ypos)
                    elif state == 4:
                        act_out = 2
                        print(act_out)
                        #print(xpos,ypos)
                    elif state == 5:
                        act_out = 3
                        print(act_out)
                        #print(xpos,ypos)
                    elif state == 6:
                        act_out = self.hd_module.test_sample_only_y(current_sensor)
                        print(xpos,ypos)
                        #print(act_out)
                    elif state == 7:
                        act_out = self.hd_module.test_sample_only_x(current_sensor)
                        #print(xpos,ypos)
                        print(act_out)
                else:
                    act_out = self.hd_module.test_sample(current_sensor)


                if act_out == 0:
                    self.pos[0] -= 1
                elif act_out == 1:
                    self.pos[0] += 1
                elif act_out == 2:
                    self.pos[1] -= 1
                elif act_out == 3:
                    self.pos[1] += 1

                last_act = act_out
                if (self.check_collision(self.pos[0], self.pos[1])):
                    not_crash = False
                    print(not_crash)
                if (self.steps >= self.timeout):
                    not_crash = False
                    print(not_crash)

                self.steps += 1

                self.game_step(gametype, screen)
                pygame.display.update()

            event2 = pygame.event.wait()
            if event2.type == pygame.QUIT:
                running = False
            elif event2.type == pygame.KEYDOWN:
                not_crash = True

        pygame.display.quit()
        pygame.quit()
        return


    def play_game_from_file(self, test_filename,goal_filename):

        test_reader=csv.reader(open(test_filename,'r'))
        goal_reader=csv.reader(open(goal_filename,'r'))

        last_act = 0

        success = 0
        crash = 0
        stuck = 0
        step_count=[]
        stuck_count=[]
        crash_count=[0]*100
        stuck_count=[0]*100
        x_stuck_count = 0
        y_stuck_count = 0
        crash_after_stuck = 0
        crash_after_xstuck = 0
        stuck_after_stuck = 0
        stuck_after_xstuck = 0
        total_stuck_count = 0
        i=0
        start=time.time()
        for env,goal in zip(test_reader,goal_reader):
            # print('\nNEW ENV')
            not_crash = True
            self.setup_game(obs_id=[int(ev) for ev in env],goal_id=[int(go) for go in goal])
            # self.setup_game(obs_id=[int(ev) for ev in env])
            # print('Initial position', self.pos)
            self.steps = 0
            #Buffers that store the current and previous 3 actions
            buffer_x_delta = [1] * 4
            buffer_y_delta = [1] * 4
            stuck_bufferx = [0] * 4
            stuck_buffery = [0] * 4
            flag=0
            state = 1
            switch_goal = 5
            trial_had_xstuck = 0
            trial_had_ystuck = 0

            while not_crash:
                x_stuck_alert = 0
                y_stuck_alert = 0
                if self.goal_pos == self.pos:
                    success += 1
                    break

                current_sensor = self.get_sensor()
                current_sensor.append(last_act)

                #Updating buffers... not very elegantly
                buffer_x_delta.pop(0)
                buffer_y_delta.pop(0)
                buffer_x_delta.append(current_sensor[4])
                buffer_y_delta.append(current_sensor[5])
                stuck_bufferx.pop(0)
                stuck_buffery.pop(0)

                #Check if it's stuck in the x or y direction
                is_stuck = self.check_behaviour(buffer_x_delta, buffer_y_delta)

                stuck_bufferx.append(is_stuck[0])
                stuck_buffery.append(is_stuck[1])

                #If it has been stuck for x (e.g.) 4 iterations, turn on these flags
                #I did it like this because 1) the buffer may take some iterations to populate
                #and 2) Sometimes it gets stuck for a quick moment but it gets out of it on its
                #own, we want to identify when it has the potential to time-out
                if sum(stuck_bufferx) == len(stuck_bufferx):
                    # print("!!!!!!!!")
                    # print("STUCK ALERT in X")
                    # print("!!!!!!!!")
                    x_stuck_alert = 1
                    trial_had_xstuck = 1
                    #x_stuck_count += 1
                    #print(x_stuck_count)

                if sum(stuck_buffery) == len(stuck_buffery):
                    # print("!!!!!!!!")
                    # print("STUCK ALERT in Y")
                    # print("!!!!!!!!")
                    y_stuck_alert = 1
                    trial_had_ystuck = 1
                    #y_stuck_count += 1
                    #print(y_stuck_count)

                #print(self.pos[0],self.pos[1])
                if self.hd_module.two_states: 
                    if (self.check_state(self.pos[0],self.pos[1]) == 1):
                        act_out = self.hd_module.test_sample_state1(current_sensor)
                    #elif (self.check_state(self.pos[0],self.pos[1])==2):
                    else:
                        act_out = self.hd_module.test_sample_state2(current_sensor)
                elif self.hd_module.stuck_id:
                    if x_stuck_alert:
                        act_out = self.hd_module.test_sample_ygoal(current_sensor)
                    elif y_stuck_alert:
                        act_out = self.hd_module.test_sample_xgoal(current_sensor)
                    else:
                        act_out = self.hd_module.test_sample(current_sensor)
                elif self.hd_module.stuck_state_machine:
                    # decide state for the next move based on current sensor information and state
                    xpos = self.pos[0]
                    ypos = self.pos[1]
                    obs_to_goalx = 0
                    obs_to_goaly = 0
                    right_xpos = xpos+1
                    right_ypos = ypos
                    left_xpos = xpos-1
                    left_ypos = ypos
                    up_xpos = xpos
                    down_xpos = xpos
                    up_ypos = ypos-1 #for some reason up means decrement
                    down_ypos = ypos + 1
                    delta_x = self.goal_pos[0] - self.pos[0] # if >0 then goal is to right, else left
                    delta_y = self.goal_pos[1] - self.pos[1] # if >0 then goal is down, else up

                    #check if there is an obstacle in the way to the goal for both the y and x directions
                    if delta_x > 0:
                        obs_to_goalx = self.check_collision(right_xpos, right_ypos)
                    else:
                        obs_to_goalx = self.check_collision(left_xpos,left_ypos)
                    if delta_y > 0:
                        obs_to_goaly = self.check_collision(down_xpos, down_ypos)
                    else:
                        obs_to_goalt = self.check_collision(up_xpos, up_ypos)

                    if state == 1:
                        if x_stuck_alert:
                            avoidance_steps = 1
                            switch_goal = 5
                            if self.check_collision(right_xpos, right_ypos):
                                state = 2 # go left
                            else:
                                state = 3 # go right
                        elif y_stuck_alert:
                            avoidance_steps = 1
                            switch_goal = 5
                            if self.check_collision(up_xpos,up_ypos):
                                state = 5 # go down
                            else:
                                state = 4 # go up
                        else:
                            state = 1 # keep using main method
                    elif state == 2:
                        if obs_to_goaly: # if still an obstacle in the way of the y goal direction
                            # if timed out going left or hit an obstacle, switch directions
                            if (avoidance_steps > switch_goal) or self.check_collision(left_xpos,left_ypos):
                                state = 3 # switch and go right
                                avoidance_steps = 1
                                switch_goal += 5
                            # else keep going left
                            else:
                                state = 2 # keep going left
                                avoidance_steps += 1
                        else:
                            state = 6 # only y direction recall
                    elif state == 3:
                        if obs_to_goaly: # if still an obstacle in the way of the y goal direction
                            # if timed out going left or hit an obstacle, switch directions
                            if (avoidance_steps > switch_goal) or self.check_collision(right_xpos,right_ypos):
                                state = 2 # switch and go right
                                avoidance_steps = 1
                                switch_goal += 5
                            # else keep going left
                            else:
                                state = 3 # keep going left
                                avoidance_steps += 1
                        else:
                            state = 6 # only y direction recall
                    elif state == 4:
                        if obs_to_goalx: # if still an obstacle in the way of the x goal direction
                            # if timed out going left or hit an obstacle, switch directions
                            if (avoidance_steps > switch_goal) or self.check_collision(up_xpos,up_ypos):
                                state = 5 # switch and go down
                                avoidance_steps = 1
                                switch_goal += 5
                            # else keep going up
                            else:
                                state = 4 # keep going up
                                avoidance_steps += 1
                        else:
                            state = 7 # only x direction recall
                    elif state == 5:
                        if obs_to_goalx: # if still an obstacle in the way of the x goal direction
                            # if timed out going left or hit an obstacle, switch directions
                            if (avoidance_steps > switch_goal) or self.check_collision(down_xpos,down_ypos):
                                state = 4 # switch and go up
                                avoidance_steps = 1
                                switch_goal += 5
                            # else keep going left
                            else:
                                state = 5 # keep going down
                                avoidance_steps += 1
                        else:
                            state = 7 # only x direction recall
                    elif state == 6:
                        #if hit an obstacle or reached goal in y coordinate, switch back to main mode
                        if self.check_collision(down_xpos,down_ypos) or self.check_collision(up_xpos,up_ypos):
                            state = 1
                        else:
                            state = 6
                    elif state == 7:
                        #if hit an obstacle or reached goal in y coordinate, switch back to main mode
                        if self.check_collision(left_xpos,left_ypos) or self.check_collision(right_xpos,right_ypos):
                            state = 1
                        else:
                            state = 7                 
                    else:
                        print("unknown state")
                        state = 1

                    #print("state is: ",state)
                    # decide actuation depending on state
                    if state == 1:
                        act_out = self.hd_module.test_sample(current_sensor)
                    elif state == 2:
                        act_out = 0
                        #print(act_out)
                        #print(xpos,ypos)
                    elif state == 3:
                        act_out = 1
                        #print(act_out)
                        #print(xpos,ypos)
                    elif state == 4:
                        act_out = 2
                        #print(act_out)
                        #print(xpos,ypos)
                    elif state == 5:
                        act_out = 3
                        #print(act_out)
                        #print(xpos,ypos)
                    elif state == 6:
                        act_out = self.hd_module.test_sample_only_y(current_sensor)
                        #print(xpos,ypos)
                        #print(act_out)
                    elif state == 7:
                        act_out = self.hd_module.test_sample_only_x(current_sensor)
                        #print(xpos,ypos)
                        #print(act_out)
                else:
                    act_out = self.hd_module.test_sample(current_sensor)


                if act_out == 0:
                    self.pos[0] -= 1
                elif act_out == 1:
                    self.pos[0] += 1
                elif act_out == 2:
                    self.pos[1] -= 1
                elif act_out == 3:
                    self.pos[1] += 1

                last_act = act_out
                if (self.check_collision(self.pos[0], self.pos[1])):
                    not_crash = False
                    crash += 1
                    # print('CRASH!')
                    flag=1
                    crash_count[i]=1
                    if trial_had_ystuck or trial_had_xstuck:
                        stuck_after_stuck += 1
                    #if trial_had_xstuck:
                    #    stuck_after_xstuck += 1

                elif (self.steps >= self.timeout):
                    not_crash = False
                    stuck += 1
                    # print('STUCK!')
                    flag=1
                    stuck_count[i]=1
                    if trial_had_ystuck or trial_had_xstuck:
                        crash_after_stuck += 1
                    #if trial_had_xstuck:
                    #    crash_after_xstuck += 1                    

                self.steps += 1
            i+=1
            x_stuck_count += trial_had_xstuck
            y_stuck_count += trial_had_ystuck
            total_stuck_count += (trial_had_xstuck or trial_had_ystuck)
            # print('It took ', self.steps, ' steps')
            if flag==1:
                step_count.append(self.timeout)
            elif flag==0:
                step_count.append(self.steps)

        end=time.time()


        print("success: {} \t crash: {} \t stuck: {} \t time: {} \t average steps: {} \t total x_stuck: {} \t total y_stuck {}".format(success, crash, stuck,end-start,np.mean(np.array(step_count)),x_stuck_count,y_stuck_count))
        print("success rate: {:.2f}".format(success/(success+crash+stuck)))
        print('step count ', len(step_count), len(crash_count),len(stuck_count))
        return success,crash,stuck,step_count, crash_count, stuck_count, x_stuck_count, y_stuck_count, stuck_after_stuck, crash_after_stuck, total_stuck_count


    def test_game(self, num_test):
        not_crash = True

        last_act = 0

        success = 0
        crash = 0
        stuck = 0
        self.average_steps = 0
        success_times=[]

        for i in range(num_test):
            not_crash = True
            self.setup_game()
            self.steps = 0
            buffer_x_delta=[1]*4
            buffer_y_delta=[1]*4
            stuck_bufferx=[0]*4
            stuck_buffery=[0]*4


            begin_time=time.time()
            # trying to push a state machine into python oops
            # anyway state 1 is main encoding method
            # state 2 is move left
            # state 3 is move right
            # state 4 is move up
            # state 5 is move down
            # state 6 is only_y recall
            # state 7 is only_x recall
            state = 1
            switch_goal = 5
            while not_crash:
                x_stuck_alert = 0
                y_stuck_alert = 0
                if self.goal_pos == self.pos:
                    self.random_goal_location()
                    success += 1
                    end_time=time.time()
                    time_total=end_time-begin_time
                    success_times.append(time_total)
                    break

                current_sensor = self.get_sensor()
                current_sensor.append(last_act)

                buffer_x_delta.pop(0)
                buffer_y_delta.pop(0)
                buffer_x_delta.append(current_sensor[4])
                buffer_y_delta.append(current_sensor[5])
                stuck_bufferx.pop(0)
                stuck_buffery.pop(0)

                is_stuck=self.check_behaviour(buffer_x_delta,buffer_y_delta)
                stuck_bufferx.append(is_stuck[0])
                stuck_buffery.append(is_stuck[1])

                if sum(stuck_bufferx)==len(stuck_bufferx):
                    # print("!!!!!!!!")
                    # print("STUCK ALERT in X")
                    # print("!!!!!!!!")
                    x_stuck_alert=1

                if sum(stuck_buffery)==len(stuck_buffery):
                    # print("!!!!!!!!")
                    # print("STUCK ALERT in Y")
                    # print("!!!!!!!!")
                    y_stuck_alert=1

                #print(self.pos[0],self.pos[1])
                if self.hd_module.two_states: 
                    if (self.check_state(self.pos[0],self.pos[1]) == 1):
                        act_out = self.hd_module.test_sample_state1(current_sensor)
                    #elif (self.check_state(self.pos[0],self.pos[1])==2):
                    else:
                        act_out = self.hd_module.test_sample_state2(current_sensor)
                elif self.hd_module.stuck_id:
                    if x_stuck_alert:
                        act_out = self.hd_module.test_sample_ygoal(current_sensor)
                    elif y_stuck_alert:
                        act_out = self.hd_module.test_sample_xgoal(current_sensor)
                    else:
                        act_out = self.hd_module.test_sample(current_sensor)
                elif self.hd_module.stuck_state_machine:
                    # decide state for the next move based on current sensor information and state
                    xpos = self.pos[0]
                    ypos = self.pos[1]
                    obs_to_goalx = 0
                    obs_to_goaly = 0
                    right_xpos = xpos+1
                    right_ypos = ypos
                    left_xpos = xpos-1
                    left_ypos = ypos
                    up_xpos = xpos
                    down_xpos = xpos
                    up_ypos = ypos-1 #for some reason up means decrement
                    down_ypos = ypos + 1
                    delta_x = self.goal_pos[0] - self.pos[0] # if >0 then goal is to right, else left
                    delta_y = self.goal_pos[1] - self.pos[1] # if >0 then goal is down, else up

                    #check if there is an obstacle in the way to the goal for both the y and x directions
                    if delta_x > 0:
                        obs_to_goalx = self.check_collision(right_xpos, right_ypos)
                    else:
                        obs_to_goalx = self.check_collision(left_xpos,left_ypos)
                    if delta_y > 0:
                        obs_to_goaly = self.check_collision(down_xpos, down_ypos)
                    else:
                        obs_to_goalt = self.check_collision(up_xpos, up_ypos)

                    if state == 1:
                        if x_stuck_alert:
                            avoidance_steps = 1
                            switch_goal = 5
                            if self.check_collision(right_xpos, right_ypos):
                                state = 2 # go left
                            else:
                                state = 3 # go right
                        elif y_stuck_alert:
                            avoidance_steps = 1
                            switch_goal = 5
                            if self.check_collision(up_xpos,up_ypos):
                                state = 5 # go down
                            else:
                                state = 4 # go up
                        else:
                            state = 1 # keep using main method
                    elif state == 2:
                        if obs_to_goaly: # if still an obstacle in the way of the y goal direction
                            # if timed out going left or hit an obstacle, switch directions
                            if (avoidance_steps > switch_goal) or self.check_collision(left_xpos,left_ypos):
                                state = 3 # switch and go right
                                avoidance_steps = 1
                                switch_goal += 5
                            # else keep going left
                            else:
                                state = 2 # keep going left
                                avoidance_steps += 1
                        else:
                            state = 6 # only y direction recall
                    elif state == 3:
                        if obs_to_goaly: # if still an obstacle in the way of the y goal direction
                            # if timed out going left or hit an obstacle, switch directions
                            if (avoidance_steps > switch_goal) or self.check_collision(right_xpos,right_ypos):
                                state = 2 # switch and go right
                                avoidance_steps = 1
                                switch_goal += 5
                            # else keep going left
                            else:
                                state = 3 # keep going left
                                avoidance_steps += 1
                        else:
                            state = 6 # only y direction recall
                    elif state == 4:
                        if obs_to_goalx: # if still an obstacle in the way of the x goal direction
                            # if timed out going left or hit an obstacle, switch directions
                            if (avoidance_steps > switch_goal) or self.check_collision(up_xpos,up_ypos):
                                state = 5 # switch and go down
                                avoidance_steps = 1
                                switch_goal += 5
                            # else keep going up
                            else:
                                state = 4 # keep going up
                                avoidance_steps += 1
                        else:
                            state = 7 # only x direction recall
                    elif state == 5:
                        if obs_to_goalx: # if still an obstacle in the way of the x goal direction
                            # if timed out going left or hit an obstacle, switch directions
                            if (avoidance_steps > switch_goal) or self.check_collision(down_xpos,down_ypos):
                                state = 4 # switch and go up
                                avoidance_steps = 1
                                switch_goal += 5
                            # else keep going left
                            else:
                                state = 5 # keep going down
                                avoidance_steps += 1
                        else:
                            state = 7 # only x direction recall
                    elif state == 6:
                        #if hit an obstacle or reached goal in y coordinate, switch back to main mode
                        if self.check_collision(down_xpos,down_ypos) or self.check_collision(up_xpos,up_ypos):
                            state = 1
                        else:
                            state = 6
                    elif state == 7:
                        #if hit an obstacle or reached goal in y coordinate, switch back to main mode
                        if self.check_collision(left_xpos,left_ypos) or self.check_collision(right_xpos,right_ypos):
                            state = 1
                        else:
                            state = 7                 
                    else:
                        print("unknown state")
                        state = 1

                    #print("state is: ",state)
                    # decide actuation depending on state
                    if state == 1:
                        act_out = self.hd_module.test_sample(current_sensor)
                    elif state == 2:
                        act_out = 0
                        #print(act_out)
                        #print(xpos,ypos)
                    elif state == 3:
                        act_out = 1
                        #print(act_out)
                        #print(xpos,ypos)
                    elif state == 4:
                        act_out = 2
                        #print(act_out)
                        #print(xpos,ypos)
                    elif state == 5:
                        act_out = 3
                        #print(act_out)
                        #print(xpos,ypos)
                    elif state == 6:
                        act_out = self.hd_module.test_sample_only_y(current_sensor)
                        #print(xpos,ypos)
                        #print(act_out)
                    elif state == 7:
                        act_out = self.hd_module.test_sample_only_x(current_sensor)
                        #print(xpos,ypos)
                        #print(act_out)
                else:
                    act_out = self.hd_module.test_sample(current_sensor)

                if act_out == 0:
                    self.pos[0] -= 1
                elif act_out == 1:
                    self.pos[0] += 1
                elif act_out == 2:
                    self.pos[1] -= 1
                elif act_out == 3:
                    self.pos[1] += 1

                last_act = act_out
                if (self.check_collision(self.pos[0], self.pos[1])):
                    not_crash = False
                    crash += 1
                elif (self.steps >= self.timeout):
                    not_crash = False
                    stuck += 1

                self.steps += 1
            self.average_steps += self.steps

        self.average_steps = self.average_steps / num_test
        print("success: {} \t crash: {} \t stuck: {}".format(success, crash, stuck))
        print("success rate: {:.2f}".format(success/(success+crash+stuck)))

        return success,crash,stuck,self.average_steps
        #return success,crash,stuck,np.mean(success_times)


    def game_step(self, gametype, screen):
        screen.fill(self.white)
        self.draw_walls(screen)
        self.draw_obstacles(screen)
        self.draw_me(screen)
        if (gametype):
            if self.goal_pos == self.pos:
                self.random_goal_location()
                self.steps = 0
            self.draw_goal(screen)
        return

    def draw_me(self, screen):
        xpixel = (self.pos[0]+1)*self.scale
        ypixel = (self.pos[1]+1)*self.scale
        pygame.draw.rect(screen, self.blue, [xpixel,ypixel,self.scale,self.scale])
        return


    def draw_obstacles(self, screen):
        for pos in self.obs:
            xpos = (pos[0]+1)*self.scale
            ypos = (pos[1]+1)*self.scale
            pygame.draw.rect(screen, self.black, [xpos,ypos,self.scale,self.scale])
        return

    def draw_goal(self, screen):
        xpixel = (self.goal_pos[0]+1)*self.scale
        ypixel = (self.goal_pos[1]+1)*self.scale
        pygame.draw.rect(screen, self.green, [xpixel,ypixel,self.scale,self.scale])
        return

    def draw_walls(self, screen):
        pygame.draw.rect(screen, self.black, [0,0,self.pixel_dim[0]-self.scale,self.scale])
        pygame.draw.rect(screen, self.black, [self.pixel_dim[0]-self.scale,0,self.scale,self.pixel_dim[1]-self.scale])
        pygame.draw.rect(screen, self.black, [self.scale,self.pixel_dim[1]-self.scale,self.pixel_dim[0]-self.scale,self.scale])
        pygame.draw.rect(screen, self.black, [0,self.scale,self.scale,self.pixel_dim[1]-self.scale])
        return

    def pos_oob(self, xpos, ypos):
        # Check if (xpos,ypos) is out of bounds
        oob = 0
        if (xpos < 0 or xpos >= self.world_size[0]):
            oob = 1
        if (ypos < 0 or ypos >= self.world_size[1]):
            oob = 1
        return oob

    def check_collision(self, xpos, ypos):
        # Check if (xpos,ypos) is out of bounds or occupied by object
        collision = 0
        if (self.pos_oob(xpos, ypos)):
            collision = 1
            #print(collision)
        else:
            if (self.obs_mat[xpos, ypos]):
                collision = 1
                #print(collision)
        return collision

    def check_state(self,xpos,ypos):
        # nothing around means state one, goal oriented behavior
        anything_around = 1
        right_xpos = xpos+1
        right_ypos = ypos
        left_xpos = xpos-1
        left_ypos = ypos
        up_xpos = xpos
        down_xpos = xpos
        up_ypos = ypos-1 #for some reason up means decrement
        down_ypos = ypos + 1
        #is there anything (edge or obstacle) to the left or right
        if (self.pos_oob(right_xpos, right_ypos) or self.obs_mat[right_xpos, right_ypos] or self.pos_oob(left_xpos, left_ypos) or self.obs_mat[left_xpos, left_ypos]):
            anything_around = 2
        else:
            #is there anything (edge or obstacle) above or below
            if (self.pos_oob(up_xpos, up_ypos) or self.obs_mat[up_xpos, up_ypos] or self.pos_oob(down_xpos, down_ypos) or self.obs_mat[down_xpos, down_ypos]):
                anything_around = 2
        #if there is anything in any of the four directions, state is 2 and need to trigger obstacle avoidance
        return anything_around

    #Checks if agent is "stuck" going back and forth in one direction while not progressing in the other
    def check_behaviour(self, buffer_delta_x, buffer_delta_y):
        stuck_x = 0
        stuck_y = 0
        #The buffers record the last 3 and the current delta in direction
        #A sum equal to 0 means that the agent went back and forth over 4 time steps
        sum_buffer_x = sum(buffer_delta_x)
        sum_buffer_y = sum(buffer_delta_y)
        #This checks if there was no change in delta for both directions
        unique_buffer_x = list(set(buffer_delta_x))
        unique_buffer_y = list(set(buffer_delta_y))
        # If it has been going back and forth in the x direction
        if sum_buffer_x == 0 or len(unique_buffer_x)==2:
            # But made no progress in the y direction
            if len(unique_buffer_y) == 1:
                stuck_x = 1
        if sum_buffer_y == 0 or len(unique_buffer_y)==2 :
            if len(unique_buffer_x) == 1:
                stuck_y = 1
        return stuck_x,stuck_y


    def random_goal_location(self):
        # Choose random unoccupied square for the goal position
        num_block = self.world_size[0]*self.world_size[1]
        goal_idx = random.randrange(num_block)
        row_pos = goal_idx//self.world_size[0]
        col_pos = goal_idx%self.world_size[1]
        while (self.check_collision(row_pos,col_pos)):
            goal_idx = random.randrange(num_block)
            row_pos = goal_idx//self.world_size[0]
            col_pos = goal_idx%self.world_size[1]
        self.goal_pos = [row_pos, col_pos]
        return


# *********************** CHANGE BASED ON SENSOR DATA *************************
    def get_sensor(self):
        # list of coordinates for squares around current position
        sensor_pos = [(self.pos[0]-1, self.pos[1]),
                (self.pos[0]+1, self.pos[1]),
                (self.pos[0], self.pos[1]-1),
                (self.pos[0], self.pos[1]+1)]
        sensor_vals = [self.check_collision(xpos,ypos) for (xpos,ypos) in sensor_pos]
        delta_x = self.goal_pos[0] - self.pos[0]
        delta_y = self.goal_pos[1] - self.pos[1]
        sensor_vals.extend([delta_x, delta_y])
        return sensor_vals
# *****************************************************************************

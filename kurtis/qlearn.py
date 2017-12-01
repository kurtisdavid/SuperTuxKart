import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from pykart import Kart
import time
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt

#actions = [0,1,2,33,34]
input_state_vars = ["position_along_track", "distance_to_center", "speed", "angle", "smooth_speed", "wrongway"]

class AI:
    def __init__(self):
        self.prev_dist = -10
        self.prev_pos = -10
        self.angle = 0

    def get_action(self,state):
        action = 4
        y = (state['distance_to_center'] - self.prev_dist)
        x = state['position_along_track'] - self.prev_pos
        if self.prev_pos != -10 and (x != 0 or y != 0):
            self.angle = np.arctan2(y,x)
            #print(y, ' ', x, ' ', angle)
        if (state['distance_to_center'] < 0) and self.angle <= 0.05:
            action += 2
        if (state['distance_to_center'] > 0) and self.angle >= -0.05:
            action += 1
        if abs(self.angle) > 1.5 and (state['distance_to_center']) > 3:
            action += 32
        self.prev_pos = state['position_along_track']
        self.prev_dist = state['distance_to_center']
        return action

K = Kart("lighthouse", 300, 200)
player = AI()
K.restart()
K.waitRunning()

MEM_SIZE = 10000
prev_obs_memory = []
obs_memory = []
prev_a_r = []
prev_state_memory = []
state_memory = []


prev_state = None
prev_action = None
prev_obs = None
state = None
fake_state = None
obs = None
real_action = None
start = None
action = 4

last_time = 0
last_obs = None
while len(prev_a_r) < MEM_SIZE:

    step = K.step( action )
    if step is not None and (step[0]['position_along_track']>=3):
            K.quit()
            K = Kart("lighthouse", 300, 200)
            player = AI()
            K.restart()
            K.waitRunning()
            prev_state = None
            prev_action = None
            prev_obs = None
            state = None
            fake_state = None
            obs = None
            real_action = None
            start = None
            action = 4
            print("RESTART")
            continue


    if (step and time.time()-last_time > 1/5) or (step and state is None):

        print(len(prev_a_r))
        # save previous values
        if state is not None and obs is not None:
            prev_state = state
            prev_action = real_action
            prev_obs = np.copy(obs)

        if step[1] is None:
            continue

        state = step[0]

        if (state['position_along_track']<0 and obs is None): # only at the beginning
            continue
        obs = np.copy(step[1]) # annoying things with references
        
        state['position_along_track'] = state['position_along_track']%1
        real_action = player.get_action(state)
        action = real_action
        last_time = time.time()

        if prev_state is not None and prev_obs is not None:
            # plt.figure(1)
            # plt.imshow(prev_obs)
            # plt.figure(2)
            # plt.imshow(obs)
            # plt.show()
            #reward = state['position_along_track'] - prev_state['position_along_track']

            # MUST KEEP THIS SCALED
            reward = 100*(abs(state['position_along_track']) - abs(prev_state['position_along_track'])) - 1*abs(state['wrongway']) - .001 * abs(state['distance_to_center'])
            
            prev_obs_memory.append(np.copy(prev_obs)) # annoying again
            obs_memory.append(np.copy(obs))
            prev_a_r.append([prev_action,reward])
            prev_state_memory.append([prev_state[s] for s in input_state_vars])
            state_memory.append([state[s] for s in input_state_vars])
            
    elif step:
        fake_state, _ = step
        fake_state['position_along_track'] = fake_state['position_along_track'] % 1 # only care about distance in race?
        action = player.get_action(fake_state)

    
    
# save memory
prev_obs_memory = np.asarray(prev_obs_memory)
np.save("prev_obs_memory",prev_obs_memory)
prev_a_r = np.asarray(prev_a_r)
np.save("prev_a_r",prev_a_r)
obs_memory = np.asarray(obs_memory)
np.save("obs_memory",obs_memory)
prev_state_memory = np.asarray(prev_state_memory)
np.save("prev_state_memory",prev_state_memory)
state_memory = np.asarray(state_memory)
np.save("state_memory",state_memory)

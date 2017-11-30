import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from pykart import Kart
import time
import numpy as np
import pickle
import random

#actions = [0,1,2,33,34]
input_state_vars = ["position_along_track", "distance_to_center", "speed", "angle", "smooth_speed", "wrongway"]

class AI:
    def __init__(self):
        self.prev_dist = -10
        self.prev_pos = -10
        self.angle = 0

    def get_action(self,state):
        action = 4
        y = state['distance_to_center'] - self.prev_dist
        x = state['position_along_track'] - self.prev_pos
        if self.prev_pos != -10 and (x != 0 or y != 0):
            self.angle = np.arctan2(y,x)
            state['angle'] = self.angle
            #print(y, ' ', x, ' ', angle)
        if (state['distance_to_center'] < 0) and self.angle <= 0.05:
            action += 2
        if (state['distance_to_center'] > 0) and self.angle >= -0.05:
            action += 1
        if abs(self.angle) > 1 and (state['distance_to_center']) > 2:
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
obs = None
start = None
action = 4

last_time = 0

while len(prev_a_r) < MEM_SIZE:
    step = K.step( action )
    if step:
        #save previous values for use in dataset
        if state is not None:
            prev_state = state
            prev_action = action
            prev_obs = obs

        state, obs = step
        state['position_along_track'] = state['position_along_track'] % 1 # only care about distance in race?
        action = player.get_action(state)

    if step and time.time()-last_time > 1/750:
        last_time = time.time()
        print(len(prev_obs_memory))
        if prev_state is not None and prev_obs is not None:
            reward = state['position_along_track'] - prev_state['position_along_track']
            prev_obs_memory.append(prev_obs)
            obs_memory.append(obs)
            prev_a_r.append([prev_action,reward])
            prev_state_memory.append([prev_state[s] for s in input_state_vars])
            state_memory.append([state[s] for s in input_state_vars])
            #memory.append((prev_obs, prev_action, reward, obs))
            #print(state)
    

#with open("init_memory.pkl","wb") as f:
    #pickle.dump(memory,f)

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

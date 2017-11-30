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

class AI:
    def __init__(self):
        pass

    def get_action(self,state):
        action = 4 | 16 | 128
        if (abs(state['angle'])<1 and state['distance_to_center']<-5):
            action += 2
        elif ((state['distance_to_center'] < 0 or abs(state['distance_to_center'])<1.5) and state['angle'] < 0):
            action += 2
        elif (state['angle']>0.5):
            action += 2
        if (abs(state['angle'])<1 and state['distance_to_center']>5):
            action +=  1
        elif ((state['distance_to_center'] > 0 or abs(state['distance_to_center'])<1.5) and state['angle'] > 0):
            action += 1
        elif (state['angle']<-0.5):
            action += 1
        
        #if(abs(state['angle']) > 0.5):
         #   action += 32
        #print(action)
        return action

K = Kart("lighthouse", 300, 200)
player = AI()
K.restart()
K.waitRunning()

MEM_SIZE = 10000
prev_obs_memory = []
obs_memory = []
prev_a_r = []


prev_state = None
prev_action = None
prev_obs = None
state = None
obs = None
start = None
action = 4
i = False

last_time = 0

while len(prev_a_r) < MEM_SIZE:
    step = K.step( action )
    i = not i
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
        if prev_state is not None:
            reward = state['position_along_track'] - prev_state['position_along_track']
            prev_obs_memory.append(prev_obs)
            obs_memory.append(obs)
            prev_a_r.append([prev_action,reward])
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
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from pykart import Kart
import tensorflow as tf
from collections import deque
import time
import numpy as np
import random
import util

# actions = [4,1,2]
# input_state_vars = ["position_along_track", "distance_to_center", "speed", "angle", "smooth_speed"]
# state_std = np.load('./lstm_data_10_safe/state_std.npy')
# state_mean = np.load('./lstm_data_10_safe/state_mean.npy')
actions = [4,5,6,36,33,34]
input_state_vars = ["position_along_track", "distance_to_center", "speed", "angle", "smooth_speed"]
state_std = np.load('./lstm_data/state_std.npy')
state_mean = np.load('./lstm_data/state_mean.npy')
# based on our graders
class DeepAI:

    def __init__(self,graph,sess,actions,seq=0):
        self.g = graph
        self.sess = sess
        self.actions = actions
        self.input_state = self.get_tensor('state_input')
        self.action = self.get_tensor('action')
        #self.history = np.zeros((seq,len(self.actions)))
        self.current_index = 0


    def get_tensor(self, name):
        if not ':' in name: name = name + ':0'
        try:
            return self.g.get_tensor_by_name(name)

        except KeyError:
            return None

    # def __call__(self,state):
    #     decision = self.sess.run(self.action,{self.input_state:state})[0]
    #     print(self.history)
    #     print(decision)
    #     print(self.current_index)
    #     for i in range(self.history.shape[0]):
    #         self.history[(i+self.current_index)%self.history.shape[0]][decision[i]] += 1/((i+1))
    #         print((i+self.current_index)%self.history.shape[0],self.history[(i+self.current_index)%self.history.shape[0]][decision[i]])
    #     correct = np.argmax(self.history[(i+self.current_index)%self.history.shape[0]])
    #     self.history[(self.current_index)%6] = np.zeros(len(self.actions))
    #     print(self.history)
    #     print("~~~~~~~~~")
    #     self.current_index = (self.current_index + 1)%self.history.shape[0]
    #     return self.actions[correct]

    def __call__(self,state):
        return self.actions[self.sess.run(self.action,{self.input_state:state})[0][5]]



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
graph = util.load('kart_lstm_0.965.tfg',session=sess)

times = []
action_times = []
with graph.as_default():
    time_steps = 6
    K = Kart("lighthouse", 900, 600)
    #input("Ready?") # for recording
    player = DeepAI(graph,sess,actions)
    K.restart()
    K.waitRunning()

    last_states = deque()
    action = 4
    last_time = 0
    finished = False
    n = 0
    i = 0
    idle_count = 0
    while True and not finished:
        
        
        step = K.step(action)
        print(action)
        if step is not None and (len(times)==0 or times[len(times)-1]!=step[0]['timestamp']):
            times.append(step[0]['timestamp'])
            action_times.append(action)

        #print(action)
        if step is not None and time.time()-last_time>1/30:
            last_time = time.time()
            if (step[0]['finish_time']>0):
                finished = True
            #print(step[0]['position_along_track'])
            # hard code rescuing just in case
            if step[0]['speed']<0.01 and step[0]['smooth_speed']<0.01 and abs(step[0]['distance_to_center'])>5:
                idle_count += 1
                if idle_count % 60 == 0: # 2 second idle
                    start_rescue = time.time()
                    step = K.step(64)
                    last_states.clear()
                    n = 0
                    player.history = np.zeros((time_steps,len(actions)))
                    times.append(step[0]['timestamp'])
                    action_times.append(action)
                    while time.time()-start_rescue < 2:
                        pass
                    step = K.step(0)

            curr_state = np.asarray([step[0][s] for s in input_state_vars])
            curr_state[0] %= 1 # we trained on percent lab
            if curr_state[0]<0:
                continue
            curr_state = (curr_state - state_mean)/state_std
            
            if n<time_steps:
                last_states.append(curr_state)
                n+=1
            else:
                last_states.popleft()
                last_states.append(curr_state)
                action = player([list(last_states)])
                if i % 60 == 0:
                    action |= 128
                i += 1
                if step[0]['position_along_track']%1>0.96 or step[0]['position_along_track']%1<0.04:
                    action |= 16


    with open('actions_new_6.txt','w') as f:
        for i in range(len(times)):
            f.write(str(action_times[i]) + ':' + str(times[i])+'\n')









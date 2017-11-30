import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from pykart import Kart
from time import sleep
import numpy as np

class AI:
    def __init__(self):

    def get_action(state):
        action = 4 | 16
        if state['distance_to_center'] < -0.1 and state['angle'] < 0.0:
            action += 2
        if state['distance_to_center'] > -0.01 and state['angle'] > -0.0:
            action += 1
        if(state['wrongway'] > 0):
            action += 2
        if(abs(state['angle']) > 0.5):
            action += 32
        return action

K = Kart("lighthouse", 300, 200)
player = AI(K)
K.restart()
K.waitRunning()

MEM_SIZE = 10000
memory = []


prev_state = None
prev_action = None
prev_position = None
state = None
start = None
action = 4
i = False

while len(memory) < MEM_SIZE:
    step = K.step( action )
    i = not i
    if(step):
        #save previous values for use in dataset
        if state is not None:
            prev_state = state
            prev_action = action
            prev_position  = prev_state['position_along_track']

        state, obs = step
        if prev_state is not None:
            reward = state['position_along_track'] - prev_state['position_along_track']
            memory.append((prev_state, prev_action, reward, state))
        action = player.get_action(state)
        if i:
            action = action | 128

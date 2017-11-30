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

start = None
action = 4
i = False

while True:
    step = K.step( action )
    i = not i
    if(step):
        state, obs = step
        action = player.get_action(state)
        if i:
            action = act

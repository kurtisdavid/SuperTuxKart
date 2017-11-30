import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from pykart import Kart
from time import sleep
import numpy as np

K = Kart("lighthouse", 300, 200)
K.restart()
K.waitRunning()

start = None
action = 4
best = -10
idle = 0
i = False
while True:
    step = K.step( action )
    i = not i
    if step:
        state, obs = step
        if(start is None):
            start = state['timestamp']
        #print("angle = " + str(state['angle']))
        #print("distance = " + str(state['distance_to_center']))
        print("Idle = " + str(idle))
        action = 4 | 16
        if i:
            action = action | 128
        if state['distance_to_center'] < -0.1 and state['angle'] < 0.0:
            action += 2
        if state['distance_to_center'] > -0.01 and state['angle'] > -0.0:
            action += 1
        if(state['wrongway'] > 0):
            action += 2
        if(abs(state['angle']) > 0.5):
            action += 32
        if state['position_along_track'] > best:
            best = state['position_along_track']   

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

action = 4
best = -10
idle = 0
i = False
angle = 0
prev_pos = -10
prev_dist = -10
while True:
    step = K.step( action )
    i = not i
    if step:
        state, obs = step
        action = 4 + 16
        y = state['distance_to_center'] - prev_dist
        x = state['position_along_track'] - prev_pos
        if prev_pos != -10 and (x != 0 or y != 0):
            angle = np.arctan2(y,x)
            #print(y, ' ', x, ' ', angle)
        if i:
            action += 128
        if (state['distance_to_center'] < 0) and angle <= 0.05:
            action += 2
        if (state['distance_to_center'] > 0) and angle >= -0.05:
            action += 1
        if abs(angle) > 1 and (state['distance_to_center']) > 2:
            action += 32
        prev_pos = state['position_along_track']
        prev_dist = state['distance_to_center']
        """
        if state['position_along_track'] < best:
            idle += 1
        if state['position_along_track'] > best:
            best = state['position_along_track']
            idle = 0
        if idle > 200 or state['wrongway'] != 0:
            action += 64
            idle = 0
        """

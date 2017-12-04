import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from pykart import Kart
from time import time, sleep
import pygame
from pygame.locals import *
from pygame.compat import geterror
import numpy as np
import tensorflow as tf
import random

pygame.init()
screen = pygame.display.set_mode((300,200))
clock = pygame.time.Clock()
keys = {K_RIGHT: 0, K_LEFT: 0, K_UP: 0, K_DOWN: 0, K_v: 0, K_SPACE: 0, K_n: 0}
key_to_action = {K_RIGHT: 2, K_LEFT: 1, K_UP: 4, K_DOWN: 8, K_v: 32, K_SPACE: 128, K_n: 16}
input_state_vars = ["position_along_track", "distance_to_center", "speed", "angle", "smooth_speed", "wrongway"]


cur_run = 6
K = Kart("lighthouse", 300, 200)
while True:
    K.restart()
    K.waitRunning()

    states = []
    actions = []
    obss = []

    action = 4
    i = 0
    prev_state = None
    prev_obs = None
    while True:
        i += 1
        # Play Game
        action = 0
        clock.tick(30)
        # Grab Input
        for event in pygame.event.get():
            if event.type == QUIT:
                going = False
            elif event.type == KEYDOWN:
                if event.key in keys:
                    keys[event.key] = 1
            elif event.type == KEYUP:
                if event.key in keys:
                    keys[event.key] = 0
        # Calculate action
        for key in keys.keys():
            if keys[key]:
                action += key_to_action[key]
        step = K.step(action)

        if step:
            state, obs = step
            if not prev_state:
                prev_state = state
                prev_action = action
                prev_obs = obs
                continue
            # Next game
            if state['finish_time']:
                break

            if i % 3:
                prev_state['position_along_track'] = prev_state['position_along_track'] % 1
                states.append([prev_state[key] for key in input_state_vars])
                actions.append([prev_action, 0])
                obss.append(prev_obs)
            prev_state = state
            prev_action = action
            prev_obs = obs
    np.save('init_data/prev_state_memory_%d.npy' % (cur_run), np.asarray(states))
    np.save('init_data/prev_a_r_%d.npy' % (cur_run), np.asarray(actions))
    np.save('init_data/prev_obs_memory_%d.npy' % (cur_run), np.asarray(obss))
    cur_run += 1

pygame.quit()

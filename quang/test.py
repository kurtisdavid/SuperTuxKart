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
going = True
keys = {K_RIGHT: 0, K_LEFT: 0, K_UP: 0, K_DOWN: 0, K_v: 0, K_SPACE: 0, K_n: 0}
key_to_action = {K_RIGHT: 2, K_LEFT: 1, K_UP: 4, K_DOWN: 8, K_v: 32, K_SPACE: 128, K_n: 16}

K = Kart("lighthouse", 300, 200)
K.restart()
K.waitRunning()

states = []
prev_state = None
t = 0
action = 4
state = None
accepted_keys = ['distance_to_center', 'position_along_track']

while True:
    action = 0
    clock.tick(30)
    for event in pygame.event.get():
        if event.type == QUIT:
            going = False
        elif event.type == KEYDOWN:
            if event.key in keys:
                keys[event.key] = 1
        elif event.type == KEYUP:
            if event.key in keys:
                keys[event.key] = 0
    turn = 0
    drift = 0
    for key in keys.keys():
        if keys[key]:
            action += key_to_action[key]
        if keys[key]:
            if key == K_RIGHT:
                turn = 2
            if key == K_LEFT:
                turn = 1
            if key == K_v:
                drift = 1
    step = K.step(action)

    if step:
        state, obs = step
        if state['finish_time']:
            break
        if not prev_state:
            prev_state = [state['distance_to_center'], state['position_along_track'] % 1]
            continue

        state = [state['distance_to_center'], state['position_along_track'] % 1]
        prev_state.extend(state)
        states.append({'input': prev_state, 'turn': turn, 'drift': drift})
        prev_state = state

pygame.quit()
K.quit()

I = tf.placeholder(tf.float64, (None, len(accepted_keys) * 2), name="input")
label = tf.placeholder(tf.int64, (None,), name="label")
h = tf.contrib.layers.fully_connected(I, 100, activation_fn=tf.nn.relu);
h = tf.contrib.layers.fully_connected(h, 100, activation_fn=tf.nn.relu);
h = tf.contrib.layers.fully_connected(h, 100, activation_fn=tf.nn.relu);
h = tf.contrib.layers.fully_connected(h, 100, activation_fn=tf.nn.relu);
output = tf.contrib.layers.fully_connected(h, 3, activation_fn=None)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=label)

optimizer = tf.train.AdamOptimizer(0.001, 0.9, 0.999)
train = optimizer.minimize(loss)

label_drift = tf.placeholder(tf.int64, (None,), name="label_drift")
h = tf.contrib.layers.fully_connected(I, 100, activation_fn=tf.nn.relu);
h = tf.contrib.layers.fully_connected(h, 100, activation_fn=tf.nn.relu);
h = tf.contrib.layers.fully_connected(h, 100, activation_fn=tf.nn.relu);
h = tf.contrib.layers.fully_connected(h, 100, activation_fn=tf.nn.relu);
output_drift = tf.contrib.layers.fully_connected(h, 2, activation_fn=None)

loss_drift = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_drift, labels=label_drift)

optimizer = tf.train.AdamOptimizer(0.001, 0.9, 0.999)
train_drift = optimizer.minimize(loss_drift)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(200):
    random.shuffle(states)
    for j in range(int(len(states) / 32)):
        batch_inputs = [state['input'] for state in states[j*32:(j+1)*32]]
        batch_labels = [state['turn'] for state in states[j*32:(j+1)*32]]
        batch_drift  = [state['drift'] for state in states[j*32:(j+1)*32]]
        loss_val, loss_val_drift, _, _ = sess.run(
                [loss, loss_drift, train, train_drift],
                {I: batch_inputs, label: batch_labels, label_drift: batch_drift})

print('Training done')

action = 4

K = Kart('lighthouse', 300, 200)
K.restart()
K.waitRunning()

prev_state = None

while True:
    step = K.step( action )
    if step:
        state, obs = step
        if t == state['timestamp']:
            continue
        t = state['timestamp']

        if not prev_state:
            prev_state = [state['distance_to_center'], state['position_along_track'] % 1]
            continue

        # Accelerate + Nitro
        action = 4 + 16

        # Toggle use item
        i = not i
        if i:
            action += 128

        state = [state['distance_to_center'], state['position_along_track'] % 1]
        prev_state.extend(state)

        out, out_d = sess.run([output, output_drift], {I: [prev_state]})
        action += int(np.argmax(out)) + int(np.argmax(out_d)) * 32

        prev_state = state

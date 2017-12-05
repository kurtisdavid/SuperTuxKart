import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from pykart import Kart
from time import time, sleep
import numpy as np
import tensorflow as tf

# Collect Data
K = Kart('lighthouse', 300, 200)
K.restart()
K.waitRunning()

action = 4
i = False
prev_pos = -10
prev_dist = -10
dist_delta = 0

t = 0
start_t = time()
states = []
while True:
    step = K.step( action )
    if step:
        state, obs = step
        if t == state['timestamp']:
            continue
        t = state['timestamp']
        state['angle_to_center'] = 0
        if prev_pos == -10:
            prev_pos = state['position_along_track']
            prev_dist = state['distance_to_center']
            continue
        if state['finish_time']:
            break

        # Accelerate + Nitro
        action = 4 + 16

        # Toggle use item
        i = not i
        if i:
            action += 128

        # Calculate Angle
        if prev_dist != -10:
            dist_delta = (state['distance_to_center'] - prev_dist) / 10
            pos_delta = state['position_along_track'] - prev_pos
            state['angle_to_center'] = np.arctan2(dist_delta, pos_delta)

        # Decide when to turn
        if state['distance_to_center'] < 0 and state['angle_to_center'] < 0:
            action += 2
            state['action'] = 2
        elif state['distance_to_center'] > 0 and state['angle_to_center'] > 0:
            action += 1
            state['action'] = 1
        else:
            state['action'] = 0
        states.append({'inputs': [
                state['distance_to_center'],
                state['position_along_track'] % 1,
                prev_pos % 1,
                prev_dist
            ],
            'action': state['action']
        })
        prev_pos = state['position_along_track']
        prev_dist = state['distance_to_center']
K.quit()
print('Collected ', len(states), ' timesteps')

I = tf.placeholder(tf.float64, (None, 4), name="input")
label = tf.placeholder(tf.int64, (None,), name="label")
h = tf.contrib.layers.fully_connected(I, 64);
h = tf.contrib.layers.fully_connected(h, 64);
h = tf.contrib.layers.fully_connected(h, 64);
h = tf.contrib.layers.fully_connected(h, 64);
output = tf.contrib.layers.fully_connected(h, 3, activation_fn=None)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=label)

optimizer = tf.train.AdamOptimizer(0.001, 0.9, 0.999)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
    for j in range(int(len(states) / 32)):
        batch_inputs = [state['inputs'] for state in states[j*32:(j+1)*32]]
        batch_labels = [state['action'] for state in states[j*32:(j+1)*32]]
        loss_val, _ = sess.run([loss, train], {I: batch_inputs, label: batch_labels})

print('Training done')

action = 4

K = Kart('lighthouse', 300, 200)
K.restart()
K.waitRunning()

total = 0
n = 0
while True:
    step = K.step( action )
    if step:
        state, obs = step
        if t == state['timestamp']:
            continue
        t = state['timestamp']
        if not start_t:
            start_t = t
        state['angle_to_center'] = 0
        if prev_pos == -10:
            prev_pos = state['position_along_track']
            prev_dist = state['distance_to_center']
            continue
        if state['finish_time']:
            break

        # Accelerate + Nitro
        action = 4 + 16

        # Toggle use item
        i = not i
        if i:
            action += 128

        # Calculate Angle
        if prev_dist != -10:
            dist_delta = (state['distance_to_center'] - prev_dist) / 10
            pos_delta = state['position_along_track'] - prev_pos
            state['angle_to_center'] = np.arctan2(dist_delta, pos_delta)

        # Decide when to turn
        if state['distance_to_center'] < 0 and state['angle_to_center'] < 0:
            state['action'] = 2
        elif state['distance_to_center'] > 0 and state['angle_to_center'] > 0:
            state['action'] = 1
        else:
            state['action'] = 0

        out = np.argmax(sess.run(output, {I: [[state['distance_to_center'],
            state['position_along_track'] % 1, prev_pos % 1, prev_dist]]}))
        action += int(out)
        if int(out) != state['action']:
            total += 1
        n += 1

        prev_pos = state['position_along_track']
        prev_dist = state['distance_to_center']
print(1 - total / n)

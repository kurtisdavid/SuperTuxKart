import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from pykart import Kart
import pykart
import tensorflow as tf
import numpy as np

VALID_ACTIONS = [0,1,2,33,34]

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

class QFunc:
    def __init__(self, init_memory):
        self.memory = init_memory

        #Make the network
        self.image = tf.placeholder(tf.float32, (None,200,300,3), name='input image')
        self.action = tf.placeholder(tf.float32, (None,len(VALID_ACTIONS)), name='action input')
        self.state = tf.placeholder(tf.float32, (None, 10), name='state input')
        resize = tf.image.resize_images(self.image,[64,96])
        _, std = tf.nn.moments(self.image, axes=[1])
        mu = tf.reduce_mean(self.image,1) # per pixel mean
        white_image = (self.image - mu) / std

        layer1 = tf.contrib.layers.conv2d(inputs = white_image, stride = 4, kernel_size = (3,3),num_outputs = 32, weights_regularizer = tf.nn.l2_loss)
        layer2 = tf.contrib.layers.conv2d(inputs = layer1, stride = 2,kernel_size = (3,3), num_outputs = 16, weights_regularizer = tf.nn.l2_loss)
        layer3 = tf.contrib.layers.conv2d(inputs = layer2, stride = 2,kernel_size = (3,3), num_outputs = 16, weights_regularizer = tf.nn.l2_loss)
        flat = tf.contrib.layers.flatten(layer3)

        fc1 = tf.contrib.layers.fully_connected(inputs = flat, num_outputs = 8 ,weights_regularizer = tf.nn.l2_loss)
        combined1 = tf.contrib.layers.fully_connected(inputs=tf.concat([fc1,tf.expand_dims(self.state,0)],axis=1),num_outputs = 16,weights_regularizer = tf.nn.l2_loss)
        combined2 = tf.contrib.layers.fully_connected(inputs=tf.concat([combined1,tf.expand_dims(self.action,0)],axis=1),num_outputs = 16,weights_regularizer = tf.nn.l2_loss)

        Qvalues = tf.contrib.layers.fully_connected(inputs = combined2, num_outputs = len(VALID_ACTIONS),weights_regularizer = tf.nn.l2_loss, activation_fn = None)


        self.variables =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.__vars_ph = [tf.placeholder(tf.float32, v.get_shape()) for v in self.variables]
        self.__assign = [v.assign(p) for v,p in zip(self.variables, self.__vars_ph)]

        print("Total number of variables:",np.sum([v.get_shape().num_elements() for v in tf.trainable_variables()]))

        

    #Computes maximal Q value from a given state
    def Q(self, state, obs):
        qvals = sess.run(self.Qvalues, {self.image: obs, self.state: state })
        action_index = np.argmax(qvals)
        return qvals[action_index]

    def train:
        #Don't use this
        EPSILON = 0.01
        NUM_EPOCHS = 100000
        G = Kart("lighthouse", 300, 200)
        K.restart()
        K.waitRunning()
        prev_state = None
        prev_action = None
        prev_position = None
        prev_obs = None
        state = None
        default = 4 | 16
        for  i in range(NUM_EPOCHS):
            #select an action




K = Kart("lighthouse", 300, 200)
player = AI()
K.restart()
K.waitRunning()

MEM_SIZE = 10000
memory = []


prev_state = None
prev_action = None
prev_position = None
prev_obs = None
state = None
start = None
action = 4
i = False

#Initialize the memory using basic AI
while len(memory) < MEM_SIZE:
    step = K.step( action )
    i = not i
    if(step):
        #save previous values for use in dataset
        if state is not None:
            prev_state = state
            prev_action = action
            prev_position  = prev_state['position_along_track']
            prev_obs = obs

        state, obs = step
        if prev_state is not None:
            reward = state['position_along_track'] - prev_state['position_along_track']
            memory.append((prev_state, prev_obs, prev_action, reward, state, obs))
        action = player.get_action(state)
        if i:
            action = action | 128

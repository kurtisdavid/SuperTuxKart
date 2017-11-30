import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from pykart import Kart
import time
import numpy as np
import pickle
import random
import tensorflow as tf

VALID_ACTIONS = [0,1,2]

class AI:
    def __init__(self):
        pass

    def get_action(self,state):
        action = 4 | 16 | 128
        if (abs(state['angle'])<1 and state['distance_to_center']<-5):
            action += 2
        elif ((state['distance_to_center'] < 0 or abs(state['distance_to_center'])<2) and state['angle'] < 0):
            action += 2
        elif (state['angle']>0.5):
            action += 2
        if (abs(state['angle'])<1 and state['distance_to_center']>5):
            action +=  1
        elif ((state['distance_to_center'] > 0 or abs(state['distance_to_center'])<2) and state['angle'] > 0):
            action += 1
        elif (state['angle']<-1.5):
            action += 1
        
        #if(abs(state['angle']) > 0.5):
         #   action += 32
        #print(action)
        return action

sess = tf.Session()

class QFunc:
    def __init__(self):
        self.I = tf.placeholder(tf.float32, (None,200,300,3), name='input')
        self.state = tf.placeholder(tf.float32, (None, 10), name='state')

        #input labels going to be the Q values we precomputed
        self.labels = tf.placeholder(tf.float32, (None, len(VALID_ACTIONS)), name = 'labels')
        
        resize = tf.image.resize_images(self.I,[64,96])
        _, std = tf.nn.moments(self.I, axes=[1])
        mu = tf.reduce_mean(self.I,1) # per pixel mean
        white_image = (self.I - mu) / std


        layer1 = tf.contrib.layers.conv2d(inputs = white_image, stride = 4, kernel_size = (5,5),num_outputs = 32, weights_regularizer = tf.nn.l2_loss)
        layer2 = tf.contrib.layers.conv2d(inputs = layer1, stride = 2,kernel_size = (3,3), num_outputs = 16, weights_regularizer = tf.nn.l2_loss)
        layer3 = tf.contrib.layers.conv2d(inputs = layer2, stride = 2,kernel_size = (3,3), num_outputs = 16, weights_regularizer = tf.nn.l2_loss)
        flat = tf.contrib.layers.flatten(layer3)

        fc1 = tf.contrib.layers.fully_connected(inputs = flat, num_outputs = 16,weights_regularizer = tf.nn.l2_loss)
        combined = tf.concat([fc1,tf.expand_dims(self.state,0)],axis=1)

        fc2 = tf.contrib.layers.fully_connected(inputs = combined, num_outputs = 16,weights_regularizer = tf.nn.l2_loss)
        
        self.qvalues = tf.identity(tf.contrib.layers.fully_connected(inputs = fc2, num_outputs = len(VALID_ACTIONS), weights_regularizer = tf.nn.l2_loss, activation_fn = None), name = 'qvalues')

        self.max_action = tf.identity(tf.argmax(input = qvalues, axis = 1), 'name' = action)
        
        #self.variables =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        #self.__vars_ph = [tf.placeholder(tf.float32, v.get_shape()) for v in self.variables]
        #self.__assign = [v.assign(p) for v,p in zip(self.variables, self.__vars_ph)]

        self.loss = tf.reduce_mean(tf.multiply(0.5, tf.square(self.labels - self.qvalues)))
        optimizer = tf.train.AdamOptimizer(0.001, 0.9, 0.999)
        self.train = optimizer.minimize(self.loss)

        print("Total number of variables:",np.sum([v.get_shape().num_elements() for v in tf.trainable_variables()]))

    def train_batch(self, image, state, labels):
        loss, _ = sess.run( [self.loss, self.train], {self.I: image, self.labels: labels} )
        print("loss: " + str(loss))

    #Returns the maximal action as well as its q value
    def __call__(self, image, state):
        qvalue, qaction = sess.run( [self.qvalues, self.max_action], {self.I: image, self.state: state})
        return qvalue, qaction

    
Q = QFunc()
K.restart()
K.waitRunning()

NUM_STEPS = 10000
EPSILON = 0.01
GAMMA = 0.5
BATCH_SIZE = 32

prev_state = None
prev_action = None
prev_obs = None
state = None
obs = None
start = None
action = 4
i = False
last_time = 0

#Training the Q function
for i in range(NUM_STEPS):
    step = K.step( action )
    i = not i
    if step:
        #Collect some data
        if state is not None:
            prev_state = state
            prev_action = action
            prev_obs = obs

        state, obs = step
        state['position_along_track'] = state['position_along_track'] % 1
        if(np.random.rand() < EPSILON):
            #choose a random action
            action = random.choice(VALID_ACTIONS)
        else:
            _, action = Q(obs, state)
        if i:
            action += 128

        if step and time.time()-last_time > 1/750:
            last_time = time.time()
            if prev_state is not None:
                reward = state['position_along_track'] - prev_state['position_along_track']
                memory.append((prev_state, prev_obs, prev_action, reward, state, obs))

        #Train one batch
        sample = np.random.choice(memory, size = BATCH_SIZE)
        #used to compute labels
        label_states = [s[0] for s in sample]
        label_images = [s[1] for s in sample]
        label_reward = [s[3] for s in sample]

        train_states = [s[4] for s in sample]
        train_images = [s[5] for s in sample]

        label_qvals = label_reward + GAMMA * Q(label_images, label_states)
        Q.train_batch(train_images, train_states, train_reward, label_qvals)





#with open("init_memory.pkl","wb") as f:
    #pickle.dump(memory,f)

#prev_obs_memory = np.asarray(prev_obs_memory)
#np.save("prev_obs_memory",prev_obs_memory)
#prev_a_r = np.asarray(prev_a_r)
#np.save("prev_a_r",prev_a_r)
#obs_memory = np.asarray(obs_memory)
#np.save("obs_memory",obs_memory)
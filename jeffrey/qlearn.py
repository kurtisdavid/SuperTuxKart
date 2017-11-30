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


NUM_EPOCHS = 100
EPSILON = 0.01
GAMMA = 0.5
BATCH_SIZE = 32
NEW_DATA_SIZE = 1000

prev_state = None
prev_obs = None
state = None
obs = None
start = None
action = 4
i = False
last_time = 0

#Pull the data
prev_obs_memory = np.load('small_prev_obs_memory')
prev_r = np.load('prev_r')
obs_memory = np.load('small_obs')
prev_state_memory = np.load('prev_state_memory')
state_memory = np.load('state_memory')

for e in range(NUM_EPOCHS):
    print("EPOCH: " + str(e))
    K = Kart("lighthouse", 300, 200)
    K.restart()
    K.waitRunning()

    new_prev_obs = []
    new_r = []
    new_obs = []
    new_prev_state = []
    new_state = []
    #Gather data
    while(len(new_state) < NEW_DATA_SIZE):
        step = K.step( action )
        if step:
            if state is not None:
                prev_state = state
                prev_obs = obs
            state, obs = step
            state['position_along_track'] = state['position_along_track'] % 1 # only care about distance in race?

            #Random action
            if np.random.rand() < EPSILON:
                 action = VALID_ACTIONS[np.random.randint(0, len(VALID_ACTIONS))]
            else:
                #Otherwise, pick maximal Q action
                action = Q(obs, state)

        if step and time.time()-last_time > 1/750:
            last_time = time.time()
            if prev_state is not None:
                reward = state['position_along_track'] - prev_state['position_along_track']
                new_prev_obs.append(prev_obs)
                new_obs.append(obs)
                new_r.append(reward)
                new_prev_state.append(prev_state)
                new_state.append(state)

    #add new data to dataset
    prev_obs_memory.extend(new_prev_obs)
    prev_r.extend(new_r)
    obs_memory.extend(new_obs)
    prev_state_memory.extend(new_prev_state)
    state_memory.extend(new_state)
    K.quit()
    #Train now
    sample_indices = np.random.choice(range(len(prev_r)), BATCH_SIZE)

    sample_prev_obs = [prev_obs_memory[i] for i in sample_indices]
    sample_r = [prev_r[i] for i in sample_indices]
    sample_obs = [obs_memory[i] for i in sample_indices]
    sample_prev_state = [prev_state_memory[i] for i in sample_indices]
    sample_state = [state_memory[i] for i in sample_indices]

    qval_label, _ = Q(sample_obs, sample_state)
    labels = sample_r + GAMMA * qval_label
    Q.train_batch(sample_obs, sample_state, labels)











#Training the Q function
# High level idea: Gather data for some time
# Then grab a batch from the data set and train





#with open("init_memory.pkl","wb") as f:
    #pickle.dump(memory,f)

#prev_obs_memory = np.asarray(prev_obs_memory)
#np.save("prev_obs_memory",prev_obs_memory)
#prev_a_r = np.asarray(prev_a_r)
#np.save("prev_a_r",prev_a_r)
#obs_memory = np.asarray(obs_memory)
#np.save("obs_memory",obs_memory)

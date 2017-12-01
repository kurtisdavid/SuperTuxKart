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

VALID_ACTIONS = np.asarray([0,1,2])

input_state_vars = ["position_along_track", "distance_to_center", "speed", "angle", "smooth_speed", "wrongway"]

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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

class QFunc:
    def __init__(self):
        self.I = tf.placeholder_with_default(tf.zeros([1,200,300,3]), (None,200,300,3), name='input')
        self.state = tf.placeholder(tf.float32, (None, len(input_state_vars)), name='state')
        #input labels going to be the Q values we precomputed
        self.labels = tf.placeholder(tf.float32, (None, 1), name = 'labels')

        self.resize = tf.image.resize_images(self.I,[64,96])
        #_, std = tf.nn.moments(self.resize, axes=[0])
        #mu = tf.reduce_mean(self.resize,0) # per pixel mean
        #print(std.shape)
        #white_image = (self.resize - mu) / std
        white_image = tf.map_fn(tf.image.per_image_standardization,self.resize)

        layer1 = tf.contrib.layers.conv2d(inputs = white_image, stride = 4, kernel_size = (5,5),num_outputs = 32, weights_regularizer = tf.nn.l2_loss)
        layer2 = tf.contrib.layers.conv2d(inputs = layer1, stride = 2,kernel_size = (3,3), num_outputs = 16, weights_regularizer = tf.nn.l2_loss)
        layer3 = tf.contrib.layers.conv2d(inputs = layer2, stride = 2,kernel_size = (3,3), num_outputs = 16, weights_regularizer = tf.nn.l2_loss)
        flat = tf.contrib.layers.flatten(layer3)

        fc1 = tf.contrib.layers.fully_connected(inputs = flat, num_outputs = 16,weights_regularizer = tf.nn.l2_loss)
        
        combined = tf.concat([fc1,self.state],axis=1)

        fc2 = tf.contrib.layers.fully_connected(inputs = combined, num_outputs = 16,weights_regularizer = tf.nn.l2_loss)

        self.qvalues = tf.identity(tf.contrib.layers.fully_connected(inputs = fc2, num_outputs = VALID_ACTIONS.shape[0], weights_regularizer = tf.nn.l2_loss, activation_fn = None), name = 'qvalues')

        self.max_action = tf.identity(tf.argmax(input = self.qvalues, axis = 1), name = 'action')

        #self.variables =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        #self.__vars_ph = [tf.placeholder(tf.float32, v.get_shape()) for v in self.variables]
        #self.__assign = [v.assign(p) for v,p in zip(self.variables, self.__vars_ph)]

        self.loss = tf.reduce_mean(tf.multiply(0.5, tf.square(self.labels - tf.reduce_max(self.qvalues,axis=1))))
        optimizer = tf.train.AdamOptimizer(0.001, 0.9, 0.999)
        self.train = optimizer.minimize(self.loss)

        print("Total number of variables:",np.sum([v.get_shape().num_elements() for v in tf.trainable_variables()]))

    def train_batch(self, image, state, labels):
        loss, _ = sess.run( [self.loss, self.train], {self.resize: image, self.labels: labels, self.state:state} )
        print("loss: " + str(loss))

    def resize_i(self,image):
        return sess.run([self.resize],{self.I:np.reshape(image,[-1,image.shape[0],image.shape[1],image.shape[2]])})[0]

    #Returns the maximal action as well as its q value
    def __call__(self, image, state):
        if isinstance(state,dict):
            state = [[state[s] for s in input_state_vars]]
        qvalue, qaction = sess.run( [self.qvalues, self.max_action], {self.resize: image, self.state: state})
        return qvalue, qaction


Q = QFunc()
sess.run(tf.global_variables_initializer())


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
prev_obs_memory = np.load('./init_data/small_prev_obs.npy')
prev_r = np.load('./init_data/prev_a_r.npy')[:,1]
obs_memory = np.load('./init_data/small_obs.npy')
prev_state_memory = np.load('./init_data/prev_state_memory.npy')
state_memory = np.load('./init_data/state_memory.npy')

for e in range(NUM_EPOCHS):
    print("EPOCH: " + str(e))
    K = Kart("lighthouse", 300, 200)
    K.restart()
    K.waitRunning()

    #Train now
    sample_indices = np.random.choice(prev_r.shape[0], BATCH_SIZE,replace=False)

    sample_prev_obs = [prev_obs_memory[i] for i in sample_indices]
    sample_r = [prev_r[i] for i in sample_indices]
    sample_obs = [obs_memory[i] for i in sample_indices]
    sample_prev_state = [prev_state_memory[i] for i in sample_indices]
    sample_state = [state_memory[i] for i in sample_indices]
    qvals,qval_label = Q(sample_obs, sample_state)
    qvals = np.asarray([qvals[i][qval_label[i]] for i in range(qvals.shape[0])])
    print(qvals.shape)
    labels = np.reshape(sample_r + GAMMA * qvals,[BATCH_SIZE,1])
    
    Q.train_batch(sample_obs, sample_state, labels)

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
            if obs is None:
                continue
            obs = Q.resize_i(obs)
            state['position_along_track'] = state['position_along_track'] % 1 # only care about distance in race?

            #Random action
            if np.random.rand() < EPSILON:
                 action = 4 + int(VALID_ACTIONS[np.random.randint(0, VALID_ACTIONS.shape[0])])
            else:
                #Otherwise, pick maximal Q action
                action = 4 + int(VALID_ACTIONS[Q(obs, state)[1]][0])

        if step and time.time()-last_time > 1/500:
            last_time = time.time()
            if prev_state is not None and prev_obs is not None:
                reward = state['position_along_track'] - prev_state['position_along_track']
                new_prev_obs.append(prev_obs)
                new_obs.append(obs)
                new_r.append(reward)
                new_prev_state.append([prev_state[s] for s in input_state_vars])
                new_state.append([state[s] for s in input_state_vars])

    

    #add new data to dataset
    replace_indeces = np.random.choice(prev_r.shape[0],NEW_DATA_SIZE,replace=False)
    for i,replace_index in enumerate(replace_indeces):
        prev_obs_memory[replace_index] = new_prev_obs[i]
        prev_r[replace_index] = new_r[i]
        obs_memory[replace_index] = new_obs[i]
        prev_state_memory[replace_index] = new_prev_state[i]
        state_memory[replace_index] = new_state[i]

    K.quit()
    #prev_obs_memory.extend(new_prev_obs)
    #prev_r.extend(new_r)
    #obs_memory.extend(new_obs)
    #prev_state_memory.extend(new_prev_state)
    #state_memory.extend(new_state)


#See the final result
action = 4
while True:
        step = K.step( action )
        if step:
            state, obs = step
            action = 4 + int(VALID_ACTIONS[Q(obs, state)[1]][0])









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

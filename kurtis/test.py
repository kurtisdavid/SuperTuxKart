import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from pykart import Kart
import pykart
import tensorflow as tf
import numpy as np

steer_actions = [0,1,2]
drift_actions = [0,32]
fire_actions = [0,128]

input_state_vars = ["position_along_track", "distance_to_center", "speed", "angle"]
reward_state_vars = ["position_along_track","position_in_race","smooth_speed","wrongway"]
SCORE_WEIGHT = [1,.1, .5, -.5]

def play_level(policy, K, **kwargs):
    K.restart()
    if not K.waitRunning():
        return None

    default = 4
    step = K.step(default)
    while not step:
        step = K.step(default)

    state, obs = step
    while obs is None:
        step = K.step(default)
        while not step:
            step = K.step(default)
        state,obs = step
    
    Is = [1*obs]
    Ss = [[state[s] for s in input_state_vars]]
    Rs = [[[state[s] for s in reward_state_vars]]]
    As = []
    start = state['timestamp']

    
    best_progress, idle_step = -100, 0
    rescue = False
    total_idle = 0
    while state['timestamp']-start<60000:
        A = policy(Is[-1], Ss[-1], **kwargs) | default # feed in last image and state, OR result with accelerate
        As.append(A)
        step = K.step(A)
        while not step:
            step = K.step(A)
        state,obs = step
        Is.append(1*obs)
        Rs.append([[state[s] for s in reward_state_vars]])
        Ss.append([state[s] for s in input_state_vars])
        
        
        if state['position_along_track']<=best_progress:
           idle_step += 1
           total_idle += 1

        if idle_step > 250:
            break

        if best_progress < state['position_along_track']:
            best_progress = state['position_along_track']
            idle_step = 0

        #print("Action:",A,"\tidle_step:",idle_step)



    while len(As) < len(Is):
        As.append(0)
    return np.array(Is), np.array(Rs), total_idle

def score_policy(policy, K,**kwargs):
    score = []
    _,s,idle = play_level(policy,K, **kwargs)
    score.append(s[-1])
    return np.mean(score, axis=0),idle


class CNNPolicy:
    # Only ever create one single CNNPolicy network per graph, otherwise things will FAIL!
    def __init__(self):
        self.I = tf.placeholder(tf.float32, (None,200,300,3), name='input')
        self.state = tf.placeholder(tf.float32, (4), name='input')
        resize = tf.image.resize_images(self.I,[64,96])
        _, std = tf.nn.moments(self.I, axes=[1])
        mu = tf.reduce_mean(self.I,1) # per pixel mean
        white_image = (self.I - mu) / std
        # TODO: Define your convnet
        # You don't need an auxiliary auto-encoder loss, just create
        # a few encoding conv layers.
        layer1 = tf.contrib.layers.conv2d(inputs = white_image, stride = 2, kernel_size = (5,5),num_outputs = 32, weights_regularizer = tf.nn.l2_loss)
        layer2 = tf.contrib.layers.conv2d(inputs = layer1, stride = 2,kernel_size = (3,3), num_outputs = 16, weights_regularizer = tf.nn.l2_loss)
        layer3 = tf.contrib.layers.conv2d(inputs = layer2, stride = 2,kernel_size = (3,3), num_outputs = 16, weights_regularizer = tf.nn.l2_loss)
        layer4 = tf.contrib.layers.conv2d(inputs = layer2, stride = 2,kernel_size = (3,3), num_outputs = 16, weights_regularizer = tf.nn.l2_loss)
        flat = tf.contrib.layers.flatten(layer4)
        fc1 = tf.contrib.layers.fully_connected(inputs = flat, num_outputs = 16,weights_regularizer = tf.nn.l2_loss)

        state_layer = tf.contrib.layers.fully_connected(inputs=tf.expand_dims(self.state, 0),num_outputs = 16,weights_regularizer = tf.nn.l2_loss)

        combined = tf.concat([fc1,state_layer],axis=1)
        
        steer_layer = tf.contrib.layers.fully_connected(inputs = combined,num_outputs = 3, weights_regularizer = tf.nn.l2_loss, activation_fn = None)
        drift_layer = tf.contrib.layers.fully_connected(inputs = combined, num_outputs = 2, weights_regularizer = tf.nn.l2_loss, activation_fn = None)
        fire_layer = tf.contrib.layers.fully_connected(inputs = combined, num_outputs = 2, weights_regularizer = tf.nn.l2_loss, activation_fn = None)
      

        self.steer_logit = tf.identity(steer_layer)
        self.drift_logit = tf.identity(drift_layer)
        self.fire_logit = tf.identity(fire_layer)

        self.predicted_steer = tf.identity(tf.argmax(self.steer_logit, axis=1), name='steer')
        self.predicted_drift = tf.identity(tf.argmax(self.drift_logit, axis=1), name='drift')
        self.predicted_fire = tf.identity(tf.argmax(self.fire_logit, axis=1), name='fire')

        self.variables =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.__vars_ph = [tf.placeholder(tf.float32, v.get_shape()) for v in self.variables]
        self.__assign = [v.assign(p) for v,p in zip(self.variables, self.__vars_ph)]

        print("Total number of variables:",np.sum([v.get_shape().num_elements() for v in tf.trainable_variables()]))
    
    def __call__(self, I, prev_state, verbose=False):
        s,d,f = sess.run([self.predicted_steer,self.predicted_drift,self.predicted_fire], {self.I: I[None], self.state: prev_state})
        #print(s,d,f)
        return steer_actions[int(s)]|drift_actions[int(d)]|fire_actions[int(f)]
    
    @property
    def flat_weights(self):
        import numpy as np
        W = self.weights
        return np.concatenate([w.flat for w in W])
    
    @flat_weights.setter
    def flat_weights(self, w):
        import numpy as np
        S = [v.get_shape().as_list() for v in self.variables]
        s = [np.prod(i) for i in S]
        O = [0] + list(np.cumsum(s))
        assert O[-1] <= w.size
        W = [w[O[i]:O[i+1]].reshape(S[i]) for i in range(len(S))]
        self.weights = W
    
    @property
    def weights(self):
        return sess.run(self.variables)
    
    @weights.setter
    def weights(self, weights):
        sess.run(self.__assign, {v:w for v,w in zip(self.__vars_ph, weights)})


K = Kart("lighthouse",300,200)

P = CNNPolicy()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())


def f(x):
    P.flat_weights = x
    results,idle = score_policy(P,K)
    return np.sum(results*SCORE_WEIGHT)+min(-1,idle*-.001)

SAMPLES_PER_EPOCH = 10
EPOCHS = 30
SURVIVAL_RATE = 0.2
VARIANCE_EPS = 0.1
e = 0.001
prob = 0.01
init_weights = []
scores = []
for i in range(SAMPLES_PER_EPOCH):
    # extract initial weights using xavier initialization
    #tf.reset_default_graph()
    init_weights.append(P.flat_weights + np.random.normal(0,.01,size = (P.flat_weights.shape[0])))
    scores.append(f(init_weights[-1]))


#print(scores)
best = np.argmax(scores)
P.flat_weights = init_weights[best]

# print("Init max: ",scores[max_])
# print("Calc again: ",f(P.flat_weights))
mean = np.zeros(P.flat_weights.shape)
std = 0*mean + 1


# TODO: Use whichever optimization algorithm you want here
first = P.flat_weights
best_policies = [P.flat_weights]
for it in range(EPOCHS):
    individuals = np.random.normal(0,VARIANCE_EPS,size=(SAMPLES_PER_EPOCH,P.flat_weights.shape[0]))+P.flat_weights
    for i in range(individuals.shape[0]):
        lel = np.random.random(size=P.flat_weights.shape[0])
        jitter = np.asarray([e for i in range(lel.shape[0])])
        individuals[i] = individuals[i]+jitter
    results = np.asarray([f(i) for i in individuals])
    winner = np.argmax(results)
    P.flat_weights = individuals[winner]
    best_policies.append(individuals[winner])
    print("["+str(it)+"]\t",results[winner])
    if results[winner]>0.1:
        util.save('final_gf'+str(results[winner])[2:]+'.tfg', session=sess)

best_policies = np.asarray(best_policies)
# who's really the best?
results = np.asarray([f(i) for i in best_policies])
winner = np.argmax(results)

P.flat_weights = best_policies[winner]
util.save('final_gf_winner'+str(results[winner])[2:]+'.tfg', session=sess)
print("PLAYING FINAL")
play_level(P,K)
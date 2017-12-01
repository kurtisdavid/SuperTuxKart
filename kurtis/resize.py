import numpy as np
import tensorflow as tf

obs = np.load("./obs_memory.npy")
prev = np.load("./prev_obs_memory.npy")

I = tf.placeholder(tf.float16,(None,200,300,3))
resize = tf.cast(tf.image.resize_images(I,[64,96]),dtype = tf.uint8)

sess = tf.Session()
I = tf.placeholder(tf.float16,(None,200,300,3))
resize = tf.cast(tf.image.resize_images(I,[64,96]),dtype = tf.uint8)

resized_obs = []
resized_prev = []
for i in range(10):
    resized_obs.append(sess.run(resize,{I:obs[i*1000:(i+1)*1000]}))
    resized_prev.append(sess.run(resize,{I:prev[i*1000:(i+1)*1000]}))
resized_obs = np.asarray(resized_obs).reshape([-1,64,96,3])
resized_prev = np.asarray(resized_prev).reshape([-1,64,96,3])

np.save("./small_obs.npy",resized_obs)
np.save("./small_prev_obs.npy",resized_prev)
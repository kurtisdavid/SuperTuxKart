import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from pykart import Kart

K = Kart("lighthouse", 300, 200)
K.restart()
K.waitRunning()

# Pick your favorite number of iterations here
for i in range(10000):
  state, obs = K.step( 4 ) # For now we just accelerate and crash
  # state is a dict with the internal game state
  # obs is a 300 x 200 x 3 uint8 image
  # Your job is to figure out what action to perform in the next step.
  # Note unlike pytux, pykart is not synchronized. If it takes you too to make
  # up your mind here, your action will be applied with a certain delay.

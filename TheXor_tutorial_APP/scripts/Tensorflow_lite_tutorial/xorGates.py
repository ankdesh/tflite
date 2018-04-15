import numpy as np
import tensorflow as tf
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import sigmoid
from keras.losses import MSE
from keras.optimizers import SGD
from keras.metrics import binary_accuracy
from tensorflow.core.protobuf import saver_pb2
#sess=tf.Session()
#k.set_session(sess)

logicand=np.array([[0,0,0],
                 [0,1,0],
                 [1,0,0],
                 [1,1,1]])


logicor=np.array([[0,0,0],
                 [0,1,1],
                 [1,0,1],
                 [1,1,1]])

logicxor=np.array([[0,0,0],
                 [0,1,1],
                 [1,0,1],
                 [1,1,0]])

logicnot=np.array([[0,1],
                 [1,0]])
# x=logicand[:,:2]
# y=logicand[:,-1]

# x=logicor[:,:2]
# y=logicor[:,-1]

x=logicxor[:,:2]
y=logicxor[:,-1]

# x=logicnot[:,:1]
# y=logicnot[:,-1]

  # Symbols
X = tf.placeholder("float", shape=[1,1,4, 3])
y = tf.placeholder("float", shape=[2])

# Convolutional Layer #1
conv1 = tf.layers.conv2d(
    inputs=X,
    filters=1,
    kernel_size=[3,3],
    padding="same",
    activation=tf.nn.relu)

logits = tf.layers.dense(inputs=conv1, units=2)

graph = tf.Graph()

with graph.as_default():
  
  w1 = tf.Variable(tf.random_normal(shape=[2,3]), name='w1')
  w2 = tf.Variable(tf.random_normal(shape=[3,5]), name='w2')
  
  outp = tf.matmul(w1,w2, name = "outp")
    
  saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(outp))
    saver.save(sess, './frozen_graph/my_test_model.ckpt')

#  model = Sequential()
#  model.add(Dense(2,activation=sigmoid,input_dim=2))
#  model.add(Dense(1,activation=sigmoid))
#  model.compile(loss=MSE,optimizer=SGD(lr=1))
#  model.fit(x,y,epochs=1000)
#  #model.save("xorGates.h5")
#
#  saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
#  saver.save(sess, './xorGates.ckpt')


#print(model.predict(np.array([[,1]])))x

# model = Sequential()
# model.add(Dense(1,activation=sigmoid,input_dim=1))
# model.compile(loss=MSE,optimizer=SGD(lr=1))
# model.fit(x,y,epochs=10000)
# model.save("notGAtes.h5")

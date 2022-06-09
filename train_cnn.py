# Wei Wang (ww8137@mail.ustc.edu.cn)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file, You
# can obtain one at http://mozilla.org/MPL/2.0/.
# ==============================================================================

import time
import sys
import math
import numpy as np
import os

# load MNIST data
from input_data import *
# start tensorflow interactiveSession
import tensorflow as tf
# Note: if class numer is 2 or 20, please edit the variable named "num_classes" in /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py"
# DATA_DIR = sys.argv[1]
# CLASS_NUM = int(sys.argv[2])
# TRAIN_ROUND = int(sys.argv[3])
DATA_DIR = '/home/fitmta/Binh53/DoAnTotNghiep/dataset/temp/Core_DataProcessing/5_Mnist'
CLASS_NUM = 12
TRAIN_ROUND = 700000
EPOCHS = 50
BATCH_SIZE = 100

dict = {0: 'TPLink Router Bridge LAN (Gateway)', 1: 'Amazon Echo_Wireless', 2: 'Withings Smart Baby Monitor_Wired', 3: 'Netatmo Welcome_Wireless', 4: 'Triby Speaker_Wireless', 5: 'Samsung SmartCam_Wireless', 6: 'HP Printer_Wireless', 7: 'Non-IoT_Wireless', 8: 'Dropcam_Wireless', 9: 'Belkin wemo motion sensor_Wireless', 10: 'Belkin Wemo switch_Wireless', 11: 'Samsung Galaxy Tab_Wireless'}

# dict= {0:'BitTorrent',1:'Facetime',2:'FTP',3:'Gmail',4:'MySQL',5:'Outlook',6:'Skype',7:'SMB',8:'Weibo',9:'WorldOfWarcraft',10:'Cridex',11:'Geodo',12:'Htbot',13:'Miuref',14:'Neris',15:'Nsis-ay',16:'Shifu',17:'Tinba',18:'Virut',19:'Zeus'}
folder = os.path.split(DATA_DIR)[1]

sess = tf.InteractiveSession()

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', DATA_DIR, 'Directory for storing data')

mnist = read_data_sets(FLAGS.data_dir, one_hot=True)
print(mnist.train.images)
# function: find a element in a list
def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return -1

# weight initialization
# def weight_variable(name, shape):
#     """
#     Create a weight variable with appropriate initialization
#     :param name: weight name
#     :param shape: weight shape
#     :return: initialized weight variable
#     """
#     initer = tf.truncated_normal_initializer(stddev=0.01)
#     return tf.get_variable('W_' + name,
#                            dtype=tf.float32,
#                            shape=shape,
#                            initializer=initer)

# def bias_variable(name, shape):
#     """
#     Create a bias variable with appropriate initialization
#     :param name: bias variable name
#     :param shape: bias variable shape
#     :return: initialized bias variable
#     """
#     initial = tf.constant(0., shape=shape, dtype=tf.float32)
#     return tf.get_variable('b_' + name,
#                            dtype=tf.float32,
#                            initializer=initial)
# def fc_layer(x, num_units,name, use_relu=True, use_softmax = False):
#     """
#     Create a fully-connected layer
#     :param x: input from previous layer
#     :param num_units: number of hidden units in the fully-connected layer
#     :param name: layer name
#     :param use_relu: boolean to add ReLU non-linearity (or not)
#     :return: The output array
#     """
#     in_dim = x.get_shape()[1]
#     W = weight_variable(name,shape=[in_dim, num_units])
#     b = bias_variable(name,[num_units])
#     layer = tf.matmul(x, W)
#     layer += b
#     if use_relu:
#         layer = tf.nn.relu(layer)
#     if use_softmax:
#         layer = tf.nn.softmax(layer)
#     return layer
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Create the model
# tf.disable_eager_execution()
# placeholder
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, CLASS_NUM])

# first convolutinal layer
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# # # dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# # readout layer
w_fc2 = weight_variable([1024, CLASS_NUM])
b_fc2 = bias_variable([CLASS_NUM])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# =============== test
# weights = {
#     'h1': tf.Variable(tf.random_normal([784, 784])),
#     'out': tf.Variable(tf.random_normal([784, CLASS_NUM]))
# }
# biases = {
#     'b1': tf.Variable(tf.random_normal([784])),
#     'out': tf.Variable(tf.random_normal([CLASS_NUM]))
# }
# def neural_net(x):
#     # Lớp fully connected với 256 neurons
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     # Lớp output fully connected với mỗi neuron cho một class
#     out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
#     return out_layer
# y_conv = neural_net(x)
# ==================== test




# Create a fully-connected layer with h1 nodes as hidden layer
# fc1 = fc_layer(x, 784,'FC1', use_relu=True)
#Create a fully-connected layer with n_classes nodes as output layer
# y_conv = fc_layer(fc1, CLASS_NUM,'OUT', use_softmax=True)
# define var&op of training&testing
actual_label = tf.argmax(y_, 1)
label,idx,count = tf.unique_with_counts(actual_label)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
predict_label = tf.argmax(y_conv, 1)
label_p,idx_p,count_p = tf.unique_with_counts(predict_label)
correct_prediction = tf.equal(predict_label, actual_label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
correct_label=tf.boolean_mask(actual_label,correct_prediction)
label_c,idx_c,count_c=tf.unique_with_counts(correct_label)

# if model exists: restore it
# else: train a new model and save it
saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
model_name = "model_" + str(CLASS_NUM) + "class_" + folder
model =  model_name + '/' + model_name + ".ckpt"
if not os.path.exists(model):
    sess.run(tf.global_variables_initializer())
    if not os.path.exists(model_name):
        os.makedirs(model_name)
    # with open('out.txt','a') as f:
    #     f.write(time.strftime('%Y-%m-%d %X',time.localtime()) + "\n")
    #     f.write('DATA_DIR: ' + DATA_DIR+ "\n")
    print(f'Your train samples: {mnist.train.num_examples}')
    # for i in range(TRAIN_ROUND+1):
    #     ts = time.time()
    #     batch = mnist.train.next_batch(BATCH_SIZE)
    #     if i%100 == 0:
    #         dt = time.time() - ts
    #         train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
    #         s = "step %d, train accuracy %g, speed: %f sec" %(i, train_accuracy, dt)
    #         print(s)
    #         tf.reset_default_graph()
    #         # if i%2000 == 0:
    #         #     with open('out.txt','a') as f:
    #         #         f.write(s + "\n")
    #     train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
    num_tr_iter = int(mnist.train.num_examples / BATCH_SIZE)
    for epoch in range(EPOCHS):
        ts = time.time()
        for iteration in range(num_tr_iter):
            x_batch, y_batch = mnist.train.next_batch(BATCH_SIZE)
            train_step.run(feed_dict={x:x_batch, y_:y_batch, keep_prob:0.5})
        dt = time.time() - ts
        train_accuracy = accuracy.eval(feed_dict={x:x_batch, y_:y_batch, keep_prob:1.0})
        s = "step %d, train accuracy %g, speed: %f sec" %(epoch, train_accuracy, dt)
        print(s)
        tf.reset_default_graph()
    save_path = saver.save(sess, model)
    print("Model saved in file:", save_path)
else:        
    saver.restore(sess, model)
    print("Model restored: " + model)
    
# evaluate the model
print(mnist.test.num_examples)
# label,count,label_p,count_p,label_c,count_c,acc=sess.run([label,count,label_p,count_p,label_c,count_c,accuracy],{x: mnist.test.images[0:10000], y_: mnist.test.labels[0:10000], keep_prob:1.0})
BATCH_SIZE_TEST = 10000
step_per_epoch = math.ceil(mnist.test.num_examples / BATCH_SIZE_TEST)
count_actual = [0] * CLASS_NUM
count_correct = [0] * CLASS_NUM
count_predict = [0] * CLASS_NUM
acc_arr = [0] * step_per_epoch
for step in range(step_per_epoch):
    tf.reset_default_graph()
    start = int(step*BATCH_SIZE_TEST)
    end = int((step+1)*BATCH_SIZE_TEST) if step != (step_per_epoch - 1) else mnist.test.num_examples
    label_step,count_step,label_p_step,count_p_step,label_c_step,count_c_step,acc_step=sess.run([label,count,label_p,count_p,label_c,count_c,accuracy],feed_dict = {x: mnist.test.images[start:end], y_: mnist.test.labels[start:end], keep_prob:1.0})
    acc_arr[step] = acc_step
    for i in range(CLASS_NUM):
        n1 = find_element_in_list(i,label_step.tolist())
        count_actual[i] += count_step[n1]
        n2 = find_element_in_list(i,label_c_step.tolist())
        count_correct[i] += count_c_step[n2] if n2>-1 else 0
        n3 = find_element_in_list(i,label_p_step.tolist())
        count_predict[i] += count_p_step[n3] if n3>-1 else 0
    # print(count_actual,count_correct,count_predict)
acc_list = []

for i in range(CLASS_NUM):
    recall = float(count_correct[i])/float(count_actual[i])
    precision = float(count_correct[i])/float(count_predict[i]) if count_predict[i]>0 else -1
    print(str(i),dict[i],str(precision),str(recall))
    acc_list.append([str(i),dict[i],str(precision),str(recall)])
with open('out.txt','a') as f:
    f.write("\n")
    t = time.strftime('%Y-%m-%d %X',time.localtime())
    f.write(t + "\n")
    f.write('DATA_DIR: ' + DATA_DIR+ "\n")
    for item in acc_list:
        f.write(', '.join(item) + "\n")
    f.write('Total accuracy: ' + str(sum(acc_arr) / len(acc_arr)) + "\n\n")

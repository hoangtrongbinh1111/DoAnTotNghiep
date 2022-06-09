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
sess = tf.InteractiveSession()
# Note: if class numer is 2 or 20, please edit the variable named "num_classes" in /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py"
# DATA_DIR = sys.argv[1]
# CLASS_NUM = int(sys.argv[2])
# TRAIN_ROUND = int(sys.argv[3])
DATA_DIR = '/home/fitmta/Binh53/DoAnTotNghiep/dataset/temp/Core_DataProcessing/5_Mnist'
CLASS_NUM = 12
TRAIN_ROUND = 700000
EPOCHS = 400
BATCH_SIZE = 100

dict = {0: 'TPLink Router Bridge LAN (Gateway)', 1: 'Amazon Echo_Wireless', 2: 'Withings Smart Baby Monitor_Wired', 3: 'Netatmo Welcome_Wireless', 4: 'Triby Speaker_Wireless', 5: 'Samsung SmartCam_Wireless', 6: 'HP Printer_Wireless', 7: 'Non-IoT_Wireless', 8: 'Dropcam_Wireless', 9: 'Belkin wemo motion sensor_Wireless', 10: 'Belkin Wemo switch_Wireless', 11: 'Samsung Galaxy Tab_Wireless'}

# dict= {0:'BitTorrent',1:'Facetime',2:'FTP',3:'Gmail',4:'MySQL',5:'Outlook',6:'Skype',7:'SMB',8:'Weibo',9:'WorldOfWarcraft',10:'Cridex',11:'Geodo',12:'Htbot',13:'Miuref',14:'Neris',15:'Nsis-ay',16:'Shifu',17:'Tinba',18:'Virut',19:'Zeus'}
folder = os.path.split(DATA_DIR)[1]

sess = tf.InteractiveSession()

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', DATA_DIR, 'Directory for storing data')

mnist = read_data_sets(FLAGS.data_dir, one_hot=True)
print(len(mnist.train.images))
# function: find a element in a list
# Các biến lưu trữ weight & bias của các layers
n_hidden_1 = 784 # layer thứ nhất với 256 neurons
n_hidden_2 = 784 # layer thứ hai với 256 neurons
num_input = 784 # Số features đầu vào (tập MNIST với shape: 28*28)
num_classes = 12 # Tổng số lớp của MNIST (các số từ 0-9)
learning_rate = 0.1
num_steps = 5000
batch_size = 128
display_step = 100
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}
def neural_net(x):
    # Lớp fully connected với 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Lớp output fully connected với mỗi neuron cho một class
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer
# Tạo model
logits = neural_net(X)

# Định nghĩa hàm loss và optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Đánh giá model
actual_label = tf.argmax(Y, 1)
label,idx,count = tf.unique_with_counts(actual_label)
cross_entropy = -tf.reduce_sum(Y*tf.log(logits))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
predict_label = tf.argmax(logits, 1)
label_p,idx_p,count_p = tf.unique_with_counts(predict_label)
correct_prediction = tf.equal(predict_label, actual_label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
correct_label=tf.boolean_mask(actual_label,correct_prediction)
label_c,idx_c,count_c=tf.unique_with_counts(correct_label)

# create model
saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
model_name = "model_" + str(CLASS_NUM) + "class_" + folder
model =  model_name + '/' + model_name + ".ckpt"
# if not os.path.exists(model):
sess.run(tf.global_variables_initializer())
if not os.path.exists(model_name):
    os.makedirs(model_name)
# Hàm khởi tạo các biến (gán giá trị mặc định)
sess.run(tf.global_variables_initializer())
if not os.path.exists(model_name):
    os.makedirs(model_name)
print(f'Your train samples: {mnist.train.num_examples}')
num_tr_iter = int(mnist.train.num_examples / BATCH_SIZE)
for epoch in range(EPOCHS):
    ts = time.time()
    for iteration in range(num_tr_iter):
        x_batch, y_batch = mnist.train.next_batch(BATCH_SIZE)
        train_step.run(feed_dict={X:x_batch, Y:y_batch})
    dt = time.time() - ts
    train_accuracy = accuracy.eval(feed_dict={X:x_batch, Y:y_batch})
    s = "step %d, train accuracy %g, speed: %f sec" %(epoch, train_accuracy, dt)
    print(s)
    tf.reset_default_graph()
save_path = saver.save(sess, model)
print("Model saved in file:", save_path)
# else:        
#     saver.restore(sess, model)
#     print("Model restored: " + model)


print(mnist.test.num_examples)
def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return -1

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
    label_step,count_step,label_p_step,count_p_step,label_c_step,count_c_step,acc_step=sess.run([label,count,label_p,count_p,label_c,count_c,accuracy],feed_dict = {X: mnist.test.images[start:end], Y: mnist.test.labels[start:end]})
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
    print(acc_arr)

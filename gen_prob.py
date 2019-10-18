# Source code to reproduce the results in
# J. Yang, D. Ruan, J. Huang, X. Kang and Y. Shi, "An Embedding Cost Learning Framework Using GAN," in IEEE Transactions on Information Forensics and Security, vol. 15, pp. 839-851, 2020.
# By Jianhua Yang,  yangjh48@mail2.sysu.edu.cn


import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import os
from scipy import ndimage
import scipy.io as sio  # loading the mat file
from batch_norm_layer import batch_norm_layer
import scipy
from scipy import ndimage

# select the graphic card
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
path1 =  '/.../dataset/BOSS256'
# ******************************************* constant value settings ************************************************


NUM_IMG = 10000
BATCH_SIZE = 1
IMAGE_SIZE =  256
NUM_CHANNEL = 1  # gray image
NUM_LABELS = 2  # binary classification
G_DIM = 16  # number of feature maps in generator
STRIDE = 2
KENEL_SIZE = 3
DKENEL_SIZE = 5

PAD_SIZE = int((KENEL_SIZE - 1) / 2)
Initial_learning_rate = 0.0001 # 0.0001
Adam_beta = 0.5
TANH_LAMBDA = 1000000 # For test
# Input Cover and corresponding probability map
cover = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])  # attention  input
is_training = tf.placeholder(tf.bool, name='is_training')

# *********************************************definition of G*********************************************************
def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
# -------------- contracting path ---------------------
with tf.variable_scope("Gen1") as scope:
    NUM = G_DIM * 1
    kernel1_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, NUM_CHANNEL, NUM], stddev=0.02),name="kernel1_G")
    conv1_G = tf.nn.conv2d(cover/255, kernel1_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv1_G")
    bn1_G = batch_norm_layer(conv1_G, is_training,'bn1_G')
    # Embeding_prob feature map shape: 128*128

with tf.variable_scope("Gen2") as scope:
    NUM = G_DIM * 2
    kernel2_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, int(NUM/2), NUM], stddev=0.02),name="kernel2_G")
    conv2_G = tf.nn.conv2d(lrelu(bn1_G, 0.2), kernel2_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv2_G")
    bn2_G = batch_norm_layer(conv2_G, is_training,'bn2_G')
    # Embeding_prob feature map shape: 64*64

with tf.variable_scope("Gen3") as scope:
    NUM = G_DIM * 4
    kernel3_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, int(NUM/2), NUM], stddev=0.02),name="kernel3_G")
    conv3_G = tf.nn.conv2d(lrelu(bn2_G, 0.2), kernel3_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv3_G")
    bn3_G = batch_norm_layer(conv3_G, is_training,'bn3_G')
    # Embeding_prob feature map shape: 32*32

with tf.variable_scope("Gen4") as scope:
    NUM = G_DIM * 8
    kernel4_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, int(NUM/2), NUM], stddev=0.02),name="kernel4_G")
    conv4_G = tf.nn.conv2d(lrelu(bn3_G, 0.2), kernel4_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv4_G")
    bn4_G = batch_norm_layer(conv4_G, is_training,'bn4_G')
    # Embeding_prob feature map shape: 16*16

with tf.variable_scope("Gen5") as scope:
    NUM = G_DIM * 8
    kernel5_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, NUM, NUM], stddev=0.02),name="kernel5_G")
    conv5_G = tf.nn.conv2d(lrelu(bn4_G,0.2), kernel5_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv5_G")
    bn5_G = batch_norm_layer(conv5_G, is_training, 'bn5_G')
    # Embeding_prob feature map shape: 8*8

with tf.variable_scope("Gen6") as scope:
    NUM = G_DIM * 8
    kernel6_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, NUM, NUM], stddev=0.02),name="kernel6_G")
    conv6_G = tf.nn.conv2d(lrelu(bn5_G, 0.2), kernel6_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv6_G")
    bn6_G = batch_norm_layer(conv6_G, is_training, 'bn6_G')
    # Embeding_prob feature map shape: 4*4

with tf.variable_scope("Gen7") as scope:
    NUM = G_DIM * 8
    kernel7_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, NUM, NUM], stddev=0.02),name="kernel7_G")
    conv7_G = tf.nn.conv2d(lrelu(bn6_G, 0.2), kernel7_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv7_G")
    bn7_G = batch_norm_layer(conv7_G, is_training, 'bn7_G')
    # 2*2

with tf.variable_scope("Gen8") as scope:
    NUM = G_DIM * 8
    kernel8_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, NUM, NUM], stddev=0.02),name="kernel8_G")
    conv8_G = tf.nn.conv2d(lrelu(bn7_G,0.2), kernel8_G, [1, STRIDE,STRIDE, 1], padding='SAME', name="conv8_G")
    bn8_G = batch_norm_layer(conv8_G, is_training, 'bn8_G')
    # 1*1

s = IMAGE_SIZE
s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
# -------------- expanding path -----------------
with tf.variable_scope("Gen9") as scope:
    NUM = G_DIM * 8
    out_shape = [BATCH_SIZE, s128, s128, NUM]
    kernel9_G = tf.Variable(tf.random_normal([DKENEL_SIZE, DKENEL_SIZE, NUM, NUM], stddev=0.02), name="kernel9_G")
    conv9_G = tf.nn.conv2d_transpose(tf.nn.relu(bn8_G), kernel9_G, out_shape, [1, STRIDE, STRIDE, 1], name="conv9_G")
    bn9_G = batch_norm_layer(conv9_G, is_training, 'bn9_G')
    bn9_G = tf.nn.dropout(bn9_G, 0.5)
    bn9_G = tf.concat([bn9_G, bn7_G], 3)

with tf.variable_scope("Gen10") as scope:
    NUM = G_DIM * 8
    out_shape = [BATCH_SIZE, s64, s64, NUM]
    kernel10_G = tf.Variable(tf.random_normal([DKENEL_SIZE, DKENEL_SIZE, NUM, NUM*2], stddev=0.02), name="kerne10_G")
    conv10_G = tf.nn.conv2d_transpose(tf.nn.relu(bn9_G), kernel10_G, out_shape, [1, STRIDE, STRIDE, 1], name="conv10_G")
    bn10_G = batch_norm_layer(conv10_G, is_training, 'bn10_G')
    bn10_G = tf.nn.dropout(bn10_G, 0.5)
    bn10_G = tf.concat([bn10_G, bn6_G], 3)

with tf.variable_scope("Gen11") as scope:
    NUM = G_DIM * 8
    out_shape = [BATCH_SIZE, s32, s32, NUM]
    kernel11_G = tf.Variable(tf.random_normal([DKENEL_SIZE, DKENEL_SIZE, NUM, NUM*2], stddev=0.02), name="kerne11_G")
    conv11_G = tf.nn.conv2d_transpose(tf.nn.relu(bn10_G), kernel11_G, out_shape, [1, STRIDE, STRIDE, 1], name="conv11_G")
    bn11_G = batch_norm_layer(conv11_G, is_training, 'bn11_G')
    bn11_G = tf.nn.dropout(bn11_G, 0.5)
    bn11_G = tf.concat([bn11_G, bn5_G], 3)

with tf.variable_scope("Gen12") as scope:
    NUM = G_DIM * 8
    out_shape = [BATCH_SIZE, s16, s16, NUM]
    kernel12_G = tf.Variable(tf.random_normal([DKENEL_SIZE, DKENEL_SIZE, NUM, NUM*2], stddev=0.02), name="kerne12_G")
    conv12_G = tf.nn.conv2d_transpose(tf.nn.relu(bn11_G), kernel12_G, out_shape, [1, STRIDE, STRIDE, 1],
                                      name="conv12_G")
    bn12_G = batch_norm_layer(conv12_G, is_training, 'bn12_G')
    bn12_G = tf.concat([bn12_G, bn4_G], 3)

with tf.variable_scope("Gen13") as scope:
    NUM = G_DIM * 4
    out_shape = [BATCH_SIZE, s8, s8, NUM]
    kernel13_G = tf.Variable(tf.random_normal([DKENEL_SIZE, DKENEL_SIZE, NUM, NUM*4], stddev=0.02), name="kerne13_G")
    conv13_G = tf.nn.conv2d_transpose(tf.nn.relu(bn12_G), kernel13_G, out_shape, [1, STRIDE, STRIDE, 1],name="conv13_G")
    bn13_G = batch_norm_layer(conv13_G, is_training, 'bn13_G')
    bn13_G = tf.concat([bn13_G, bn3_G], 3)

with tf.variable_scope("Gen14") as scope:
    NUM = G_DIM * 2
    out_shape = [BATCH_SIZE, s4, s4, NUM]
    kernel14_G = tf.Variable(tf.random_normal([DKENEL_SIZE, DKENEL_SIZE, NUM, NUM*4], stddev=0.02), name="kerne14_G")
    conv14_G = tf.nn.conv2d_transpose(tf.nn.relu(bn13_G), kernel14_G, out_shape, [1, STRIDE, STRIDE, 1],
                                      name="conv14_G")
    bn14_G = batch_norm_layer(conv14_G, is_training, 'bn14_G')
    bn14_G = tf.concat([bn14_G, bn2_G], 3)

with tf.variable_scope("Gen15") as scope:
    NUM = G_DIM
    out_shape = [BATCH_SIZE, s2, s2, NUM]
    kernel15_G = tf.Variable(tf.random_normal([DKENEL_SIZE, DKENEL_SIZE, NUM, NUM*4], stddev=0.02), name="kerne15_G")
    conv15_G = tf.nn.conv2d_transpose(tf.nn.relu(bn14_G), kernel15_G, out_shape, [1, STRIDE, STRIDE, 1],
                                      name="conv15_G")
    bn15_G = batch_norm_layer(conv15_G, is_training, 'bn15_G')
    bn15_G = tf.concat([bn15_G, bn1_G], 3)

with tf.variable_scope("Gen16") as scope:
    NUM = NUM_CHANNEL
    out_shape = [BATCH_SIZE, s, s, NUM]
    kernel16_G = tf.Variable(tf.random_normal([DKENEL_SIZE, DKENEL_SIZE, NUM, G_DIM * 2], stddev=0.02), name="kerne16_G")
    conv16_G = tf.nn.conv2d_transpose(tf.nn.relu(bn15_G), kernel16_G, out_shape, [1, STRIDE, STRIDE, 1],
                                      name="conv16_G")
Embeding_prob = tf.nn.relu(tf.nn.sigmoid(conv16_G) - 0.5)
 
Embeding_prob_shape = Embeding_prob.get_shape().as_list()

#################################################Ternary encoding module###########################################################
noise = tf.placeholder(tf.float32, Embeding_prob_shape)  # noise holder
modification = -0.5 * tf.nn.tanh((tf.subtract(Embeding_prob, 2*noise)) * TANH_LAMBDA) + 0.5 * tf.nn.tanh((tf.subtract(Embeding_prob, tf.subtract(2.0, 2*noise))) * TANH_LAMBDA)
stego = cover + modification

global_variables = tf.global_variables()
print(global_variables)
# ****************************************************************Training***************************************************************************
image_index = range(1,NUM_IMG+1)
seed = 0

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # ---------------loading the saved parameters------------------
    NUM_ITERATION = 120000
    prob_path = './prob_iter_' + '%d' % NUM_ITERATION
    isExists = os.path.exists(prob_path)
    if not isExists:
        os.makedirs(prob_path)
    else:
        print prob_path
    tf.train.Saver(global_variables).restore(sess,"./model/%d.ckpt"%NUM_ITERATION)
    data_x = np.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
    count = 0
    for epoch in range(0, int(NUM_IMG/BATCH_SIZE)):

        for j in range(BATCH_SIZE):
            count = count % NUM_IMG
            imc = ndimage.imread(path1 + '/' + '%d' %(count+1) + '.pgm')
            data_x[j, :, :, 0] = imc
            count = count + 1

        data_noise = np.random.rand(Embeding_prob_shape[0], Embeding_prob_shape[1], Embeding_prob_shape[2], Embeding_prob_shape[3])
        prob_= sess.run(Embeding_prob , feed_dict={cover: data_x, noise: data_noise, is_training: False})

        sio.savemat(prob_path + '/' + str(count) + '.mat', {'prob': prob_})
        print('processing the %d image' %count)
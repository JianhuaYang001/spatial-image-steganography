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
import scipy.io as sio
from batch_norm_layer import batch_norm_layer

# select the graphic card
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
path1 = '/.../YJH/dataset/SZ256_4w/' # path of training set

# ******************************************* constant value settings ************************************************
NUM_ITERATION = 120000
NUM_IMG = 40000  # The number of images used to train the network
BATCH_SIZE = 24
IMAGE_SIZE =  256
NUM_CHANNEL = 1  # gray image
NUM_LABELS = 2  # binary classification
G_DIM = 16  # number of feature maps in generator
STRIDE = 2
KENEL_SIZE = 3
DKENEL_SIZE = 5
PAYLOAD = 0.4     # Target embedding payload
PAD_SIZE = int((KENEL_SIZE - 1) / 2)
Initial_learning_rate = 0.0001
Adam_beta = 0.5
TANH_LAMBDA = 60 # To balance the embedding simulate and avoid gradient vanish problem

cover = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
is_training = tf.placeholder(tf.bool, name='is_training') # True for training, false for test

# ********************************************* definition of the generator *********************************************************
def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

# -------------- contracting path ---------------------
with tf.variable_scope("Gen1") as scope:
    NUM = G_DIM * 1
    kernel1_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, NUM_CHANNEL, NUM], stddev=0.02),name="kernel1_G")
    conv1_G = tf.nn.conv2d(cover/255, kernel1_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv1_G")
    bn1_G = batch_norm_layer(conv1_G, is_training,'bn1_G')
    # feature map shape: 128*128

with tf.variable_scope("Gen2") as scope:
    NUM = G_DIM * 2
    kernel2_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, int(NUM/2), NUM], stddev=0.02),name="kernel2_G")
    conv2_G = tf.nn.conv2d(lrelu(bn1_G, 0.2), kernel2_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv2_G")
    bn2_G = batch_norm_layer(conv2_G, is_training,'bn2_G')
    # feature map shape: 64*64

with tf.variable_scope("Gen3") as scope:
    NUM = G_DIM * 4
    kernel3_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, int(NUM/2), NUM], stddev=0.02),name="kernel3_G")
    conv3_G = tf.nn.conv2d(lrelu(bn2_G, 0.2), kernel3_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv3_G")
    bn3_G = batch_norm_layer(conv3_G, is_training,'bn3_G')
    # feature map shape: 32*32

with tf.variable_scope("Gen4") as scope:
    NUM = G_DIM * 8
    kernel4_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, int(NUM/2), NUM], stddev=0.02),name="kernel4_G")
    conv4_G = tf.nn.conv2d(lrelu(bn3_G, 0.2), kernel4_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv4_G")
    bn4_G = batch_norm_layer(conv4_G, is_training,'bn4_G')
    #  feature map shape: 16*16

with tf.variable_scope("Gen5") as scope:
    NUM = G_DIM * 8
    kernel5_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, NUM, NUM], stddev=0.02),name="kernel5_G")
    conv5_G = tf.nn.conv2d(lrelu(bn4_G,0.2), kernel5_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv5_G")
    bn5_G = batch_norm_layer(conv5_G, is_training, 'bn5_G')
    # feature map shape: 8*8

with tf.variable_scope("Gen6") as scope:
    NUM = G_DIM * 8
    kernel6_G = tf.Variable(tf.truncated_normal([KENEL_SIZE, KENEL_SIZE, NUM, NUM], stddev=0.02),name="kernel6_G")
    conv6_G = tf.nn.conv2d(lrelu(bn5_G, 0.2), kernel6_G, [1, STRIDE, STRIDE, 1], padding='SAME', name="conv6_G")
    bn6_G = batch_norm_layer(conv6_G, is_training, 'bn6_G')
    # feature map shape: 4*4

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

#***************************************************  double-tanh function for embedding simulation ***************************************************
noise = tf.placeholder(tf.float32, Embeding_prob_shape)  # noise holder
modification = -0.5 * tf.nn.tanh((tf.subtract(Embeding_prob, 2*noise)) * TANH_LAMBDA) + 0.5 * tf.nn.tanh((tf.subtract(Embeding_prob, tf.subtract(2.0, 2*noise))) * TANH_LAMBDA)
stego = cover + modification

# *************************************************** definition of the discriminator **************************************************************
Img = tf.concat([cover, stego], 0)
y_array = np.zeros([BATCH_SIZE * 2, NUM_LABELS], dtype=np.float32)
for i in range(0, BATCH_SIZE):
    y_array[i, 1] = 1
for i in range(BATCH_SIZE, BATCH_SIZE * 2):
    y_array[i, 0] = 1
y = tf.constant(y_array)

Img_label = tf.constant(y_array)

# *********************** high pass filters ***********************
HPF = np.zeros( [5,5,1,6],dtype=np.float32 )
HPF[:, :, 0, 0] = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,-1,1,0],[0,0,0,0,0],[0,0,0,0,0]],dtype=np.float32)
HPF[:, :, 0, 1] = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,-1,0,0],[0,0,1,0,0],[0,0,0,0,0]],dtype=np.float32)
HPF[:, :, 0, 2] = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],dtype=np.float32)
HPF[:, :, 0, 3] = np.array([[0,0,0,0,0],[0,0,1,0,0],[0,0,-2,0,0],[0,0,1,0,0],[0,0,0,0,0]],dtype=np.float32)
HPF[:, :, 0, 4] = np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],dtype=np.float32)
HPF[:, :, 0, 5]= np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=np.float32)

skernel0 = tf.Variable(HPF, name="skernel0")
sconv0 = tf.nn.conv2d(Img, skernel0, [1,1,1,1], 'SAME', name="sconv0")

with tf.variable_scope("Group1") as scope:
    skernel1 = tf.Variable(tf.random_normal( [5,5,6,8],mean=0.0,stddev=0.01 ),name="skernel1")
    sconv1 = tf.nn.conv2d(sconv0, skernel1, [1,1,1,1], padding='SAME',name="sconv1")
    sabs1 = tf.abs(sconv1, name="sabs1")
    sbn1 = batch_norm_layer(sabs1, is_training, 'sbn1')
    stanh1 = tf.nn.tanh(sbn1,name="stanh1")
    spool1 = tf.nn.avg_pool(stanh1, ksize=[1,5,5,1], strides=[1,2,2,1], padding='SAME',name="spool1" )

with tf.variable_scope("Group2") as scope:
    skernel2 = tf.Variable( tf.random_normal( [5,5,8,16],mean=0.0,stddev=0.01 ),name="skernel2")
    sconv2 = tf.nn.conv2d(spool1, skernel2, [1,1,1,1], padding="SAME",name="sconv2"  )
    sbn2 = batch_norm_layer(sconv2,is_training, 'sbn2')
    stanh2 = tf.nn.tanh(sbn2,name="stanh2")
    spool2 = tf.nn.avg_pool( stanh2, ksize=[1,5,5,1], strides=[1,2,2,1], padding='SAME',name="spool2" )

with tf.variable_scope("Group3") as scope:
    skernel3 = tf.Variable( tf.random_normal( [1,1,16,32],mean=0.0,stddev=0.01 ),name="skernel3" )
    sconv3 = tf.nn.conv2d( spool2, skernel3, [1,1,1,1], padding="SAME",name="sconv3"  )
    sbn3 = batch_norm_layer(sconv3,is_training, 'sbn3')
    srelu3 = tf.nn.relu(sbn3,name="sbn3")
    spool3 = tf.nn.avg_pool( srelu3, ksize=[1,5,5,1], strides=[1,2,2,1], padding="SAME",name="spool3" ) #[input,height,width,ouput]

with tf.variable_scope("Group4") as scope:
    skernel4 = tf.Variable( tf.random_normal( [1,1,32,64],mean=0.0,stddev=0.01 ),name="skernel4" )
    sconv4 = tf.nn.conv2d( spool3, skernel4, [1,1,1,1], padding="SAME",name="sconv4"  )
    sbn4 = batch_norm_layer(sconv4, is_training, 'sbn4')
    srelu4 = tf.nn.relu(sbn4,name="srelu4")
    spool4 = tf.nn.avg_pool( srelu4, ksize=[1,5,5,1], strides=[1,2,2,1], padding="SAME",name="spool4" ) #[input,height,width,ouput]

with tf.variable_scope("Group5") as scope:
    skernel5 = tf.Variable( tf.random_normal( [1,1,64,128],mean=0.0,stddev=0.01 ),name="skernel5" )
    sconv5 = tf.nn.conv2d( spool4, skernel5, [1,1,1,1], padding="SAME",name="sconv5"  )
    sbn5 = batch_norm_layer(sconv5, is_training, 'sbn5')
    srelu5 = tf.nn.relu(sbn5,name="srelu5")
    spool5 = tf.nn.avg_pool( srelu5, ksize=[1,16,16,1], strides=[1,1,1,1], padding="VALID",name="spool5" ) #[input,height,width,ouput]

with tf.variable_scope('Group6') as scope:
    spool_shape = spool5.get_shape().as_list()
    spool_reshape = tf.reshape( spool5, [spool_shape[0], spool_shape[1] * spool_shape[2] * spool_shape[3]])
    sweights = tf.Variable( tf.random_normal( [128,2],mean=0.0,stddev=0.01 ),name="sweights" )
    sbias = tf.Variable( tf.random_normal([2],mean=0.0,stddev=0.01),name="sbias" )
    D_y = tf.matmul(spool_reshape, sweights) + sbias

correct_predictionS = tf.equal(tf.argmax(D_y, 1), tf.argmax(Img_label, 1))
accuracyD = tf.reduce_mean(tf.cast(correct_predictionS, tf.float32))
lossD = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_y, labels=Img_label))  # loss of D

# *******************************************************loss function ************************************************************************
gamma = 1
lambda_ent = 1e-7
proChangeP = Embeding_prob / 2.0 + 1e-5
proChangeM = Embeding_prob / 2.0 + 1e-5
proUnchange = 1 - Embeding_prob + 1e-5
entropy = tf.reduce_sum(-(proChangeP) * tf.log(proChangeP) / tf.log(2.0) - (proChangeM) * tf.log(proChangeM) / tf.log(2.0) - proUnchange * tf.log(proUnchange) / tf.log(2.0), reduction_indices=[1, 2, 3])
Payload_learned = tf.reduce_sum(entropy,reduction_indices=0)/IMAGE_SIZE/IMAGE_SIZE/BATCH_SIZE

Capacity = IMAGE_SIZE * IMAGE_SIZE * PAYLOAD
lossEntropy = tf.reduce_mean(tf.pow(entropy - Capacity, 2), reduction_indices=0)
# -------------------loss of the generator -------------
lossGen = gamma * (-lossD) + lambda_ent * lossEntropy
# -------------------trainable variables----------------
variables = tf.trainable_variables()
paramsG = [v for v in variables if (v.name.startswith('Gen'))]
paramsD = [v for v in variables if (v.name.startswith('Group'))]

# -------------------- optimizers ---------------------------
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    optG = tf.train.AdamOptimizer(Initial_learning_rate).minimize(lossGen, var_list=paramsG)
    optD = tf.train.AdamOptimizer(Initial_learning_rate).minimize(lossD, var_list=paramsD)

global_variables = tf.global_variables()
# **************************************************************** adversary training process ***************************************************************************
image_index = range(1,NUM_IMG+1)
seed = 0
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    iteration_num = 0
    data_x = np.zeros([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
    count = 0
    start_time = time.time()
    for Iter_ in range(0+1, NUM_ITERATION+1):
        for j in range(BATCH_SIZE):
            count = count % NUM_IMG
            if (count == 0):
                print('----------- Epoch %d------------'%seed)
                np.random.seed(seed)
                seed = seed+1
                temp_image_index = np.random.permutation(image_index) # shuffle the training set every epoch
            imc = ndimage.imread(path1 + '/' + '%06d' %temp_image_index[count] + '.tif')  # %06d tif
            data_x[j, :, :, 0] = imc
            count = count + 1

        if Iter_ % 5000 == 0:
            saver = tf.train.Saver()
            saver.save(sess, './model/' + '%d' % Iter_ + '.ckpt')
        data_noise = np.random.rand(Embeding_prob_shape[0],Embeding_prob_shape[1],Embeding_prob_shape[2], Embeding_prob_shape[3])
        # update S
        sess.run(optD, feed_dict={cover: data_x,  noise: data_noise, is_training: True})
        # update G 
        _, lG, payload_learned,  OUT, modified, accD,loD, = sess.run([optG, lossGen, Payload_learned, Embeding_prob, modification,
                                                          accuracyD,lossD],
                                                        feed_dict={cover: data_x, noise: data_noise,  is_training: True})
        spend_time = time.time() - start_time
        if Iter_ % 100 == 0:
            print('Iter %d' % Iter_ + '\tlossG=%f' % lG + '\tPayload=%f' % payload_learned + '\tlossD =%f' % loD
                  + '\taccuracyD = %f' % accD + '\tspend time:  %f' % spend_time + 's')
    writer.close()

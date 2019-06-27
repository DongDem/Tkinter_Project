# Parameters
LAMBDA = 0.1
CENTER_LOSS_ALPHA = 0.5
NUM_CLASSES = 7

# Import modules
import tensorflow as tf
import os
import numpy as np
import tflearn
import cv2
import time
import h5py
slim = tf.contrib.slim

train_path = "./fold1_new/training_augment"
classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
image_size = 64
start_time = time.time()

def load_test(test_path):
    image = cv2.imread(test_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
    image = cv2.equalizeHist(image)
    image = np.array(image)
    data = image.flatten()
    return data

# Network Parameters
n_classes = len(classes)


# Construct Network
with tf.name_scope('input'):
    input_images = tf.placeholder(tf.float32, shape=[None, 4096], name='input_images')
    labels = tf.placeholder(tf.int64, shape=(None), name='labels')
    keep_prob = tf.placeholder(tf.float32, name="dropout")
global_step = tf.Variable(0, trainable=False, name='global_step')
labels_center_1 = []
labels_center_2 = []


for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if (i==j) or (j<i):
            continue
        else:
            print([i,j])
            labels_center_1.append(i)
            labels_center_2.append(j)

labels_center_1 = tf.convert_to_tensor(labels_center_1)
labels_center_2 = tf.convert_to_tensor(labels_center_2)
def get_center_loss(features, labels, alpha, num_classes):
    '''
    Arguments:
        features: Tensor, [batch_size, feature_length]
        labels: Tensor, [batch_size]
        alpha: 0-1
        num_classes
    Return:
        loss: Tensor, softmax loss, loss
        centers: Tensor
        centers_update_op: op
    '''
    len_features = features.get_shape()[1]
    num_features = tf.shape(features)[0]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32, initializer=tf.constant_initializer(0), trainable=False)


    labels_0 = tf.fill([num_features,], 0)
    labels_0 = tf.reshape(labels_0, [-1])
    centers_batch_0 = tf.gather(centers, labels_0)
    loss_0 = tf.nn.l2_loss(features - centers_batch_0)

    labels_1 = tf.fill([num_features, ], 1)
    labels_1 = tf.reshape(labels_1, [-1])
    centers_batch_1 = tf.gather(centers, labels_1)
    loss_1 = tf.nn.l2_loss(features - centers_batch_1)

    labels_2 = tf.fill([num_features, ], 2)
    labels_2 = tf.reshape(labels_2, [-1])
    centers_batch_2 = tf.gather(centers, labels_2)
    loss_2 = tf.nn.l2_loss(features - centers_batch_2)

    labels_3 = tf.fill([num_features, ], 3)
    labels_3 = tf.reshape(labels_3, [-1])
    centers_batch_3 = tf.gather(centers, labels_3)
    loss_3 = tf.nn.l2_loss(features - centers_batch_3)

    labels_4 = tf.fill([num_features, ], 4)
    labels_4 = tf.reshape(labels_4, [-1])
    centers_batch_4 = tf.gather(centers, labels_4)
    loss_4 = tf.nn.l2_loss(features - centers_batch_4)

    labels_5 = tf.fill([num_features, ], 5)
    labels_5 = tf.reshape(labels_5, [-1])
    centers_batch_5 = tf.gather(centers, labels_5)
    loss_5 = tf.nn.l2_loss(features - centers_batch_5)

    labels_6 = tf.fill([num_features, ], 6)
    labels_6 = tf.reshape(labels_6, [-1])
    centers_batch_6 = tf.gather(centers, labels_6)
    loss_6 = tf.nn.l2_loss(features - centers_batch_6)

    total_loss = tf.add(loss_0, loss_1)
    total_loss = tf.add(total_loss, loss_2)
    total_loss = tf.add(total_loss, loss_3)
    total_loss = tf.add(total_loss, loss_4)
    total_loss = tf.add(total_loss, loss_5)
    total_loss = tf.add(total_loss, loss_6)


    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)
    # loss
    intra_loss = tf.nn.l2_loss(features - centers_batch)
    inter_loss = tf.subtract(total_loss, intra_loss)
    epsilon1 = 10e-3
    inter_counter_loss = tf.divide(1., tf.add(epsilon1, inter_loss))

    # calculate distance between center points
    centers_point_batch_1 = tf.gather(centers, labels_center_1)
    centers_point_batch_2 = tf.gather(centers, labels_center_2)
    pair_center_loss = tf.nn.l2_loss(centers_point_batch_1 - centers_point_batch_2)

    epsilon2 = 10e-3
    counter_pair_center_loss = tf.divide(1., tf.add(epsilon2, pair_center_loss))
    # difference
    diff = centers_batch - features
    # differences update from center point

    # mini-batch
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1,1])

    diff = diff/ tf.cast((1+appear_times), tf.float32)
    diff = alpha* diff
    centers_update_op = tf.scatter_sub(centers, labels, diff)
    return  intra_loss, inter_counter_loss, counter_pair_center_loss,inter_loss, pair_center_loss,  centers, centers_update_op

def inference(input_images,keep_prob):#with slim.arg_scope([slim.conv2d],padding='SAME',weights_regularizer=slim.l2_regularizer(0.001)):
    input_images1 = tf.reshape(input_images, [-1, image_size, image_size, 1])
    with slim.arg_scope([slim.conv2d], kernel_size=3, padding='SAME'):
        with slim.arg_scope([slim.max_pool2d], kernel_size=2):
            x = slim.conv2d(input_images1, num_outputs=64, scope='conv1_1')
            x = slim.conv2d(x, num_outputs=64, scope='conv1_2')
            x = slim.max_pool2d(x, scope='pool1')

            x = slim.conv2d(x, num_outputs=64, scope='conv2_1')
            x = slim.conv2d(x, num_outputs=64, scope='conv2_2')
            x = slim.max_pool2d(x, scope='pool2')

            x = slim.conv2d(x, num_outputs=128, scope='conv3_1')
            x = slim.conv2d(x, num_outputs=128, scope='conv3_2')
            x = slim.max_pool2d(x, scope='pool3')

            x = slim.flatten(x, scope='flatten')

            x = slim.fully_connected(x, num_outputs=1024, activation_fn=None, scope='fc1')
            x = tflearn.prelu(x)

            feature = slim.fully_connected(x, num_outputs=512, activation_fn=None, scope='fc2')
            x = tflearn.prelu(feature)

            x = slim.dropout(x,keep_prob, scope='dropout2')

            x = slim.fully_connected(x, num_outputs=n_classes, activation_fn=None, scope='fc3')

    return x, feature

def build_network(input_images, labels, ratio, keep_prob):
    logits, features = inference(input_images,keep_prob)

    with tf.name_scope('loss'):
        with tf.name_scope('center_loss'):
            intra_loss, inter_counter_loss, counter_pair_center_loss, inter_loss, pair_center_loss, centers, centers_update_op = get_center_loss(features, labels, CENTER_LOSS_ALPHA, NUM_CLASSES)
        with tf.name_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        with tf.name_scope('total_loss'):
            total_loss = softmax_loss + ratio * (intra_loss + 0.5* inter_counter_loss + 0.5 * counter_pair_center_loss)

    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))

    return logits, features, total_loss,softmax_loss, intra_loss, inter_loss, pair_center_loss, accuracy, centers_update_op


logits, features, total_loss, softmax_loss,intra_loss,inter_loss , pair_center_loss, accuracy, centers_update_op = build_network(input_images,labels, ratio=LAMBDA, keep_prob=keep_prob)

predict = tf.argmax(logits, 1)


def get_result_dd(test_path):
    data_test = load_test(test_path)
    data_test = np.array(data_test)
    # Seesion and Summary
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Train
    h5f = h5py.File('mean_data.h5','r')
    mean_data = h5f['mean_data'][:]

    saver = tf.train.Saver()
    save_path = saver.restore(sess, "./model_alexnet/cohn_kanade_with_proposed_loss_alexnet_fold1.ckpt")
    vali_image = data_test - mean_data
    vali_image = np.reshape(vali_image, [1,4096])
    predict_labels= sess.run(predict, feed_dict={input_images: vali_image, keep_prob:1})
    print(predict_labels)
    print(classes[predict_labels[-1]])
    return classes[predict_labels[-1]]


print(time.time()- start_time)
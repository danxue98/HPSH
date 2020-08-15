#coding:utf-8
import os
import sys

sys.path.append(os.getcwd())
sys.path.append('../')

import time
import matplotlib
matplotlib.use('agg')
import numpy as np
np.random.seed(123)
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from sklearn.manifold import TSNE
import tflib as lib
import tsne_plt
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.plot
import eval2
import pickle
from PIL import Image
import argparse
import sys
from mxnet.gluon import nn
from mxnet import nd
from scipy.spatial.distance import hamming, euclidean
import os, sys

sys.path.append(os.getcwd())
sys.path.append('../')

import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
sns.set()

# specify the GPU where you want to run the code
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

FLAGS = None

NUM_FEATURES = 2048
NUM_CHANNELS = 3
IMG_SIZE = 64
CLASSES = 30  # Number of classes in UCMD
BN_EPSILON = 0.001
beta = 0.001
gamma = 1
isTrain=1
lib.print_model_settings(locals().copy())

# load training data
train_data = np.load('./data/train_data.npy')
#train_labels = np.load('./data/train_single_labels.npy')  #source
train_labels = np.load('./data/train_single_labels.npy', allow_pickle=True)  # i change

# load testing data
test_data = np.load('./data/test_data.npy')
test_labels = np.load('./data/test_single_labels.npy',allow_pickle=True)


def get_triplets2(training_samples, training_labels, nrof_classes):
    classes = nrof_classes
    triplets = np.empty((0, NUM_FEATURES))

    for i in range(classes):
        samples = np.where(training_labels == i)[0]
        np.random.shuffle(samples)
        samples0 = training_samples[samples[0:FLAGS.k], :]
        triplets = np.append(triplets, samples0, axis=0)

    return triplets

def mySamples3(embeddings,d):
    x=embeddings
    k=FLAGS.k
    cutoff = 0.5
    nonzero_loss_cutoff=1.4
    #d=FLAGS.HASH_BITS  #hash bit
    n=FLAGS.BATCH_SIZE
    distance = get_distance1(embeddings, n)
    distance=tf.maximum(distance, cutoff)

    log_weights = ((2.0 - d) * tf.log(distance)
                   - ((d - 3) / 2) * tf.log(1.0 - 0.25 * (tf.square(distance))))

    weights = tf.exp(log_weights - tf.reduce_max(log_weights))
    mask = np.ones(shape=(n, n))
    for i in range(0, n, k):
        mask[i:i+k, i:i+k] = 0
    mask_uniform_probs = mask*(1.0/(n-k))
    # Sample only negative examples by setting weights of
    # the same-class examples to 0.
    ppp=distance<nonzero_loss_cutoff
    ppp=tf.to_float(ppp)
    weights = weights * ppp * mask
    weights_sum = tf.reduce_sum(weights, 1, True)
    weights = weights / weights_sum
    for i in range(n):
        block_idx = i // k
        an_indices=tf.multinomial(weights, k-1)
        for oo in range(k-1):
            ppp=an_indices[i][oo]
            ppp=tf.to_int32(ppp, name='ToInt32')
            if i==0 and oo==0:
                negatives=x[ppp]
            else:
                qqq=x[ppp]
                negatives=tf.concat([negatives, qqq], 0)

        for j in range(block_idx * k, (block_idx + 1) * k):
            if j != i:
                if i==0 and j==1:
                    anchors = x[i]
                    positives = x[j]
                else:
                    anchors = tf.concat([anchors, x[i]], 0)
                    positives = tf.concat([positives, x[j]], 0)
    anchors=tf.reshape(anchors,shape=(-1, d))
    positives = tf.reshape(positives,shape=(-1, d))
    a0=tf.reshape(negatives,shape=(-1, d))
    return anchors, positives, a0

def get_distance1(x,n):
    x=tf.square(x)
    square=tf.reduce_sum(x, 1, keep_dims=True)
    distance_square = square + tf.transpose(square) - (2.0*tf.matmul(x, tf.transpose(x)))
    ones=tf.eye(n)
    return tf.sqrt(distance_square+ones)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def main_network(inputs):
    global isTrain
    output = tf.reshape(inputs, shape=(-1, NUM_FEATURES, 1, 1))
    output = lib.ops.conv2d.Conv2D(name='Classifier.Input', input_dim=NUM_FEATURES, output_dim=1024, filter_size=1, inputs=output, stride=1)
    output = LeakyReLU(output)
    output = lib.ops.conv2d.Conv2D(name='Classifier.2', input_dim=1024, output_dim=512, filter_size=1, inputs=output, stride=1)
    output = LeakyReLU(output)
    output = lib.ops.conv2d.Conv2D(name='Classifier.3', input_dim=512, output_dim=FLAGS.HASH_BITS, filter_size=1, inputs=output, stride=1)
    output_sigmoid = tf.nn.sigmoid(output)

    return tf.reshape(output_sigmoid, shape=[-1, FLAGS.HASH_BITS])

def cosh(x):
    return (tf.exp(x)+tf.exp(-1.0*x))/2.0
def triplet_loss4(embeddings_0, anchors, positives, negatives):
    #hamming loss
    anchors0 = ((tf.sign(anchors - 0.5) + 1) / 2)
    positives0 = ((tf.sign(positives - 0.5) + 1) / 2)
    negatives0 = ((tf.sign(negatives - 0.5) + 1) / 2)
    a0=tf.abs(anchors0-anchors)
    p0=tf.abs(positives0-positives)
    n0=tf.abs(negatives0-negatives)
    loss1=tf.reduce_sum(tf.square(tf.log(cosh(a0)))+tf.square(tf.log(cosh(p0)))+tf.square(tf.log(cosh(n0))))
    _margin=0.3
    # Jointly train class-specific beta.
    beta_margin=0.6
    beta_reg_loss = 0.000000

    d_ap = tf.sqrt(tf.reduce_sum(tf.square(positives - anchors), axis=1) + 1e-8)
    d_an = tf.sqrt(tf.reduce_sum(tf.square(negatives - anchors), axis=1) + 1e-8)
    pos_loss = tf.maximum(d_ap - beta_margin + _margin, 0.0)
    neg_loss = tf.maximum(beta_margin - d_an + _margin, 0.0)
    # aaa=pos_loss > 0.0
    # bbb=neg_loss > 0.0
    # aaa=tf.to_float(aaa)
    # bbb=tf.to_float(bbb)
    # pair_cnt = tf.reduce_sum(aaa+bbb)
    loss = (tf.reduce_mean(pos_loss + neg_loss) + beta_reg_loss)
    loss_2 = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(embeddings_0, 0.5 * tf.ones_like(embeddings_0))), 1))

    # We want to minimize this loss to ensure that the output of each node has nearly 50% chance of being 0 or 1
    loss_3 = tf.reduce_mean(tf.square(tf.reduce_mean(embeddings_0, 1) - 0.5))

    combined_loss = loss - beta*loss_2/FLAGS.HASH_BITS + gamma*loss_3
    return combined_loss+loss1/100


def train(self):
    all_samples = tf.placeholder(tf.float32, shape=[None, NUM_FEATURES])
    sigmoid_activations = main_network(all_samples)
    embeddings = tf.nn.l2_normalize(sigmoid_activations, 1, 1e-10, name='embeddings')

    # Hard negative samples
    anchors, positives, negatives= mySamples3(embeddings, FLAGS.HASH_BITS)

    #LOSS FUNCTION
    loss = triplet_loss4(sigmoid_activations, anchors, positives, negatives) #MARGIN LOSS
    regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([loss] + regularization_loss, name='total_loss')
    ssx=lib.params_with_name('Classifier')
    train_op1 = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
    train_op=train_op1.minimize(loss, var_list=ssx)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        max_map=0.0
        max_epoch=0.
        count=0
        for epoch in range(FLAGS.ITERS):
            print('epoch:', epoch)
            path_list = []
            for i in range(30):
                for j in range(30):
                    triplets = get_triplets2(train_data, train_labels, CLASSES)
                    count=count+1
                    iters=count
                    _, cost = session.run([train_op, total_loss], feed_dict={all_samples: triplets})

                    if (iters < 10) or (iters % 100 == 99):
                        lib.plot.flush()

                    lib.plot.plot('Loss', cost)

                    lib.plot.tick()

            if epoch<35:
                continue
            with open('features_AID.pkl', 'rb') as f:
                features = pickle.load(f)

            test_images = np.empty((0, IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
            test_embeddings = np.empty((0, FLAGS.HASH_BITS)).astype(np.int8)
            test_single_labels = np.empty((0,))

            features_ar = np.empty((0, NUM_FEATURES))
            labels_ar = np.empty((0,))
            paths = []

            if not os.path.exists('dump_dir/'):
                os.mkdir('dump_dir/')

            for idx in range(len(features)):
                feature = features[idx][0]
                label = features[idx][1]
                path = features[idx][2]

                features_ar = np.append(features_ar, np.reshape(feature, newshape=(-1, NUM_FEATURES)), axis=0)
                labels_ar = np.append(labels_ar, [label, ], axis=0)
                paths.append(path)

            start_idx = 0
            for idx_class in range(CLASSES):
                class_label_idxs = np.where(labels_ar == idx_class)[0]
                nrof_samples_in_class = len(class_label_idxs)
                nrof_train_samples = int(np.ceil(FLAGS.train_test_split * nrof_samples_in_class))
                nrof_test_samples = nrof_samples_in_class - nrof_train_samples
                end_idx = start_idx + nrof_samples_in_class

                # test data
                class_features_test = features_ar[class_label_idxs[nrof_train_samples:], :]
                class_labels_test = labels_ar[class_label_idxs[nrof_train_samples:]]
                class_paths_test = []
                for _, idx in enumerate(class_label_idxs[nrof_train_samples:]):
                    class_paths_test.append(paths[idx])

                for i in range(nrof_test_samples):
                    test_input = class_features_test[i,:]
                    test_single_label = class_labels_test[i]
                    test_img_path = class_paths_test[i]

                    # Store the embedding
                    test_embedding0 = session.run(sigmoid_activations, feed_dict={all_samples: np.reshape(test_input, newshape=(-1, NUM_FEATURES))})
                    test_embedding = ((np.sign(test_embedding0 - 0.5) + 1) / 2)
                    test_embeddings = np.append(test_embeddings, test_embedding.astype(np.int8), axis=0)

                    # Store the label
                    test_single_labels = np.append(test_single_labels, [test_single_label, ], axis=0)

                    # Store the image
                    img = Image.open(test_img_path)
                    path_list.append(test_img_path)
                    img = img.resize([IMG_SIZE, IMG_SIZE], Image.ANTIALIAS)
                    img.save(os.path.join('dump_dir', 'temp.jpg'))

                    read_image_path = os.path.join('dump_dir', 'temp.jpg')
                    img = np.array(Image.open(read_image_path), dtype=int) / 256.
                    img = np.reshape(img, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
                    test_images = np.append(test_images, img, axis=0)

                start_idx = end_idx

            out_dir = 'data/'
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            np.save('data/test_labels.npy', test_single_labels)
            np.save('data/test_embeddings.npy', test_embeddings)
            np.save('data/test_images.npy', test_images)
            map = eval2.eval(FLAGS.HASH_BITS)
            if map > max_map:
                max_map = map
                max_epoch = epoch
        print("max_map:", max_map)
        print("max_epoch", max_epoch)
        print('Optimization finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--HASH_BITS',
        type=int,
        default=32,
        help="Hash-bit length (K)"
    )
    parser.add_argument(
        '--ALPHA',
        type=float,
        default=0.2,
        help="The alpha separation between the positive and negative samples"
    )
    parser.add_argument(
        '--BATCH_SIZE',
        type=int,
        default=150,
        help="The number of samples in a mini-batch of triplets. Must be divisible by 3."
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help="the number of batch_k"
    )
    parser.add_argument(
        '--ITERS',
        type=int,
        default=40,
        help="Number of training iterations."
    )
    parser.add_argument(
        '--train_test_split',
        type=float,
        default=0.6,
        help="Train test split ratio."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=train, argv=[sys.argv[0]] + unparsed)

    train()
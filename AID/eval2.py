import os, sys

sys.path.append(os.getcwd())
sys.path.append('../')
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import operator
import warnings
import argparse
warnings.filterwarnings("ignore")
import tensorflow as tf
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import hamming
import seaborn as sns
sns.set()

FLAGS = None
IMG_SIZE = 64
NUM_CHANNELS = 3
CLASSES = 30
class_names = ['airport', 'bareland', 'baseball\nfield', 'beach', 'bridge', 'center',
               'church', 'commercial', 'dense\nresidential', 'desert', 'farmland', 'forest',
               'industrial', 'meadow', 'medium\nresidential', 'mountain', 'park', 'parking',
               'playground', 'pond', 'port', 'railway\nstation', 'resort', 'river', 'school',
               'sparse\nresidential', 'square', 'stadium', 'storage\ntanks', 'viaduct']


def hamming_distance(instance1, instance2):
    return hamming(instance1, instance2)

def get_k_hamming_neighbours(nrof_neighbors, enc_test, test_img, test_lab, index):
    _neighbours = []  # 1(query image) + nrof_neighbours
    distances = []
    test_data = np.load('./data/test_images.npy')
    test_labels = np.load('./data/test_labels.npy')
    test_encodings = np.load('./data/test_embeddings.npy')
    for i in range(len(test_encodings)):
        if index != i:  # exclude the test instance itself from the search set
            dist = hamming_distance(test_encodings[i, :], enc_test)
            distances.append((test_data[i, :, :, :], test_labels[i], dist, test_encodings[i,:]))

    distances.sort(key=operator.itemgetter(2))
    for j in range(nrof_neighbors):
        _neighbours.append((distances[j][0], distances[j][1]))

    return _neighbours


def top_k_accuracy(neghbours_list, nrof_neighbors):
    test_sample_label = neghbours_list[0][0]
    true_label_vector = [test_sample_label] * nrof_neighbors
    pred_label_vector = []

    for i in range(1, nrof_neighbors + 1):
        pred_label_vector.append(neghbours_list[i][1])

    accuracy = accuracy_score(y_true=true_label_vector, y_pred=pred_label_vector)
    return accuracy

def get_mAP(neghbours_list, nrof_neighbors):
    test_sample_label = neghbours_list[0][1]
    acc = np.empty((0,)).astype(np.float)
    correct = 1
    for i in range(1, nrof_neighbors):
        if test_sample_label == neghbours_list[i][1]:
            precision = (correct / float(i))
            acc = np.append(acc, [precision, ], axis=0)
            correct += 1
    if correct == 1:
        return 0.
    num = np.sum(acc)
    den = correct - 1
    return num/den
def rmse(a,b,hash_bits):
    return np.sum(((a-b)*(a-b)/hash_bits)**0.5)
def get_error(neghbours_list, nrof_neighbors,hash_bits):
    test_sample_label = neghbours_list[0][1]
    errors = np.empty((0,)).astype(np.float)
    wrong = 1
    for i in range(1, nrof_neighbors):
        if test_sample_label != neghbours_list[i][1]:
            ppp=rmse(neghbours_list[0][2],neghbours_list[i][2],hash_bits)
            error = (ppp / float(i))
            errors = np.append(errors, [error, ], axis=0)
            wrong += 1
    if wrong == 1:
        return 0.
    num = np.sum(errors)
    den = wrong - 1
    return num

def plot_or_save_singlelabel_images(_neighbours, filename=''):
    # create figure with sub-plots
    fig, axes = plt.subplots(7, 3, figsize=(6,6), squeeze=False)
    # adjust vertical spacing if we need to print ensemble and best-net
    fig.subplots_adjust(hspace=0.6, wspace=0.1)
    k=21
    query_label = _neighbours[0][1]
    acc = round(get_mAP(_neighbours, k), 4)
    acc_str = str(acc * 100) + "%"
    for i, ax in enumerate(axes.flat):
        # Plot image
        img = np.reshape(_neighbours[i][0], newshape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
        #img = img.transpose(1, 2, 0)
        ax.imshow(img, cmap='gray')
        # Name of the true class
        cls_true_label = _neighbours[i][1]
        cls_true_name = class_names[int(cls_true_label)]

        # Show the true and predicted classes
        if i == 0:
            xlabel = "{0}".format(class_names[int(query_label)])
            #ylabel = acc_str
            ylabel = "Query"
            ax.set_ylabel(ylabel)
        else:
            # name of the predicted class
            xlabel = "({0}) {1}".format(i, cls_true_name)
        # show the classes as the label on the x-axis
        ax.set_xlabel(xlabel)
        # show the accuracy on y-axis
        # remove the ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])
        if query_label != cls_true_label:
            ax.spines['left'].set_color('red')
            ax.spines['right'].set_color('red')
            ax.spines['top'].set_color('red')
            ax.spines['bottom'].set_color('red')
            ax.spines['left'].set_linewidth(3)
            ax.spines['right'].set_linewidth(3)
            ax.spines['top'].set_linewidth(3)
            ax.spines['bottom'].set_linewidth(3)

        if i == 0:
            ax.spines['left'].set_color('blue')
            ax.spines['right'].set_color('blue')
            ax.spines['top'].set_color('blue')
            ax.spines['bottom'].set_color('blue')
            ax.spines['left'].set_linewidth(3)
            ax.spines['right'].set_linewidth(3)
            ax.spines['top'].set_linewidth(3)
            ax.spines['bottom'].set_linewidth(3)

    plt.savefig( filename, bbox_inches='tight')
    return

def get_precision_recall(neghbours_list, nrof_neighbors):
    test_sample_label = neghbours_list[0][1]
    # acc = np.empty((0,)).astype(np.float)
    correct = 0
    for i in range(1, nrof_neighbors):
        if test_sample_label == neghbours_list[i][1]:
            correct = correct +  1
    if correct == 0:
        return 0.
    precision = correct / 20
    # recall = correct/39
    return precision
def eval(hash_bits):
    total_map = 0.
    total_error =0.
    total_precision = 0.
    k=21
    out_dir = 'hamming_out/'
    test_data = np.load('./data/test_images.npy')
    test_labels = np.load('./data/test_labels.npy')
    test_encodings = np.load('./data/test_embeddings.npy')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for idx in range(len(test_encodings)):

        test_encoding = test_encodings[idx, :]
        test_image = test_data[idx, :, :, :]
        test_label = test_labels[idx]
        neighbours = get_k_hamming_neighbours(nrof_neighbors=k, enc_test=test_encoding, test_img=test_image, test_lab=test_label, index=idx)
        total_map += get_mAP(neighbours, k)
        #total_error += get_error(neighbours, k, hash_bits)
        precision= get_precision_recall(neighbours, k)
        total_precision += precision


    print('mAP@20:{0}'.format((total_map / len(test_encodings)) * 100))
    # print('precision@20:{0}'.format((total_precision / len(test_encodings)) * 100))
    # print('error:{0}'.format((total_error / len(test_encodings)) * 100))
    return (total_map/len(test_encodings))*100
#coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import cv2
import  os
from sklearn.manifold import TSNE; HAS_SK = True
IMG_SIZE = 7
import matplotlib.image as mping
from matplotlib.offsetbox import OffsetImage,AnnotationBbox
def plot_with_labels(lowDWeights, labels,path ,epoch):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    plt.axis('off')
    imscatter(lowDWeights[:, 0], lowDWeights[:, 1], path, zoom=0.1, ax=ax)
    plt.savefig('imgs/tsne{}.png'.format(epoch))
    plt.show()

def imscatter(x, y, images, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, image in zip(x, y, images):
        im = cv2.imread(image)
        im = cv2.resize(im, (256, 256))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_f = OffsetImage(im, zoom=zoom)
        ab = AnnotationBbox(im_f, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


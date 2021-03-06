from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import numpy as np
import time
import functools
import time
import cv2
import matplotlib.pyplot as plt
from webcam import *
from tf_style import *
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('-p', action='store', required = True, dest='cont_path',
                    help='Path to Content Image')

parser.add_argument('-i', action='store', required = True, dest='server_ip',
                    help='IP address of mobile server')

parser.add_argument('-m', action='store', required = False, dest='mode',
                    help='mode: s for static or d for dynamic', default='s')

parser.add_argument('-e', action='store', required = False, dest='epochs',
                    help='number of style transfer epochs to run', default=10)

results = parser.parse_args()

def static_style():

    style_image = get_image_from_device(ip = results.server_ip, imsize = (512,512))[None, ...]

    content_image = (cv2.resize(cv2.cvtColor(cv2.imread(results.cont_path), cv2.COLOR_BGR2RGB), (512, 512))/255).astype(np.float32)[None, ...]

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    # Content layer where we will pull our feature maps
    content_layers = ['block5_conv2'] 

    # Style layers of interest
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']

    S = Styler(vgg, content_image, style_image, content_layers, style_layers)


    style_targets = S.StyleContentModel(style_image)['style']
    content_targets = S.StyleContentModel(content_image)['content']

    image = tf.Variable(content_image)

    def clip_0_1(image):
      return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


    opt = tf.train.AdamOptimizer(learning_rate=0.02, beta1=0.99, epsilon=1e-1)

    style_weight=1e-2 
    content_weight=1e4

    total_variation_weight=100

    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = S.StyleContentModel(image)
            loss = S.style_content_loss(outputs, style_targets, content_targets)
            loss += total_variation_weight*tf.image.total_variation(image)


        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    start = time.time()

    epochs = 10
    steps_per_epoch = 25
    print("\nImage captured from device, max, value = ", np.ptp(style_image))

    print('\n---------Running Style Transfer----------')
    step = 0
    for n in tqdm(range(results.epochs)):
      for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print(".", end='')


    imshow(image.read_value())
    plt.title("Train step: {}".format(step))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.savefig(results.cont_path + 'style.jpg')
    end = time.time()
    print("Total time: {:.1f}".format(end-start))


def dynamic_style():

    content_image = (cv2.resize(cv2.cvtColor(cv2.imread(results.cont_path), cv2.COLOR_BGR2RGB), (512, 512))/255).astype(np.float32)[None, ...]

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    # Content layer where will pull our feature maps
    content_layers = ['block5_conv2'] 

    # Style layer of interest
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']

    def clip_0_1(image):
      return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


    opt = tf.train.AdamOptimizer(learning_rate=0.02, beta1=0.99, epsilon=1e-1)

    style_weight=1e-2 
    content_weight=1e4

    total_variation_weight=100

    image = tf.Variable(content_image)

    def capture_image():

        style_image = get_image_from_device(ip = results.server_ip, imsize = (512,512))[None, ...]

        print("\nImage captured from device, max, value = ", np.ptp(style_image))

        return style_image

    def train_step(image, style_image):

        S = Styler(vgg, content_image, style_image, content_layers, style_layers)

        style_targets = S.StyleContentModel(style_image)['style']
        content_targets = S.StyleContentModel(content_image)['content']

        with tf.GradientTape() as tape:
            outputs = S.StyleContentModel(image)
            loss = S.style_content_loss(outputs, style_targets, content_targets)
            loss += total_variation_weight*tf.image.total_variation(image)


        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    start = time.time()

    epochs = 10
    steps_per_epoch = 25

    print('\n---------Running Style Transfer----------')
    step = 0
    for n in tqdm(range(results.epochs)):
        style_image = capture_image()
        for m in range(steps_per_epoch):
            step += 1
            train_step(image, style_image)
            print(".", end='')


    imshow(image.read_value())
    plt.title("Train step: {}".format(step))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.savefig(results.cont_path + 'style.jpg')
    end = time.time()
    print("Total time: {:.1f}".format(end-start))

if results.mode == 's':
    static_style()
else:
    dynamic_style()
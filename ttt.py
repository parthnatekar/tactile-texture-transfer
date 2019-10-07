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

# Load images
content_path = tf.keras.utils.get_file('turtle.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg')
content_image = load_img(content_path)
#style_image = cv2.imread('pebbles.jpeg')
#style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
#style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
#style_image = load_img(style_path)

style_image = get_image_from_device(imsize = (512,512))[None, ...]

print("Image captured from device, max, value = ", np.ptp(style_image))

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')


vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer of interest
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


def train_step(image):
	with tf.GradientTape() as tape:
		outputs = S.StyleContentModel(image)
		loss = S.style_content_loss(outputs, style_targets, content_targets)

	grad = tape.gradient(loss, image)
	opt.apply_gradients([(grad, image)])
	image.assign(clip_0_1(image))

start = time.time()

epochs = 10
steps_per_epoch = 25

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
    print(".", end='')

  imshow(image.read_value())
  plt.title("Train step: {}".format(step))
  plt.show()

end = time.time()
print("Total time: {:.1f}".format(end-start))

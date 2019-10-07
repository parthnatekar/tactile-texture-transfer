import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import time
import functools
import cv2

	
#content_path = tf.keras.utils.get_file('turtle.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg')
#style_path = tf.keras.utils.get_file('kandinsky.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

#content_image = cv2.imread(content_path)
#style_image = cv2.imread(style_path)

#x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
#x = tf.image.resize_images(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
# prediction_probabilities = vgg(x[None, ...])
# prediction_probabilities.shape
# print(vgg.inputs)
#print(vgg.input.shape)
#predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities)[0]
#[(class_name, prob) for (number, class_name, prob) in predicted_top_5]

def intermediate_extractor(layers):

		outputs = []

		for layer_name in layers:
			outputs.append(vgg.get_layer(layer_name).output)

		model = tf.keras.Model([vgg.inputs[0]], outputs)
		print(model.summary())

		return(model)

def gram_style(input_tensor):
  		result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  		input_shape = tf.shape(input_tensor)
  		num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  		return result/(num_locations)


class StyleContentModel(tf.keras.models.Model):
	
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg =  intermediate_extractor(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = tf.cast(inputs,dtype=tf.float32)
    inputs = tf.math.multiply(inputs, 255.0)
    
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    preprocessed_input = tf.image.resize_images(preprocessed_input, (224, 224))[None, ...]
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_style(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name:value 
                    for content_name, value 
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}
    
    return {'content':content_dict, 'style':style_dict}


class Styler():

	def __init__(self, model):
		self.model = model
		self.model.trainable = False

	
	def style_content_creator(self, style_layers, content_layers, style_image, content_image):

		style_model = self.intermediate_extractor(style_layers+content_layers)

		#style_image = tf.cast(style_image,dtype=tf.float32)
		#content_image = tf.cast(content_image,dtype=tf.float32)

		style_inputs = tf.math.multiply(style_image, 255.0)
		content_inputs = tf.math.multiply(content_image, 255.0)
		preprocessed_style = tf.keras.applications.vgg19.preprocess_input(style_inputs)
		preprocessed_content = tf.keras.applications.vgg19.preprocess_input(content_inputs)
		preprocessed_style = tf.image.resize_images(preprocessed_style, (224, 224))
		preprocessed_content = tf.image.resize_images(preprocessed_content, (224, 224))

		#print(tf.Variable(preprocessed_style[None, ...]	))
		#print(style_model(tf.Variable(preprocessed_style[None, ...])))

		#print(style_model(tf.constant(preprocessed_style)))

		style_outputs = [style_output for style_output in style_model(tf.Variable(preprocessed_style[None, ...]))[:len(style_layers)]]
		content_outputs = [content_output for content_output in style_model(tf.Variable(preprocessed_content[None, ...]))[len(style_layers):]]

		style_representations = {}

		content_representations = {}

		for index, style_layer in enumerate(style_layers):

			style_representations[style_layer] = style_outputs[index]

		for index, content_layer in enumerate(content_layers):

			content_representations[content_layer] = content_outputs[index]

		return(style_representations, content_representations)

		
	def optimize(self):

		content_path = tf.keras.utils.get_file('turtle.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg')
		style_path = tf.keras.utils.get_file('kandinsky.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

		content_image = cv2.imread(content_path)
		style_image = cv2.imread(style_path)

		image = tf.constant(content_image)

		def clip_0_1(image):
			return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

		opt = tf.train.AdamOptimizer(learning_rate=0.02, beta1=0.99, epsilon=1e-1)
		style_weight=1e-2
		content_weight=1e4

		style_loss, content_loss = 0, 0

		style_layers = ['block1_conv1',
		'block2_conv1',
		'block3_conv1', 
		'block4_conv1', 
		'block5_conv1']

		content_layers = ['block5_conv2']

		#style_targets, content_targets = self.style_content_creator(style_layers, content_layers, style_image, content_image)

		extractor = StyleContentModel(style_layers, content_layers)

		style_targets = extractor(style_image)['style']
		content_targets = extractor(content_image)['content']

		with tf.GradientTape() as tape:
			#outputs = self.style_content_creator(style_layers, content_layers, image, image)
			outputs = extractor(image)

			style_loss = tf.add_n([tf.reduce_mean(style_targets[key] - outputs['style'][key])**2 for key in style_targets.keys()])

			content_loss = tf.add_n([tf.reduce_mean(content_targets[key] - outputs['content'][key])**2 for key in content_targets.keys()])

			print(tf.trainable_variables()) 

			style_loss *= style_weight / len(style_layers)
			content_loss *= content_weight / len(content_layers)

			loss = style_loss+content_loss

			grad = tape.gradient(loss, image)
			opt.apply_gradients([(grad, image)])
			image.assign(clip_0_1(image))

		
if __name__ == '__main__':

	S = Styler(vgg)

	S.optimize()





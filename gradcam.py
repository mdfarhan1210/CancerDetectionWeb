# import the necessary packages
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
class GradCAM:
	def __init__(self, model, classIdx, layerName=None):
		# store the model, the class index used to measure the class
		# activation map, and the layer to be used when visualizing
		# the class activation map
		self.model = model
		self.classIdx = classIdx
		self.layerName = layerName
		# if the layer name is None, attempt to automatically find
		# the target output layer
		if self.layerName is None:
			self.layerName = self.find_target_layer()
	def find_target_layer(self):
		# attempt to find the final convolutional layer in the network
		# by looping over the layers of the network in reverse order
		for layer in reversed(self.model.layers):
       		# Check if the layer is a convolutional layer by checking its output shape
			if isinstance(layer, tf.keras.layers.Conv2D):
				return layer.name
		# otherwise, we could not find a 4D layer so the GradCAM
		# algorithm cannot be applied
		raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


	def compute_heatmap(self, image, eps=1e-8):
		# Construct the gradient model
		gradModel = Model(
			inputs=[self.model.inputs],
			outputs=[self.model.get_layer(self.layerName).output, self.model.output]
		)

		# Record operations for automatic differentiation
		with tf.GradientTape() as tape:
			# Cast the image tensor to float32 and pass it through the gradient model
			inputs = tf.cast(image, tf.float32)
			(convOutputs, predictions) = gradModel(inputs)
			
			# Ensure predictions is a tensor, then extract the loss for the given class index
			predictions = tf.convert_to_tensor(predictions)
			# loss = predictions[:, self.classIdx]
			loss = predictions[0]
			
		# Compute gradients of the loss with respect to the output feature map
		grads = tape.gradient(loss, convOutputs)
		
		# Compute guided gradients
		castConvOutputs = tf.cast(convOutputs > 0, "float32")
		castGrads = tf.cast(grads > 0, "float32")
		guidedGrads = castConvOutputs * castGrads * grads

		# Remove batch dimension
		convOutputs = convOutputs[0]
		guidedGrads = guidedGrads[0]

		# Compute the average of the gradient values as weights
		weights = tf.reduce_mean(guidedGrads, axis=(0, 1))

		# Compute the class activation map
		cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

		# Resize heatmap to match the input image
		(w, h) = (image.shape[2], image.shape[1])
		heatmap = cv2.resize(cam.numpy(), (w, h))

		# Normalize the heatmap to [0, 1]
		numer = heatmap - np.min(heatmap)
		denom = (heatmap.max() - heatmap.min()) + eps
		heatmap = numer / denom

		# Convert heatmap to an unsigned 8-bit integer
		heatmap = (heatmap * 255).astype("uint8")

		return heatmap
	def overlay_heatmap(self, heatmap, image, alpha=0.5,
		colormap=cv2.COLORMAP_VIRIDIS):
		# apply the supplied color map to the heatmap and then
		# overlay the heatmap on the input image
		heatmap = cv2.applyColorMap(heatmap, colormap)
		output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
		# return a 2-tuple of the color mapped heatmap and the output,
		# overlaid image
		return (heatmap, output)
		
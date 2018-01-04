import tensorflow as tf
import numpy as np
import os
import time
import cv2 as cv
import h5py
import scipy.misc
import scipy.ndimage
from prepare_test import im2double, modcrop, revert
from net import SRCNN, load_data, load_ckpt
import math

# Initialization
size_input = 33
size_label = 21
images = tf.placeholder(tf.float32, [None, size_input, size_input, 1], name='images')
labels = tf.placeholder(tf.float32, [None, size_label, size_label, 1], name='labels')
learning_rate = 1e-4
num_epoch = 15000
batch_size = 128
num_training = 21712
num_testing = 1113
ckpt_dir = './checkpoint/'
multiplier = 3

# Read and prepare the test image for SRCNN.
def prepare_data(path):
	# Settings.
	data = []
	label = []
	padding = abs(size_input - size_label) / 2
	scale = 3
	stride = 21
	# Read in image and convert to ycrcb color space.
	img = cv.imread(path)
	img = cv.cvtColor(img, cv.COLOR_RGB2YCR_CB)
	img = im2double(img) # Only use the luminance value.

	# Create groundtruth and baseline image.
	im_label = modcrop(img)
	size = im_label.shape
	h = size[0]
	w = size[1]
	im_temp = scipy.misc.imresize(im_label, 1/scale, interp='bicubic')
	im_input = scipy.misc.imresize(im_temp, size, interp='bicubic')

	# Generate subimages.
	for x in range(0, h - size_input, stride):
		for y in range(0, w - size_input, stride):
			subim_input = im_input[x : x + size_input, y : y + size_input]
			subim_label = im_label[int(x + padding) : int(x + padding + size_label), int(y + padding) : int(y + padding + size_label)]
			
			subim_input = subim_input.reshape([size_input, size_input, 1])
			subim_label = subim_label.reshape([size_label, size_label, 1])

			data.append(subim_input)
			label.append(subim_label)

	data = np.array(data)
	label = np.array(label)

	# Write to HDF5 file.
	savepath = os.path.join(os.getcwd(), 'checkpoint/test_image.h5')
	with h5py.File(savepath, 'w') as hf:
		hf.create_dataset('data', data=data)
		hf.create_dataset('label', data=label)

	return savepath

# Prepare original data without blurring.
def prepare_raw(path):
	# Settings.
	data = []
	color = []
	padding = abs(size_input - size_label) / 2
	stride = 21
	# Read in image and convert to ycrcb color space.
	img = cv.imread(path)
	im = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
	img = im2double(im) # Only use the luminance value.

	# Create groundtruth and baseline image.
	# im_label = modcrop(img)
	# size = im_label.shape
	size = img.shape
	# im_label = scipy.misc.imresize(im_label, [size[0] * 3, size[1] * 3], interp='bicubic')
	img_temp = scipy.misc.imresize(img, [size[0] * multiplier, size[1] * multiplier], interp='bicubic')
	color_temp = scipy.misc.imresize(im, [size[0] * multiplier, size[1] * multiplier], interp='bicubic')
	# img_temp = scipy.ndimage.interpolation.zoom(img, 3.0, prefilter=False)
	im_label = img_temp[:, :, 0]
	im_color = color_temp[:, :, 1:3]
	h = im_label.shape[0]
	w = im_label.shape[1]

	# Generate subimages.
	for x in range(0, h - size_input, stride):
		for y in range(0, w - size_input, stride):
			subim_input = im_label[x : x + size_input, y : y + size_input]
			subim_color = im_color[int(x + padding) : int(x + padding + size_label), int(y + padding) : int(y + padding + size_label), :]
			# subim_label = im_label[int(x + padding) : int(x + padding + size_label), int(y + padding) : int(y + padding + size_label)]
			
			subim_input = subim_input.reshape([size_input, size_input, 1])
			subim_color = subim_color.reshape([size_label, size_label, 2])
			# subim_label = subim_label.reshape([size_label, size_label, 1])

			data.append(subim_input)
			color.append(subim_color)
			# label.append(subim_label)

	data = np.array(data)
	color = np.array(color)
	# label = np.array(label)

	# Write to HDF5 file.
	# savepath = os.path.join(os.getcwd(), 'checkpoint/test_raw_image.h5')
	# with h5py.File(savepath, 'w') as hf:
	# 	hf.create_dataset('data', data=data)
	# 	hf.create_dataset('color', data=color)
		# hf.create_dataset('label', data=label)

	return data, color


# Use the trained model to generate super-resolutioned image.
def generate_SR(x, num_ver, num_hor, path, save_path):
	# Initialization.
	model = SRCNN(x)
	l2_loss = tf.reduce_mean(tf.square(labels - model))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(l2_loss)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print('Generating super-resolutioned image...')

		# Load the saved checkpoint.
		saver = tf.train.Saver()
		if load_ckpt(sess, ckpt_dir, saver):
			print('Successfully loaded checkpoint.')
		else:
			print('Failed to load checkpoint.')

		test_data, test_label = load_data(prepare_data(path))
		print('test_data has shape:', test_data.shape, 'label has shape:', test_label.shape)

		# Generate super-resolutioned image.
		conv_out = model.eval({images: test_data, labels: test_label})	# Result in patch of size 21x21.
		height, width = conv_out.shape[1], conv_out.shape[2]
		print('conv_out has shape:', conv_out.shape)
		result = np.zeros([height * num_hor, width * num_ver, 1])
		original = np.zeros([height * num_hor, width * num_ver, 1])
		print('result has shape:', result.shape)
		print('num_hor =', num_hor, 'num_ver =', num_ver)
		i, j = 0, 0
		for idx, image in enumerate(conv_out):
			j = idx // num_ver
			i = idx - j * num_ver
			print('idx =', idx, 'i =', i, 'j =', j)
			result[j * height : j * height + height, i * width : i * width + width, :] = image
		result = result.squeeze()
		result = revert(result)

		i, j = 0, 0
		for idx, image in enumerate(test_label):
			j = idx // num_ver
			i = idx - j * num_ver
			original[j * height : j * height + height, i * width : i * width + width, :] = image
		original = original.squeeze()

		size_original = original.shape
		bicubic = scipy.misc.imresize(original, 1/3, interp='bicubic')
		bicubic = scipy.misc.imresize(bicubic, size_original, interp='bicubic')

		# Display and save the image.
		cv.imshow('original', original)
		cv.waitKey(0)
		cv.imshow('bicubic', bicubic)
		cv.waitKey(0)
		cv.imshow('super-resolution', result)
		cv.waitKey(0)
		# save_path = os.path.join('./result', 'test.png')
		scipy.misc.imsave(save_path, result)
		# scipy.misc.imsave('./result/original.png', original)
		scipy.misc.imsave('./result/bicubic.png', bicubic)

# Directly feed the original image to SRCNN.
def enhance(x, path, save_dir):
	# Initialization.
	model = SRCNN(x)
	l2_loss = tf.reduce_mean(tf.square(labels - model))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(l2_loss)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print('Generating super-resolutioned image...')

		# Load the saved checkpoint.
		saver = tf.train.Saver()
		if load_ckpt(sess, ckpt_dir, saver):
			print('Successfully loaded checkpoint.')
		else:
			print('Failed to load checkpoint.')

		if os.path.isfile(path):
			print('Upscaling image', path, '...')
			img = cv.imread(path)
			h, w = img.shape[0], img.shape[1]
			num_hor = math.ceil((h * multiplier - size_input) / 21)
			num_ver = math.ceil((w * multiplier - size_input) / 21)

			test_data, test_color = prepare_raw(path)
			# with h5py.File(data_path, 'r') as hf:
			# 	test_data = np.array(hf.get('data'))
			# 	test_color = np.array(hf.get('color'))

			# Generate super-resolutioned image.
			conv_out = model.eval({images: test_data})	# Result in patch of size 21x21.
			height, width = conv_out.shape[1], conv_out.shape[2]
			# print('conv_out has shape:', conv_out.shape)
			result = np.zeros([height * num_hor, width * num_ver, 3])
			# original = np.zeros([height * num_hor, width * num_ver, 1])
			# print('result has shape:', result.shape)
			# print('num_hor =', num_hor, 'num_ver =', num_ver)
			i, j = 0, 0
			for idx, image in enumerate(conv_out):
				j = idx // num_ver
				i = idx - j * num_ver
				# print('idx =', idx, 'i =', i, 'j =', j)
				result[j * height : j * height + height, i * width : i * width + width, 0] = image.squeeze()
			# result = result.squeeze()
			result = revert(result)

			print('Filling color information...')
			i, j = 0, 0
			for idx, image in enumerate(test_color):
				j = idx // num_ver
				i = idx - j * num_ver
				result[j * height : j * height + height, i * width : i * width + width, 1 : 3] = image
			
			result = cv.cvtColor(result, cv.COLOR_YCrCb2RGB)
			# cv.imshow('super-resolution', result)
			# cv.waitKey(0)
			# save_path = os.path.join('./result', 'test_raw.png')
			save_path = os.path.join(save_dir, os.path.basename(path))
			scipy.misc.imsave(save_path, result)
			print('Finished upscaling image', path)

		elif os.path.isdir(path):
			for root, dirs, files in os.walk(path):
				for im_name in files:
					img_path = os.path.join(path, im_name)
					print('Upscaling image', img_path, '...')
					img = cv.imread(img_path)
					h, w = img.shape[0], img.shape[1]
					num_hor = math.ceil((h * multiplier - size_input) / 21)
					num_ver = math.ceil((w * multiplier - size_input) / 21)

					test_data, test_color = prepare_raw(img_path)

					# Generate super-resolutioned image.
					conv_out = model.eval({images: test_data})	# Result in patch of size 21x21.
					height, width = conv_out.shape[1], conv_out.shape[2]
					# print('conv_out has shape:', conv_out.shape)
					result = np.zeros([height * num_hor, width * num_ver, 3])
					# original = np.zeros([height * num_hor, width * num_ver, 1])
					# print('result has shape:', result.shape)
					# print('num_hor =', num_hor, 'num_ver =', num_ver)
					i, j = 0, 0
					for idx, image in enumerate(conv_out):
						j = idx // num_ver
						i = idx - j * num_ver
						# print('idx =', idx, 'i =', i, 'j =', j)
						result[j * height : j * height + height, i * width : i * width + width, 0] = image.squeeze()
					# result = result.squeeze()
					result = revert(result)

					print('Filling color information...')
					i, j = 0, 0
					for idx, image in enumerate(test_color):
						j = idx // num_ver
						i = idx - j * num_ver
						result[j * height : j * height + height, i * width : i * width + width, 1 : 3] = image
					
					result = cv.cvtColor(result, cv.COLOR_YCrCb2RGB)
					# cv.imshow('super-resolution', result)
					# cv.waitKey(0)
					# save_path = os.path.join('./result', 'test_raw.png')
					save_path = os.path.join(save_dir, os.path.basename(img_path))
					scipy.misc.imsave(save_path, result)
					print('Finished upscaling image', img_path)
			print('Finished upscaling all images.')
		else:
			print(' [*] Invalid input path.')


# Calculate num_ver and num_hor.
img_path = './Test/Set14/baboon.bmp'
# save_path = os.path.join('./result', 'test_raw.png')
save_path = './result/'
# img = cv.imread(img_path)
# print('original size =', img.shape)
# h, w = img.shape[0], img.shape[1]
# num_hor = math.ceil((h - size_input) / 21)
# num_ver = math.ceil((w - size_input) / 21)

# enh_num_hor = math.ceil((h * multiplier - size_input) / 21)
# enh_num_ver = math.ceil((w * multiplier - size_input) / 21)

# generate_SR(images, num_ver, num_hor, img_path, save_path)
enhance(images, img_path, save_path)


# def upscale_batch(input_dir, output_dir):
# 	# Traverse the images in the input_dir.
# 	print('input:', input_dir, 'output:', output_dir)
# 	for root, dirs, files in os.walk(input_dir):
# 		for im_name in files:
# 			img_path = os.path.join(input_dir, im_name)
# 			print('Upscaling image', img_path)

# 			# Calculate number of patch needed horizontally and vertically.
# 			img = cv.imread(img_path)
# 			h, w = img.shape[0], img.shape[1]
# 			hor = math.ceil((h * multiplier - size_input) / 21)
# 			ver = math.ceil((w * multiplier - size_input) / 21)

# 			save_path = os.path.join(output_dir, im_name)
# 			enhance(images, ver, hor, img_path, save_path)


# input_dir = './Test/Set5/'
# output_dir = './result/batch_result/'
# upscale_batch(input_dir, output_dir)




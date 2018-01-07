import tensorflow as tf
import numpy as np
import os
import time
import cv2 as cv
import h5py
import scipy.misc
import scipy.ndimage
from prepare_test import im2double, modcrop, revert, modcrop_color
from net import SRCNN, load_data, load_ckpt
import math

# Initialization
size_input = 33
size_label = 21
# images = tf.placeholder(tf.float32, [1536, 1536], name='images')
# labels = tf.placeholder(tf.float32, [None, size_label, size_label, 1], name='labels')
learning_rate = 1e-4
num_epoch = 15000
batch_size = 128
num_training = 21712
num_testing = 1113
ckpt_dir = './checkpoint/'
multiplier = 2

# Read and prepare the test image for SRCNN.
def prepare_data(path):
	# Settings.
	data = []
	label = []
	padding = abs(size_input - size_label) / 2
	stride = 21
	# Read in image and convert to ycrcb color space.
	img_input = cv.imread(path)
	im = cv.cvtColor(img_input, cv.COLOR_BGR2YCR_CB)
	img = im2double(im) # Only use the luminance value.

	# Create groundtruth and baseline image.
	im_label = modcrop_color(img, scale=multiplier)
	color_base = modcrop_color(im, scale=multiplier)
	size = im_label.shape
	h = size[0]
	w = size[1]
	im_blur = scipy.misc.imresize(im_label, 1 / multiplier, interp='bicubic')
	im_input = scipy.misc.imresize(im_blur, multiplier * 1.0, interp='bicubic')

	# print('im_temp shape:', im_temp.shape)
	# print('im_input shape:', im_input.shape)

	# Generate subimages.
	# for x in range(0, h - size_input, stride):
	# 	for y in range(0, w - size_input, stride):
	# 		subim_input = im_input[x : x + size_input, y : y + size_input]
	# 		subim_label = im_label[int(x + padding) : int(x + padding + size_label), int(y + padding) : int(y + padding + size_label)]
			
	# 		subim_input = subim_input.reshape([size_input, size_input, 1])
	# 		subim_label = subim_label.reshape([size_label, size_label, 1])

	# 		data.append(subim_input)
	# 		label.append(subim_label)

	data = np.array(im_input[:,:,0]).reshape([1, h, w, 1])
	color = np.array(color_base[:,:,1:3])
	label = np.array(modcrop_color(img_input))

	# Write to HDF5 file.
	# savepath = os.path.join(os.getcwd(), 'checkpoint/test_image.h5')
	# with h5py.File(savepath, 'w') as hf:
	# 	hf.create_dataset('data', data=data)
	# 	hf.create_dataset('label', data=label)

	return data, label, color

# Prepare original data without blurring.
def prepare_raw(path):
	# Settings.
	data = []
	color = []
	# Read in image and convert to ycrcb color space.
	img = cv.imread(path)
	im = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
	img = im2double(im) # Only use the luminance value.

	size = img.shape
	img_temp = scipy.misc.imresize(img, [size[0] * multiplier, size[1] * multiplier], interp='bicubic')
	color_temp = scipy.misc.imresize(im, [size[0] * multiplier, size[1] * multiplier], interp='bicubic')
	im_label = img_temp[:, :, 0]
	im_color = color_temp[:, :, 1:3]

	data = np.array(im_label).reshape([1, img.shape[0] * multiplier, img.shape[1] * multiplier, 1])
	color = np.array(im_color)

	return data, color


# Use the trained model to generate super-resolutioned image.
def generate_SR(path, save_dir):
	# Initialization.
	images = tf.placeholder(tf.float32, [None, None, None, 1], name='images')
	model = SRCNN(images)

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
			test_data, test_label, color = prepare_data(path)
			print('test_data has shape:', test_data.shape, 'label has shape:', test_label.shape)
			print('color has shape:', color.shape)

			# Generate super-resolutioned image.
			conv_out = model.eval({images: test_data})	# Result in patch of size 21x21.
			conv_out = conv_out.squeeze()
			result_bw = revert(conv_out)

			result = np.zeros([result_bw.shape[0], result_bw.shape[1], 3], dtype=np.uint8)
			result[:, :, 0] = result_bw
			result[:, :, 1:3] = color
			result = cv.cvtColor(result, cv.COLOR_YCrCb2RGB)

			bicubic = scipy.misc.imresize(test_label, 1 / multiplier, interp='bicubic')
			bicubic = scipy.misc.imresize(bicubic, multiplier * 1.0, interp='bicubic')
			bicubic = cv.cvtColor(bicubic, cv.COLOR_BGR2RGB)

			# Save the image.
			save_path = os.path.join(save_dir, os.path.basename(path))
			scipy.misc.imsave(save_path, result)
			bicubic_path = os.path.join(save_dir, 'bicubic_' + os.path.basename(path))
			scipy.misc.imsave(bicubic_path, bicubic)
			print('Finished testing', path)
		elif os.path.isdir(path):
			for root, dirs, files in os.walk(path):
				for im_name in files:
					img_path = os.path.join(path, im_name)
					print('Testing on image', img_path)

					test_data, test_label, color = prepare_data(img_path)
					print('test_data has shape:', test_data.shape, 'label has shape:', test_label.shape)
					print('color has shape:', color.shape)

					# Generate super-resolutioned image.
					conv_out = model.eval({images: test_data})	# Result in patch of size 21x21.
					conv_out = conv_out.squeeze()
					result_bw = revert(conv_out)

					result = np.zeros([result_bw.shape[0], result_bw.shape[1], 3], dtype=np.uint8)
					result[:, :, 0] = result_bw
					result[:, :, 1:3] = color
					result = cv.cvtColor(result, cv.COLOR_YCrCb2RGB)

					bicubic = scipy.misc.imresize(test_label, 1 / multiplier, interp='bicubic')
					bicubic = scipy.misc.imresize(bicubic, multiplier * 1.0, interp='bicubic')
					bicubic = cv.cvtColor(bicubic, cv.COLOR_BGR2RGB)

					# img = cv.imread(img_path)
					# h, w = img.shape[0], img.shape[1]
					# num_hor = math.ceil((h - size_input) / 21)
					# num_ver = math.ceil((w - size_input) / 21)

					# test_data, test_label = prepare_data(img_path)

					# # Generate super-resolutioned image.
					# conv_out = model.eval({images: test_data})	# Result in patch of size 21x21.
					# height, width = conv_out.shape[1], conv_out.shape[2]
					# # print('conv_out has shape:', conv_out.shape)
					# result = np.zeros([height * num_hor, width * num_ver, 1])
					# original = np.zeros([height * num_hor, width * num_ver, 1])
					# # print('result has shape:', result.shape)
					# # print('num_hor =', num_hor, 'num_ver =', num_ver)
					# i, j = 0, 0
					# for idx, image in enumerate(conv_out):
					# 	j = idx // num_ver
					# 	i = idx - j * num_ver
					# 	# print('idx =', idx, 'i =', i, 'j =', j)
					# 	result[j * height : j * height + height, i * width : i * width + width, :] = image
					# result = result.squeeze()
					# result = revert(result)

					# i, j = 0, 0
					# for idx, image in enumerate(test_label):
					# 	j = idx // num_ver
					# 	i = idx - j * num_ver
					# 	original[j * height : j * height + height, i * width : i * width + width, :] = image
					# original = original.squeeze()

					# size_original = original.shape
					# bicubic = scipy.misc.imresize(original, 1/3, interp='bicubic')
					# bicubic = scipy.misc.imresize(bicubic, size_original, interp='bicubic')

					# Save the image.
					save_path = os.path.join(save_dir, os.path.basename(img_path))
					scipy.misc.imsave(save_path, result)
					bicubic_path = os.path.join(save_dir, 'bicubic_' + os.path.basename(img_path))
					scipy.misc.imsave(bicubic_path, bicubic)
					print('Finished testing', os.path.basename(img_path))
			print('Finished testing all images.')
		else:
			print(' [*] Invalid input path.')

		



# Directly feed the original image to SRCNN.
def enhance(path, save_dir):
	# Initialization.
	images = tf.placeholder(tf.float32, [None, None, None, 1], name='images')
	model = SRCNN(images)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# Load the saved checkpoint.
		saver = tf.train.Saver()
		checkpoint_dir = ckpt_dir
		if load_ckpt(sess, checkpoint_dir, saver):
			print('Successfully loaded checkpoint.')
		else:
			print('Failed to load checkpoint.')

		if os.path.isfile(path):
			print('Upscaling image', path, '...')
			test_data, test_color = prepare_raw(path)

			# Generate super-resolutioned image.
			conv_out = model.eval({images: test_data})
			conv_out = conv_out.squeeze()

			result_bw = revert(conv_out)
			result = np.zeros([result_bw.shape[0], result_bw.shape[1], 3], dtype=np.uint8)
			result[:, :, 0] = result_bw
			result[:, :, 1:3] = test_color
			result = cv.cvtColor(result, cv.COLOR_YCrCb2RGB)
			save_path = os.path.join(save_dir, os.path.basename(path))
			scipy.misc.imsave(save_path, result)

			# Upscale_single_image(model, path, save_dir)
			print('Finished upscaling image', path)

		elif os.path.isdir(path):
			for root, dirs, files in os.walk(path):
				for im_name in files:
					img_path = os.path.join(path, im_name)

					print('Upscaling image', img_path, '...')
					test_data, test_color = prepare_raw(img_path)

					# Generate super-resolutioned image.
					conv_out = model.eval({images: test_data})
					conv_out = conv_out.squeeze()

					result_bw = revert(conv_out)
					result = np.zeros([result_bw.shape[0], result_bw.shape[1], 3], dtype=np.uint8)
					result[:, :, 0] = result_bw
					result[:, :, 1:3] = test_color
					result = cv.cvtColor(result, cv.COLOR_YCrCb2RGB)
					save_path = os.path.join(save_dir, os.path.basename(img_path))
					scipy.misc.imsave(save_path, result)
					print('Finished upscaling image', img_path)
			print('Finished upscaling all images.')
		else:
			print(' [*] Invalid input path.')


# Calculate num_ver and num_hor.
img_path = './Test/compressed_video_test/'
save_path = './result/compressed_video_result/'

# generate_SR(img_path, save_path)
enhance(img_path, save_path)

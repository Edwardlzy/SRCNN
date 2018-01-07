import tensorflow as tf
import numpy as np
import os
import time
import cv2 as cv
import h5py

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
train_path = os.path.join('./{}'.format('checkpoint'), "train.h5")
test_path = os.path.join('./{}'.format('checkpoint'), "test.h5")
ckpt_dir = './checkpoint/'

# Load the data prepared in h5 format.
def load_data(path):
	with h5py.File(path, 'r') as hf:
		data = np.array(hf.get('data'))
		label = np.array(hf.get('label'))
		return data, label

# Load the saved checkpoint. Reference: https://github.com/tegg89/SRCNN-Tensorflow/blob/master/model.py
def load_ckpt(sess, checkpoint_dir, saver):
	print(" [*] Reading checkpoints...")
	model_dir = "%s_%s" % ("srcnn", size_label)
	checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
	print('checkpoint_dir is', checkpoint_dir)

	# Require only one checkpoint in the directory.
	ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path:
	    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
	    print('Restoring from', os.path.join(checkpoint_dir, ckpt_name))
	    saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
	    return True
	else:
	    return False

# Save the current checkpoint for later use. Reference: https://github.com/tegg89/SRCNN-Tensorflow/blob/master/model.py
def save_ckpt(sess, step, saver):
	model_name = 'SRCNN.model'
	model_dir = "%s_%s" % ("srcnn", size_label)
	checkpoint_dir = os.path.join(ckpt_dir, model_dir)

	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

# Customized conv2d as described in the paper to make the code more readable.
def conv2d(x, W):
	# return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Define the computation graph of SRCNN.
def SRCNN(x):
	# Define weights and biases.
	# f1 = 9, f3 = 5, n1 = 64, n2 = 32.
	weights = {'w1' : tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3)),
			   'w2' : tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3)),
			   'w3' : tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3))}

	biases = {'b1' : tf.Variable(tf.zeros([64])),
			  'b2' : tf.Variable(tf.zeros([32])),
			  'b3' : tf.Variable(tf.zeros([1]))}

	conv1 = tf.nn.relu(conv2d(x, weights['w1']) + biases['b1'])
	conv2 = tf.nn.relu(conv2d(conv1, weights['w2']) + biases['b2'])
	conv3 = conv2d(conv2, weights['w3']) + biases['b3']
	return conv3

# Train the SRCNN and save the trained model periodically.
def train_SRCNN(x):
	# Initialization.
	model = SRCNN(x)
	l2_loss = tf.reduce_mean(tf.square(labels - model))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(l2_loss)
	train_data, train_label = load_data(train_path)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print('Training...')
		start_time = time.time()
		counter = 0
		saver = tf.train.Saver()

		if load_ckpt(sess, ckpt_dir, saver):
			print('Successfully loaded checkpoint.')
		else:
			print('Failed to load checkpoint.')

		# Training
		for epoch in range(num_epoch):
			epoch_loss = 0
			idx_batch = len(train_data) // batch_size
			for i in range(idx_batch):
				epoch_images = train_data[i * batch_size : (i + 1) * batch_size]
				epoch_labels = train_label[i * batch_size : (i + 1) * batch_size]

				_, c = sess.run([optimizer, l2_loss], feed_dict = {images: epoch_images, labels: epoch_labels})
				epoch_loss += c
				counter += 1

				# Log the training process every 10 steps.
				if counter % 10 == 0:
					print('Epoch:', epoch + 1, 'step:', counter, 'loss:', c, 'duration:', time.time() - start_time)

				# Save the checkpoint every 500 steps.
				if counter % 500 == 0:
					save_ckpt(sess, counter, saver)

# Use the trained model to generate super-resolutioned image.
# def generate_SR(x, num_ver, num_hor):
# 	# Initialization.
# 	model = SRCNN(x)
# 	l2_loss = tf.reduce_mean(tf.square(labels - model))
# 	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(l2_loss)
# 	test_data, test_label = load_data(test_path)

# 	with tf.Session() as sess:
# 		sess.run(tf.global_variables_initializer())
# 		print('Generating super-resolutioned image...')

# 		# Load the saved checkpoint.
# 		saver = tf.train.Saver()
# 		if load_ckpt(sess, ckpt_dir, saver):
# 			print('Successfully loaded checkpoint.')
# 		else:
# 			print('Failed to load checkpoint.')

# 		# Generate super-resolutioned image.
# 		conv_out = model.eval({images: test_data, labels: test_label})	# Result in patch of size 21x21.
# 		height, width = conv_out.shape[1], conv_out.shape[2]
# 		result = np.zeros([height * num_ver, width * num_hor, 1])
# 		for idx, image in enumerate(conv_out):
# 			i = idx % num_hor
# 			j = idx // num_hor
# 			result[j * height : j * height + height, i * width : i * width + width, :] = image
# 		result = result.squeeze()

# 		# Display and save the image.
# 		cv.imshow('super-resolution', result)
# 		cv.waitKey(0)
# 		save_path = os.path.join('./result', 'test.png')
# 		cv.imwrite(save_path, result)


# train_SRCNN(images)
# To generate, need to create data in h5 format then feed.
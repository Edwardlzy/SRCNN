import numpy as np
import cv2 as cv
import os
import scipy.misc
import h5py

# Settings.
scale = 3
size_input = 33
size_label = 21
stride = 14
counter = 0
# data = np.zeros([size_input, size_input, 1, 1])
# label = np.zeros([size_label, size_label, 1, 1])
data = []
label = []
padding = abs(size_input - size_label) / 2

# Source: https://stackoverflow.com/questions/29100722/equivalent-im2double-function-in-opencv-python
def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype

# Make sure the image shape is divisible by the scale factor. Also convert 3-channel image to 1-channel image.
def modcrop(image, scale=3):
	if image.shape[2] == 1:
		size = image.shape
		size -= np.mod(size, scale)
		image = image[0:size[0], 0:size[1]]
	else:
		size = image.shape[0:2]
		size -= np.mod(size, scale)
		image = image[0:size[0], 0:size[1], 0]
	return image

# Load and preprocess the training images.
dirpath = './Train/'
for root, dirs, files in os.walk(dirpath):
	for file in files:
		# Read in image and convert to ycrcb color space.
		img = cv.imread(dirpath + file)
		# cv.imshow('image',img)
		# cv.waitKey(0)
		# cv.destroyAllWindows()
		img = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
		img = im2double(img) # Only use the luminance value.

		# Create groundtruth and baseline image.
		im_label = modcrop(img)
		size = im_label.shape
		h = size[0]
		w = size[1]
		im_temp = scipy.misc.imresize(im_label, 1/scale, interp='bicubic')
		im_input = scipy.misc.imresize(im_temp, size, interp='bicubic')

		# Generate subimages for training.
		for x in range(0, h - size_input, stride):
			for y in range(0, w - size_input, stride):
				subim_input = im_input[x : x + size_input, y : y + size_input]
				subim_label = im_label[int(x + padding) : int(x + padding + size_label), int(y + padding) : int(y + padding + size_label)]
				
				subim_input = subim_input.reshape([size_input, size_input, 1])
				subim_label = subim_label.reshape([size_label, size_label, 1])

				data.append(subim_input)
				label.append(subim_label)
				counter += 1

# Shuffle the data pairs.
order = np.random.choice(counter, counter, replace=False)
data = np.array([data[i] for i in order])
label = np.array([label[i] for i in order])

print('data shape is', data.shape)
print('label shape is', label.shape)

# Write to HDF5 file.
savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)


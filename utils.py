import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

def read_data(filename, IMG_WIDTH, pad=0):
	'''
	Read data, normalize pixel value to [0... 1], 
	reshape data into square 2D matrix dim(IMG_WIDTH)

	Return data, labels
	'''
	data = pd.read_csv(filename, sep = ',')
	y = data['label']
	x = data.drop(labels = ['label'], axis = 1)
	x = x.astype('float32')
	x = x / 255.0
	x = x.values.reshape(-1, IMG_WIDTH, IMG_WIDTH)
	if pad > 0:
		x = np.pad(x, [(0, 0), (pad, pad), (pad, pad)], mode='constant')		
	del data
	return x, y

def img_show(img):
	plt.imshow(img, cmap='grey')
	plt.show()
	return

def shuffle_split(data, labels, percentile):
	'''
	Return randomly split by percentile train-test data and labels
	'''
	array_rand = np.random.rand(data.shape[0])
	split = array_rand < np.percentile(array_rand, percentile)
	test_x = data[~split]
	test_y = labels[~split]
	train_x = data[split]
	train_y = labels[split]
	return train_x, train_y, test_x, test_y

def init_filter(size, scale=1.0):
	'''
	size = (num_of_filters, depth, width, width)
	Return filters initialized with a mean=0 
	random normal distribution
	'''
	std_dev = scale/np.sqrt(np.prod(size))
	return np.random.normal(loc = 0, scale = std_dev, size = size)

def init_weight(size):
	'''
	size = (width, width)
	Return random standard normal distribution 
	'''
	return np.random.standard_normal(size = size) * 0.01

def convolution(image, filters, stride):
	
	dim_img = image.shape[0]
	dim_filter = filters[0].shape[0]
	num_filters = len(filters)
	dim_output = (dim_img - dim_filter) / stride + 1

	if dim_output.is_integer() == False:
		raise ValueError("(dim_img - dim_filter) / stride + 1 is not an integer.")

	output = np.zeros((num_filters, int(dim_output), int(dim_output)))
	#print(output.shape)
	#print(image[0:4, 0:4].shape)

	for f in range(num_filters):
		y_out = 0
		for y in range(0, dim_img, stride):
			x_out = 0
			for x in range(0, dim_img, stride):
				if (x + dim_filter <= dim_img and y + dim_filter <= dim_img):
					print(y_out, x_out, image[y:y+dim_filter, x:x+dim_filter].shape)
					#print(y, "->", y+dim_filter, x, "->", x+dim_filter, image[y:y+dim_filter, x:x+dim_filter].shape)
					output[f, y_out, x_out] = np.sum(filters[f] * image[y:y + dim_filter, x:x + dim_filter]) # + bias
				x_out += 1
			y_out += 1
	return output

def maxpool(matrix, size, stride):
	max = None
	return max


x, y = read_data('data/train.csv', 28, pad=2)
print("input shape", x[0].shape)
print("number of rows", len(x))
f1 = init_filter((4, 4))
f2 = init_filter((4, 4))
filters = [f1, f2]
conv = convolution(x[0], filters, 2)
print(conv)
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
	plt.imshow(img, cmap='gray')
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

def convolution(image, filters, stride, bias):
	'''
	image = WIDTH x WIDTH matrix
	filters = NUM_FILTERS x WIDTH x WIDTH tensor
	stride = int
	'''
	dim_img = image.shape[0]
	dim_filter = filters[0].shape[0]
	num_filters = len(filters)
	dim_out = (dim_img - dim_filter) / stride + 1
	print("Conv Input Dim:", dim_img)
	print("Conv Filter Dim:", dim_filter)

	if dim_out.is_integer() == False:
		raise ValueError("(dim_img - dim_filter) / stride + 1 is not an integer.")

	output = np.zeros((num_filters, int(dim_out), int(dim_out)))
	print("Conv Output Shape:", output.shape)
	#print(image[0:4, 0:4].shape)

	for f in range(num_filters):
		y_out = 0
		for y in range(0, dim_img, stride):
			x_out = 0
			for x in range(0, dim_img, stride):
				if (x + dim_filter <= dim_img and y + dim_filter <= dim_img):
					#print(y_out, x_out, image[y:y+dim_filter, x:x+dim_filter].shape)
					#print(y, "->", y+dim_filter, x, "->", x+dim_filter, image[y:y+dim_filter, x:x+dim_filter].shape)
					output[f, y_out, x_out] = np.sum(filters[f] * image[y:y + dim_filter, x:x + dim_filter]) # + bias
				x_out += 1
			y_out += 1
	return output

def maxpool(input, size, stride):
	dim_in = input[0].shape[0]
	dim_out = (dim_in - size) / stride + 1
	num_in = len(input)
	print("Maxpool Input Dim:", dim_in)
	
	if dim_out.is_integer() == False:
		raise ValueError("(dim_in - size) / stride + 1 is not an integer.")

	maxpool = np.zeros((num_in, int(dim_out), int(dim_out)))
	print("Maxpool Output Shape:", maxpool.shape)

	for i in range(num_in):
		y_out = 0
		for y in range(0, dim_in, stride):
			x_out = 0
			for x in range(0, dim_in, stride):
				if (x + size <= dim_in and y + size <= dim_in):
					#print(y, x)
					maxpool[i, y_out, x_out] = np.amax(input[i, y:y + size, x:x + size])
				x_out += 1
			y_out += 1	
	return maxpool

def relu(input):
	tmp = np.zeros_like(input)
	return np.where(input>tmp, input, tmp)

def flatten(in_pool):
	num_pool = len(in_pool)
	dim_pool = in_pool[0].shape[0]
	flat = in_pool.reshape((num_pool * dim_pool * dim_pool, 1))
	return flat

def dense(input, weights, bias):
	output = np.dot(weights, input) + bias
	return output

def softmax(out_dense):
	return np.exp(out_dense) / np.sum(np.exp(out_dense))

def categorical_cross_entropy(output, labels):
	return -np.sum(labels*np.log(output))

def convolution_back(in_conv, d_conv, filters, stride):
	num_filters = len(filters)
	dim_filter = filters[0].shape[0]
	num_in = len(in_conv)
	dim_in = in_conv[0].shape[0]

	d_output = np.zeros_like(in_conv)
	d_filters = np.zeros_like(filters)
	d_bias = np.zeros((num_filters, 1))

	for f in range(0, num_filters):
		y_out = 0
		for y in range(0, dim_in, stride):
			x_out = 0
			for x in range(0, dim_in, stride):
				if (x + dim_filter <= dim_in and y + dim_filter <= dim_in):
					d_filters[f] += np.dot(d_conv[f, x_out, y_out], in_conv[f, y:y+dim_filter, x:x+dim_filter])
					d_output[f, y:y+dim_filter, x:x+dim_filter] += d_conv[f, x_out, y_out] * filters[f]
					x_out += 1
			y += 1
		d_bias[f] = np.sum(filters[f])
	return d_output, d_filters, d_bias

def maxpool_back(in_pool, d_pool, size, stride):
	num_in = len(in_pool)
	dim_in = in_pool[0].shape[0]

	d_output = np.zeros_like(in_pool)

	for i in range(0, num_in):
		y_out = 0
		for y in range(0, dim_in, stride):
			x_out = 0
			for x in range(0, dim_in, stride):
				if (x + size <= dim_in and y + size <= dim_in):
					ids = np.nanargmax(in_pool[i, y:y+size, x:x+size])
					ids = np.unravel_index(ids, in_pool[i, y:y+size, x:x+size].shape)
					d_output[i, y:y+ids[0], x:x+ids[1]] = d_pool[i, y_out, x_out]
	return d_output

def relu_back(d_input, input):
	tmp = np.zeros_like(input)
	return np.where(input>tmp, d_input, tmp)

def cross_entropy_back(output, labels):
	return output - labels

def softmax_back(d_cross_entropy, in_dense, in_bias):
	d_weights = np.dot(d_cross_entropy, in_dense.T)
	d_bias = np.sum(d_cross_entropy).reshape(in_bias.shape)
	return d_weights, d_bias

def dense_back(d_out_dense, out_dense, in_dense, next_weights, bias):
	d_dense = np.dot(next_weights.T, d_out_dense)
	d_dense = relu_back(d_dense, out_dense)
	d_weights = np.dot(d_dense, in_dense.T)
	d_bias = np.sum(d_dense).reshape(bias.shape)
	return d_dense, d_weights, d_bias

def fully_connected_back(weights, d_dense, out_pool):
	d_fully_connected = np.dot(weights.T, d_dense)
	d_pool = d_fully_connected.reshape(out_pool.shape)
	return d_fully_connected, d_pool

x, y = read_data('data/train.csv', 28, pad=2)
labels = np.zeros((10, 1))
labels[y[0]] = y[0]
image = x[0]

print("Input Shape:", x[0].shape)
print("Number of Images:", len(x))

f1 = init_filter((5, 5))
f2 = init_filter((5, 5))
w = init_weight((800, 196))
w2 = init_weight((10, 800))
filters = [f1]
filter2 = [f2]
b1 = b2 = b3 = b4 = np.ones((1))

conv = convolution(image, filters, 1, b1)
conv = relu(conv)
conv2 = convolution(conv[0], filter2, 1, b2)
conv2 = relu(conv2)
mp = maxpool(conv, 2, 2)
flat = flatten(mp)
d = dense(flat, w, b3)
d = relu(d)
d2 = dense(d, w2, b4)
out = softmax(d2)
loss = categorical_cross_entropy(out, labels)

print("Flat Shape:", flat.shape)
print("Dense Output Shape:", d.shape)
print("Dense 2 Output Shape:", d2.shape)
print("Final Output Shape:", out.shape)
print("Final Output:\n", out)
print("Loss:", loss)

d_out = cross_entropy_back(out, labels)
d_w2, d_b4 = softmax_back(d_out, d, b4)
d_d, d_w, d_b3 = dense_back(d_out, d, flat, w2, b3)
d_fc, d_pool = fully_connected_back(w, d_d, mp)

print("Gradient of Output:\n", d_out)
print("Gradient of W2:", d_w2.shape)
print("Gradient of B4:", d_b4)
print("Gradient of Dense:", d_d.shape)
print("Gradient of W:", d_w.shape)
print("Gradient of B3:", d_b3)
print("Gradient of FC:", d_fc.shape)
print("Gradient of Pooled:", d_pool.shape)

d_conv2 = maxpool_back(conv2, d_pool, 2, 2)
d_conv2 = relu_back(d_conv2, conv2)
d_conv1, d_f2, d_b2 = convolution_back(conv, d_conv2, filter2, 1)
d_conv1 = relu_back(d_conv1, conv)
x = np.zeros((1, 32, 32))
x[0] = image 
d_input, d_f1, d_b1 = convolution_back(x, d_conv1, filters, 1)

print("Gradient of Conv2:", d_conv2.shape)
print("Gradient of Conv1:", d_conv1.shape)
print("Gradient of Filter2:", d_f2.shape)
print("Gradient of B2:", d_b2[0])
print("Gradient of Image:", d_input.shape)
print("Gradient of Filter1:", d_f1.shape)
print("Gradient of B1:", d_b1[0])
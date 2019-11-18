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
	dim_img = image.shape[1]
	dim_filter = filters[0].shape[1]
	num_filters = len(filters)
	dim_out = (dim_img - dim_filter) / stride + 1
	if len(image.shape) == 2:
		image = np.expand_dims(image, axis=0)
	print("Conv Input Dim:", image.shape)
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
					output[f, y_out, x_out] = np.sum(filters[f] * image[:, y:y + dim_filter, x:x + dim_filter]) # + bias
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
	dim_filter = filters[0].shape[1]
	num_in = len(in_conv)
	dim_in = in_conv[0].shape[0]
	if len(in_conv.shape) == 2:
		in_conv = np.expand_dims(in_conv, axis=0)

	d_output = np.zeros_like(in_conv)
	d_filters = np.zeros_like(filters)
	d_bias = np.zeros((num_filters, 1))
	print(in_conv.shape)
	print(d_conv.shape)

	for f in range(0, num_filters):
		y_out = 0
		for y in range(0, dim_in, stride):
			x_out = 0
			for x in range(0, dim_in, stride):
				if (x + dim_filter <= dim_in and y + dim_filter <= dim_in):
					d_filters[f] += np.dot(d_conv[f, x_out, y_out], in_conv[:, y:y+dim_filter, x:x+dim_filter])
					d_output[:, y:y+dim_filter, x:x+dim_filter] += d_conv[f, x_out, y_out] * filters[f]
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

def fully_connected_back(weights, d_dense, in_pool):
	d_fully_connected = np.dot(weights.T, d_dense)
	d_pool = d_fully_connected.reshape(in_pool.shape)
	return d_fully_connected, d_pool



def adam_gradient_descent(X, y, filters, weights, bias, alpha, beta1, beta2, epsilon, total_cost):
	[f1, f2, stride_f] = filters
	[w1, w2] = weights
	[b1, b2, b3, b4] = bias
	dim_img = X[0].shape[0]
	batch_cost = 0
	
	d_f1 = np.zeros_like(f1)
	d_f2 = np.zeros_like(f2)
	d_w1 = np.zeros_like(w1)
	d_w2 = np.zeros_like(w2)
	d_b1 = np.zeros_like(b1)
	d_b2 = np.zeros_like(b2)
	d_b3 = np.zeros_like(b3)
	d_b4 = np.zeros_like(b4)

	m_f1 = np.zeros_like(f1)
	m_f2 = np.zeros_like(f2)
	m_w1 = np.zeros_like(w1)
	m_w2 = np.zeros_like(w2)
	m_b1 = np.zeros_like(b1)
	m_b2 = np.zeros_like(b2)
	m_b3 = np.zeros_like(b3)
	m_b4 = np.zeros_like(b4)

	v_f1 = np.zeros_like(f1)
	v_f2 = np.zeros_like(f2)
	v_w1 = np.zeros_like(w1)
	v_w2 = np.zeros_like(w2)
	v_b1 = np.zeros_like(b1)
	v_b2 = np.zeros_like(b2)
	v_b3 = np.zeros_like(b3)
	v_b4 = np.zeros_like(b4)

	for i in range(len(X)):
		input = X[i]
		labels = np.zeros((10, 1))
		labels[y[i]] = y[i]
		loss, gradients = network_pass(input, labels, filters, weights, bias)
		d_f1 += gradients[0]
		d_f2 += gradients[1]
		d_w1 += gradients[2]
		d_w2 += gradients[3]
		d_b1 += gradients[4]
		d_b2 += gradients[5]
		d_b3 += gradients[6]
		d_b4 += gradients[7]
		batch_cost += loss

	m_f1 = beta1 * m_f1 + (1-beta1) + d_f1
	v_f1 = beta2 * v_f1 + (1-beta2) + d_f1**2
	f1 -= alpha / (np.sqrt(v_f1) + epsilon) * m_f1

	m_f2 = beta1 * m_f2 + (1-beta1) + d_f2
	v_f2 = beta2 * v_f2 + (1-beta2) + d_f2**2
	f2 -= alpha / (np.sqrt(v_f2) + epsilon) * m_f2

	m_w1 = beta1 * m_w1 + (1-beta1) + d_w1
	v_w1 = beta2 * v_w1 + (1-beta2) + d_w1**2
	w1 -= alpha / (np.sqrt(v_w1) + epsilon) * m_w1

	m_w2 = beta1 * m_w2 + (1-beta1) + d_w2
	v_w2 = beta2 * v_w2 + (1-beta2) + d_w2**2
	w2 -= alpha / (np.sqrt(v_w2) + epsilon) * m_w2

	m_b1 = beta1 * m_b1 + (1-beta1) + d_b1
	v_b1 = beta2 * v_b1 + (1-beta2) + d_b1**2
	b1 -= alpha / (np.sqrt(v_b1) + epsilon) * m_b1

	m_b2 = beta1 * m_b2 + (1-beta1) + d_b2
	v_b2 = beta2 * v_b2 + (1-beta2) + d_b2**2
	b2 -= alpha / (np.sqrt(v_b2) + epsilon) * m_b2

	m_b3 = beta1 * m_b3 + (1-beta1) + d_b3
	v_b3 = beta2 * v_b3 + (1-beta2) + d_b3**2
	b3 -= alpha / (np.sqrt(v_b3) + epsilon) * m_b3
	
	m_b4 = beta1 * m_b4 + (1-beta1) + d_b4
	v_b4 = beta2 * v_b4 + (1-beta2) + d_b4**2
	b4 -= alpha / (np.sqrt(v_b4) + epsilon) * m_b4

	total_cost.append(batch_cost/len(X))

	filters = [f1, f2, stride_f] 
	weights = [w1, w2]
	bias = [b1, b2, b3, b4]

	return total_cost, filters, weights, bias

def momentum_gradient_descent(X, y, filters, weights, bias, alpha, gamma, beta2, epsilon, total_cost):
	[f1, f2, stride_f] = filters
	[w1, w2] = weights
	[b1, b2, b3, b4] = bias
	dim_img = X[0].shape[0]
	batch_cost = 0
	
	d_f1 = np.zeros_like(f1)
	d_f2 = np.zeros_like(f2)
	d_w1 = np.zeros_like(w1)
	d_w2 = np.zeros_like(w2)
	d_b1 = np.zeros_like(b1)
	d_b2 = np.zeros_like(b2)
	d_b3 = np.zeros_like(b3)
	d_b4 = np.zeros_like(b4)

	v_f1 = np.zeros_like(f1)
	v_f2 = np.zeros_like(f2)
	v_w1 = np.zeros_like(w1)
	v_w2 = np.zeros_like(w2)
	v_b1 = np.zeros_like(b1)
	v_b2 = np.zeros_like(b2)
	v_b3 = np.zeros_like(b3)
	v_b4 = np.zeros_like(b4)

	for i in range(len(X)):
		input = X[i]
		labels = np.zeros((10, 1))
		labels[y[i]] = y[i]
		loss, gradients = network_pass(input, labels, filters, weights, bias)
		d_f1 += gradients[0]
		d_f2 += gradients[1]
		d_w1 += gradients[2]
		d_w2 += gradients[3]
		d_b1 += gradients[4]
		d_b2 += gradients[5]
		d_b3 += gradients[6]
		d_b4 += gradients[7]
		batch_cost += loss

	v_f1 = gamma * v_f1 + alpha * d_f1
	f1 -= v_f1

	v_f2 = gamma * v_f2 + alpha * d_f2
	f2 -= v_f2

	v_w1 = gamma * v_w1 + alpha * d_w1
	w1 -= v_w1

	v_w2 = gamma * v_w2 + alpha * d_w2
	w2 -= v_w2

	v_b1 = gamma * v_b1 + alpha * d_b1
	b1 -= v_b1

	v_b2 = gamma * v_b2 + alpha * d_b2
	b2 -= v_b2

	v_b3 = gamma * v_b3 + alpha * d_b3
	b3 -= v_b3

	v_b4 = gamma * v_b4 + alpha * d_b4
	b4 -= v_b4

	total_cost.append(batch_cost/len(X))

	filters = [f1, f2, stride_f] 
	weights = [w1, w2]
	bias = [b1, b2, b3, b4]

	return total_cost, filters, weights, bias

def network_pass(image, label, filters, weights, bias, pool=(2, 2)):
	[f1, f2, stride_f] = filters
	[w1, w2] = weights
	[b1, b2, b3, b4] = bias
	(dim_pool, stride_pool) = pool

	conv_1 = convolution(image, f1, stride_f, b1)
	conv_1 = relu(conv_1)
	conv_2 = convolution(conv_1, f2, stride_f, b2)
	conv_2 = relu(conv_2)
	pool = maxpool(conv_2, dim_pool, stride_pool)
	flat = flatten(pool)
	dense_1 = dense(flat, w, b3)
	dense_1 = relu(dense_1)
	dense_2 = dense(dense_1, w2, b4)
	output = softmax(dense_2)
	
	loss = categorical_cross_entropy(output, label)

	d_output = cross_entropy_back(output, labels)
	d_w2, d_b4 = softmax_back(d_output, dense_1, b4)
	d_dense, d_w1, d_b3 = dense_back(d_output, dense_1, flat, w2, b3)
	d_flat, d_pool = fully_connected_back(w1, d_dense, pool)
	d_conv_2 = maxpool_back(conv_2, d_pool, dim_pool, stride_pool)
	d_conv_2 = relu_back(d_conv_2, conv_2)
	d_conv_1, d_f2, d_b2 = convolution_back(conv_1, d_conv_2, f2, stride_f)
	d_conv_1 = relu_back(d_conv_1, conv_1)
	d_input, d_f1, d_b1 = convolution_back(image, d_conv_1, f1, stride_f)

	gradients = [d_f1, d_f2, d_w1, d_w2, d_b1, d_b2, d_b3, d_b4]

	return loss, gradients
	

x, y = read_data('data/train.csv', 28, pad=2)
labels = np.zeros((10, 1))
labels[y[0]] = y[0]
image = x[0]

print("Input Shape:", x[0].shape)
print("Number of Images:", len(x))

f1 = init_filter((8, 1, 5, 5))
f2 = init_filter((8, 8, 5, 5))
w = init_weight((800, 1568))
w2 = init_weight((10, 800))
#filters = [f1]
#filter2 = [f2]
b1 = b2 = b3 = b4 = np.ones((1))

conv = convolution(image, f1, 1, b1)
conv = relu(conv)
conv2 = convolution(conv, f2, 1, b2)
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
d_conv1, d_f2, d_b2 = convolution_back(conv, d_conv2, f2, 1)
d_conv1 = relu_back(d_conv1, conv)
x = np.zeros((1, 32, 32))
x[0] = image 
d_input, d_f1, d_b1 = convolution_back(x, d_conv1, f1, 1)

print("Gradient of Conv2:", d_conv2.shape)
print("Gradient of Conv1:", d_conv1.shape)
print("Gradient of Filter2:", d_f2.shape)
print("Gradient of B2:", d_b2[0])
print("Gradient of Image:", d_input.shape)
print("Gradient of Filter1:", d_f1.shape)
print("Gradient of B1:", d_b1[0])
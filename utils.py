import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import time
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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
	#print("Conv Input Dim:", image.shape)
	#print("Conv Filter Dim:", dim_filter)

	if dim_out.is_integer() == False:
		raise ValueError("(dim_img - dim_filter) / stride + 1 is not an integer.")

	output = np.zeros((num_filters, int(dim_out), int(dim_out)))
	#print("Conv Output Shape:", output.shape)
	#print(image[0:4, 0:4].shape)

	for f in range(num_filters):
		y_out = 0
		for y in range(0, dim_img, stride):
			x_out = 0
			for x in range(0, dim_img, stride):
				if (x + dim_filter <= dim_img and y + dim_filter <= dim_img):
					#print(y_out, x_out, image[y:y+dim_filter, x:x+dim_filter].shape)
					#print(y, "->", y+dim_filter, x, "->", x+dim_filter, image[y:y+dim_filter, x:x+dim_filter].shape)
					output[f, y_out, x_out] = np.sum(filters[f] * image[:, y:y + dim_filter, x:x + dim_filter]) + bias[f]
				x_out += 1
			y_out += 1
	return output

def maxpool(input, size, stride):
	dim_in = input[0].shape[0]
	dim_out = (dim_in - size) / stride + 1
	num_in = len(input)
	#print("Maxpool Input Dim:", dim_in)
	
	if dim_out.is_integer() == False:
		raise ValueError("(dim_in - size) / stride + 1 is not an integer.")

	maxpool = np.zeros((num_in, int(dim_out), int(dim_out)))
	#print("Maxpool Output Shape:", maxpool.shape)

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
	return input * (input > 0)

def flatten(in_pool):
	num_pool = len(in_pool)
	dim_pool = in_pool[0].shape[0]
	flat = in_pool.reshape((num_pool * dim_pool * dim_pool, 1))
	return flat

def dense(input, weights, bias):
	output = np.dot(weights, input) + bias
	return output

def softmax(out_dense):
	return (np.exp(out_dense)) / (np.sum(np.exp(out_dense)))

def categorical_cross_entropy(output, labels):
	return -np.sum(labels * np.log(output))

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
	#print(in_conv.shape)
	#print(d_conv.shape)

	for f in range(0, num_filters):
		y_out = 0
		for y in range(0, dim_in, stride):
			x_out = 0
			for x in range(0, dim_in, stride):
				if (x + dim_filter <= dim_in and y + dim_filter <= dim_in):
					d_filters[f] += (d_conv[f, x_out, y_out]*in_conv[:, y:y+dim_filter, x:x+dim_filter])
					d_output[:, y:y+dim_filter, x:x+dim_filter] += d_conv[f, x_out, y_out] * filters[f]
					x_out += 1
			y_out += 1
		d_bias[f] = np.sum(d_conv[f])
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
				x_out += 1
			y_out += 1
	return d_output

def relu_back(input):
	return 1 * (input > 0)

def gradient_initial(output, labels):
	gradient_out = np.zeros((10, 1))
	gradient_out[label] = -1 / output[label]
	return gradient_out

def gradient_softmax(gradient_initial, in_dense, label):
	in_exp = np.exp(in_dense)
	in_sum = np.sum(in_exp)
	d_out = -in_exp[label] * in_exp / (in_sum ** 2)
	d_out[label] = (in_exp[label] * (in_sum - in_exp[label])) / (in_sum ** 2)
	d_out = gradient_initial * d_out
	#d_weights = np.dot(d_cross_entropy, in_dense.T)
	#d_bias = np.sum(d_cross_entropy, axis = 1).reshape(in_bias.shape)
	return d_out



def gradient_dense(d_out, in_dense, in_weights):
	d_weights = np.dot(d_out, in_dense.T)
	d_bias = d_out
	d_dense = np.dot(in_weights.T, d_out)
	
	#d_dense = np.dot(next_weights.T, d_out_dense)
	#d_dense = relu_back(out_dense)
	#d_weights = np.dot(d_dense, in_dense.T)
	#d_bias = np.sum(d_dense, axis = 1).reshape(bias.shape)
	return d_dense, d_weights, d_bias

def gradient_pool(d_dense, in_pool):
	#d_fully_connected = np.dot(weights.T, d_dense)
	d_pool = d_dense.reshape(in_pool.shape)
	return d_pool

def momentum_gradient_descent(X, y, v, filters, weights, bias, total_cost, accuracy, w1_total, b1_total, predictions, alpha, gamma):
	[f1, f2, stride_f] = filters
	[w1, w2] = weights
	[b1, b2, b3, b4] = bias
	dim_img = X[0].shape[0]
	batch_cost = 0
	batch_correct = 0

	[v_f1, v_f2, v_w1, v_w2, v_b1, v_b2, v_b3, v_b4] = v

	d_f1 = np.zeros_like(f1)
	d_f2 = np.zeros_like(f2)
	d_w1 = np.zeros_like(w1)
	d_w2 = np.zeros_like(w2)
	d_b1 = np.zeros_like(b1)
	d_b2 = np.zeros_like(b2)
	d_b3 = np.zeros_like(b3)
	d_b4 = np.zeros_like(b4)

	for i in range(len(X)):
		input = X[i]
		labels = np.zeros((10, 1))
		labels[y.iloc[i]] = 1
		loss, output, gradients = network_pass(input, labels, filters, weights, bias)
		d_f1 += gradients[0]
		d_f2 += gradients[1]
		d_w1 += gradients[2]
		d_w2 += gradients[3]
		d_b1 += gradients[4]
		d_b2 += gradients[5]
		d_b3 += gradients[6]
		d_b4 += gradients[7]
		prediction = np.nanargmax(output)
		predictions.append(prediction)
		if prediction == y.iloc[i]:
			batch_correct += 1
		batch_cost += loss
	v_f1 = gamma * v_f1 + alpha * (d_f1 / len(X))
	f1 -= v_f1

	v_f2 = gamma * v_f2 + alpha * (d_f2 / len(X))
	f2 -= v_f2

	v_w1 = gamma * v_w1 + alpha * (d_w1 / len(X))
	w1 -= v_w1

	v_w2 = gamma * v_w2 + alpha * (d_w2 / len(X))
	w2 -= v_w2

	v_b1 = gamma * v_b1 + alpha * (d_b1 / len(X))
	b1 -= v_b1

	v_b2 = gamma * v_b2 + alpha * (d_b2 / len(X))
	b2 -= v_b2

	v_b3 = gamma * v_b3 + alpha * (d_b3 / len(X))
	b3 -= v_b3

	v_b4 = gamma * v_b4 + alpha * (d_b4 / len(X))
	b4 -= v_b4
	print("Accuracy:", batch_correct/len(X))
	total_cost.append(batch_cost/len(X))
	accuracy.append(batch_correct/len(X))
	w1_total.append(np.average(w1))
	b1_total.append(np.average(b1))

	v = [v_f1, v_f2, v_w1, v_w2, v_b1, v_b2, v_b3, v_b4]
	filters = [f1, f2, stride_f] 
	weights = [w1, w2]
	bias = [b1, b2, b3, b4]

	return total_cost, accuracy, w1_total, b1_total, predictions, filters, weights, bias, v

def forward_pass(image, label, filters, weights, bias, pool = (2, 2)):
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
	dense_1 = dense(flat, w1, b3)
	dense_1 = relu(dense_1)
	dense_2 = dense(dense_1, w2, b4)
	output = softmax(dense_2)
	
	loss = categorical_cross_entropy(output, label)

	return loss, output

def network_pass(image, label, filters, weights, bias, pool = (2, 2)):
	[f1, f2, stride_f] = filters
	[w1, w2] = weights
	[b1, b2, b3, b4] = bias
	(dim_pool, stride_pool) = pool
	y = np.zeros((10, 1))
	y[label] = 1

	conv_1 = convolution(image, f1, stride_f, b1)
	conv_1 = relu(conv_1)
	conv_2 = convolution(conv_1, f2, stride_f, b2)
	conv_2 = relu(conv_2)
	pool = maxpool(conv_2, dim_pool, stride_pool)
	flat = flatten(pool)
	dense_1 = dense(flat, w1, b3)
	dense_1 = relu(dense_1)
	dense_2 = dense(dense_1, w2, b4)
	output = softmax(dense_2)
	
	loss = categorical_cross_entropy(output, label)

	d_initial = gradient_initial(output, label)
	d_out = gradient_softmax(d_initial, dense_2, label)
	d_dense_2, d_w2, d_b4 = gradient_dense(d_out, dense_1, w2)
	d_dense_1, d_w1, d_b3 = gradient_dense(d_dense_2, flat, w1)
	d_dense_1 = relu_back(d_dense_1)
	d_pool = gradient_pool(d_dense_1, pool)
	d_conv_2 = maxpool_back(conv_2, d_pool, dim_pool, stride_pool)
	d_conv_2 = relu_back(conv_2)
	d_conv_1, d_f2, d_b2 = convolution_back(conv_1, d_conv_2, f2, stride_f)
	d_conv_1 = relu_back(conv_1)
	d_input, d_f1, d_b1 = convolution_back(image, d_conv_1, f1, stride_f)
	#d_w2, d_b4 = softmax_back(d_output, dense_1, b4)
	#d_dense, d_w1, d_b3 = dense_back(d_output, dense_1, flat, w2, b3)
	#d_flat, d_pool = fully_connected_back(w1, d_dense, pool)
	

	gradients = [d_f1, d_f2, d_w1, d_w2, d_b1, d_b2, d_b3, d_b4]

	return loss, output, gradients

def backward_pass(output, label, filters, weights, bias, pool = (2,2)):
	d_initial = gadient_initial(output, label)
	d_w2, d_b4 = softmax_back(d_output, dense_1, b4)
	d_dense, d_w1, d_b3 = dense_back(d_output, dense_1, flat, w2, b3)
	d_flat, d_pool = fully_connected_back(w1, d_dense, pool)
	d_conv_2 = maxpool_back(conv_2, d_pool, dim_pool, stride_pool)
	d_conv_2 = relu_back(conv_2)
	d_conv_1, d_f2, d_b2 = convolution_back(conv_1, d_conv_2, f2, stride_f)
	d_conv_1 = relu_back(conv_1)
	d_input, d_f1, d_b1 = convolution_back(image, d_conv_1, f1, stride_f)

	gradients = [d_f1, d_f2, d_w1, d_w2, d_b1, d_b2, d_b3, d_b4]

	return gradients


def train_network(images, dim_img, labels, num_filters1, num_filters2, dim_filters, stride, alpha = 0.01, gamma = 0.88, batch_size = 20, epochs = 20):	
	f1 = init_filter((num_filters1, 1, dim_filters, dim_filters))
	f2 = init_filter((num_filters2, num_filters1, dim_filters, dim_filters))
	w1 = init_weight((400, 1152))
	w2 = init_weight((10, 400))
	b1 = np.zeros((num_filters1, 1))
	b2 = np.zeros((num_filters2, 1))
	b3 = np.zeros((400, 1))
	b4 = np.zeros((10, 1))
 
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

	m = [m_f1, m_f2, m_w1, m_w2, m_b1, m_b2, m_b3, m_b4]
	v = [v_f1, v_f2, v_w1, v_w2, v_b1, v_b2, v_b3, v_b4]

	filters = [f1, f2, stride]
	weights = [w1, w2]
	biases = [b1, b2, b3, b4]
	total_cost = []
	w1_batch = []
	b1_batch = []
	accuracy = []
	predictions = []

	
	global_start = time.time()

	for epoch in range(epochs):
		count = 0
		for j in range(0, len(images), batch_size):
			count += 1
			batch = images[j:j + batch_size]
			batch_labels = labels[j:j + batch_size]
			start_time = time.time()
			total_cost, accuracy, w1_batch, b1_batch, predictions, filters, weights, biases, v, m = adam_gradient_descent(batch, batch_labels, v, m, filters, weights, biases, total_cost, count, accuracy, w1_batch, b1_batch, alpha = 0.001, beta1 = 0.95, beta2 = 0.99, epsilon = 0.000000001)
			print("Epoch: {}".format(epoch))
			print("Batch: {}".format((j+batch_size)/batch_size))
			print("Loss: {:.4e}".format(total_cost[-1]))
			print("Time: {:.4e}s".format(time.time() - start_time))
	print("Total Time: {:.4e}s".format(time.time() - global_start))
	
	recall = np.zeros((10, 1))
	precision = np.zeros((10, 1))
 
	for i in range(len(labels)):
		if predictions[i] == labels[i]:
			recall[predictions[i]] += 1

	[f1, f2, stride] = filters
	[w1, w2] = weights
	[b1, b2, b3, b4] = biases

	np.save("recall", recall)
	np.save("f1", f1)
	np.save("f2", f2)
	np.save("w1", w1)
	np.save("w2", w1)
	np.save("b1", b1)
	np.save("b2", b2)
	np.save("b3", b3)
	np.save("b4", b4)
	np.save("cost", total_cost)
			
	return total_cost, accuracy, w1_batch, b1_batch, filters, weights, biases

def validation(x, y, filters, weights, bias):
	correct_class_count = np.zeros((10, 1))
	predictions = []
	loss_score = []
	correct = 0

	for i in range(0, len(x)):
		input = x[0]
		label = np.zeros((10, 1))
		label[y[i]] = y[i]
		loss, prediction, probability = predict(input, label, filers, weights, bias)
		predictions.append(prediction)
		loss_score.append(loss)
		if prediction == y[i]:
			correct += 1
			correct_class_count[prediction] += 1

	return correct_class_count, predictions, loss_score, correct

def predict(image, label, filters, weights, bias, pool = (2, 2)):
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
	
	prediction = np.nanargmax(output)
	probability = np.max(output)

	loss = categorical_cross_entropy(output, y_test)

	return loss, prediction, probability

def plot_loss(loss):
	plt.plot(loss)
	plt.ylabel("Loss")
	plt.xlabel("Batches")
	plt.show()







#x, y = read_data('data/train.csv', 28, pad=2)
#x_train, y_train, x_test, y_test = shuffle_split(x, y, 70)
#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#total_loss, accuracy, w1_batch, b1_batch, filters, weights, bias = train_network(x_train, 28, y_train, 8, 8, 5, 1)
#plot_loss(total_loss)
#plot_loss(accuracy)
#w1 = weights[0]
#b1 = bias[0]
#plot_3d(total_loss, w1, b1)


# print("Input Shape:", x[0].shape)
# print("Number of Images:", len(x))
#image = x_train[0]
#labels = np.zeros((10, 1))
#labels[y_train[0]] = 1
#label = y_train[0]
# f1 = init_filter((8, 1, 5, 5))
# f2 = init_filter((8, 8, 5, 5))
# w1 = init_weight((128, 1568))
# w2 = init_weight((10, 128))
# b1 = np.zeros((8, 1))
# b2 = np.zeros((8, 1))
# b3 = np.zeros((128, 1))
# b4 = np.zeros((10, 1))
# filters = [f1, f2, 2]
# weights = [w1, w2]
# biases = [b1, b2, b3, b4]


# conv = convolution(image, f1, 1, b1)
# conv = relu(conv)
# conv2 = convolution(conv, f2, 1, b2)
# conv2 = relu(conv2)
# mp = maxpool(conv, 2, 2)
# flat = flatten(mp)
# d = dense(flat, w1, b3)
# d = relu(d)
# d2 = dense(d, w2, b4)
# out = softmax(d2)
# loss = categorical_cross_entropy(out, labels)

# print(mp.shape)
# print("Flat Shape:", flat.shape)
# print("Dense Output Shape:", d.shape)
# print("Dense 2 Output Shape:", d2.shape)
# print("Final Output Shape:", out.shape)
# print("Final Output:\n", out)
# print("Loss:", loss)

# d_initial = gradient_initial(out, label)
# d_out = gradient_softmax(d_initial, d2, label)
# print(d_out.shape)
# print(d.shape)
# print(w2.shape)
# d_d2, d_w2, d_b4 = gradient_dense(d_out, d, w2)
# d_d1, d_w1, d_b3 = gradient_dense(d_d2, flat, w1)
# d_pool = gradient_pool(d_d1, mp)

# print("Gradient of softmax:\n", d_initial)
# print("Gradient of Output:\n", d_out)
# print("Gradient of W2:", d_w2.shape)
# print("Gradient of B4:", d_b4.shape)
# print("Gradient of Dense:", d_d1.shape)
# print("Gradient of W:", d_w1.shape)
# print("Gradient of B3:", d_b3.shape)
# print("Gradient of Pooled:", d_pool.shape)

# d_conv2 = maxpool_back(conv2, d_pool, 2, 2)
# d_conv2 = relu_back(conv2)
# d_conv1, d_f2, d_b2 = convolution_back(conv, d_conv2, f2, 1)
# d_conv1 = relu_back(conv)
# x = np.zeros((1, 32, 32))
# x[0] = image 
# d_input, d_f1, d_b1 = convolution_back(image, d_conv1, f1, 1)

# print("Gradient of Conv2:", d_conv2.shape)
# print("Gradient of Conv1:", d_conv1.shape)
# print("Gradient of Filter2:", d_f2.shape)
# print("Gradient of B2:", d_b2.shape)
# print("Gradient of Image:", d_input.shape)
# print("Gradient of Filter1:", d_f1.shape)
# print("Gradient of B1:", d_b1.shape)
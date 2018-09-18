import numpy as np;
import tensorflow as tf;
import math
import car_extract as extractor

brand_names, samples = extractor.read_files()
brand_cnt = len(brand_names.items())
#print(samples[0:100])
samples = [x for x in samples if x[-1] <= 0.99]
samples = np.array(samples, dtype=np.float32)

randidx = np.random.randint(0, samples.shape[0], 6000)
print(randidx)

test = samples[randidx]
test_shape = np.shape(test)

##TODO:
#samples = np.delete(samples, randidx, [:])

bools = np.ones(shape=[samples.shape[0], ])
print(bools.dtype)
bools = bools[bools > 0].astype(np.bool)
print(bools.dtype)
bools[randidx] = False
samples = samples[bools]

sample_shape = np.shape(samples)

batch_size = 6000
embedding_width = 7 
embedding_width = 1 
hidden_width = 20
total_epoch = 5000
#total_epoch = 1

def sigmoid_full_layer(input, input_width, output_width, activation=tf.nn.sigmoid):
	w = tf.Variable(tf.random_normal([input_width, output_width], -1.0, 1.0))
	b = tf.Variable(tf.zeros([output_width,]))
	prod = tf.matmul(input, w)
	a = activation(tf.add(prod, b))
	#a = tf.nn.sigmoid(tf.add(prod, b))
	#a = tf.nn.tanh(tf.add(prod, b))
	#a = tf.nn.relu(tf.add(prod, b))
	return w, b, a

def res_block(hidden_width, input):
	#w1, b1, a = sigmoid_full_layer(input, hidden_width, hidden_width, tf.nn.leaky_relu)
	#w1, b1, a = sigmoid_full_layer(input, hidden_width, hidden_width, tf.nn.relu)
	#w1, b1, a = sigmoid_full_layer(input, hidden_width, hidden_width, tf.nn.selu)
	w1, b1, a = sigmoid_full_layer(input, hidden_width, hidden_width, tf.nn.sigmoid)
	#w2, b2, a = sigmoid_full_layer(a, hidden_width, hidden_width, tf.nn.leaky_relu)
	#w2, b2, a = sigmoid_full_layer(a, hidden_width, hidden_width, tf.nn.relu)
	#w2, b2, a = sigmoid_full_layer(a, hidden_width, hidden_width, tf.nn.selu)
	w2, b2, a = sigmoid_full_layer(a, hidden_width, hidden_width, tf.nn.sigmoid)
	a = tf.add(input, a)
	return w1, b1, w2, b2, a
	
x = tf.placeholder(dtype=tf.float32, shape=(None, sample_shape[1] - 2))
brand_ids = tf.placeholder(dtype=tf.int32, shape=(None, ))

embedding = tf.get_variable('embedding', [brand_cnt, embedding_width], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1e-4))
brand_embedding = tf.nn.embedding_lookup(embedding, brand_ids)
concat_layer = tf.concat([brand_embedding, x], axis=1, name='concat')

#w, b, pred = sigmoid_full_layer(concat_layer, sample_shape[1] - 2 + embedding_width, 1)

w, b, a = sigmoid_full_layer(concat_layer, sample_shape[1] - 2 + embedding_width, hidden_width)
'''
w11, b11, w12, b12, a = res_block(hidden_width, a)
w21, b21, w22, b22, a = res_block(hidden_width, a)
w31, b31, w32, b32, a = res_block(hidden_width, a)
w41, b41, w42, b42, a = res_block(hidden_width, a)
w51, b51, w52, b52, a = res_block(hidden_width, a)
w61, b61, w62, b62, a = res_block(hidden_width, a)
'''
w3, b3, pred = sigmoid_full_layer(a, hidden_width, 1)

output = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='output')
output_ave = np.average(samples[:, -1])

target_ratio = tf.ones(shape=tf.shape(output))

diff_ratio = pred / output - 1
diff_amount = pred - output
loss = tf.reduce_mean(abs(diff_ratio))
#total_loss = loss * 0.5 + tf.reduce_mean(tf.square(diff_ratio)) * 0.0 + tf.reduce_mean(tf.square(diff_amount)) * 0.5
total_loss = loss# + tf.reduce_mean(tf.square(diff_ratio)) 
#total_loss =  tf.reduce_mean(diff_amount * diff_amount)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
#optimizer = tf.train.MomentumOptimizer(learning_rate=1e-2)
#optimizer = tf.train.AdagradOptimizer(learning_rate=1e-2)
#optimizer = tf.train.AdadeltaOptimizer(learning_rate=1e-2)
optimizer = tf.train.AdamOptimizer(learning_rate=1.5e-2)
train_op = optimizer.minimize(total_loss)

batch_count = sample_shape[0] // batch_size

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
	#sess.run(tf.initialize_all_variables())
	sess.run(tf.global_variables_initializer())
	for epoch in np.arange(total_epoch):
		rmse = 0.0
		count = batch_count
		for batch in np.arange(batch_count):
			idx = batch * batch_size
			(train_op_,  loss_, output_, pred_) = sess.run((train_op, loss, output, pred), \
				feed_dict={
					brand_ids:np.reshape(samples[idx:idx+batch_size, 0:1], newshape=(batch_size, )).astype(np.int32),  
					x:samples[idx:idx+batch_size, 1:-1], 
					output:np.reshape(samples[idx:idx+batch_size, -1:], newshape=(batch_size, 1)).astype(np.float32)})
			#print("Batch %d:\tloss:%f" % (batch, loss_))
			#print("Output:")
			#print(output_)
			#print("Predict:")
			#print(pred_)
			#print()
			rmse = rmse + loss_

		idx = batch_count * batch_size
		if idx < sample_shape[0]:
			last_batch_size = sample_shape[0] - idx
			(train_op_,  loss_, output_, pred_) = sess.run((train_op, loss, output, pred), \
				feed_dict={
					brand_ids:np.reshape(samples[idx:, 0:1], newshape=(last_batch_size, )).astype(np.int32),  
					x:samples[idx:, 1:-1], 
					output:np.reshape(samples[idx:, -1:], newshape=(last_batch_size, 1)).astype(np.float32)})
			rmse = rmse + loss_
			count = batch_count + 1

		rmse = rmse / count 
		print("Epoch %d:\trmse:%f" % (epoch, rmse))

	#########################################
	print("===================================")
	batch_count = test_shape[0] // batch_size
	rmse = 0.0
	result = np.array([[0, 0]])

	count = batch_count
	for batch in np.arange(batch_count):
		idx = batch * batch_size
		(output_, pred_, loss_) = sess.run((output, pred, loss), \
			feed_dict={
				brand_ids:np.reshape(test[idx:idx+batch_size, 0:1], newshape=(batch_size, )).astype(np.int32),  
				x:test[idx:idx+batch_size, 1:-1], 
				output:np.reshape(test[idx:idx+batch_size, -1:], newshape=(batch_size, 1)).astype(np.float32)})

		tmp_result = np.hstack((output_, pred_))
		result = np.vstack((result, tmp_result))
		print(tmp_result)

		rmse = rmse + loss_

	idx = batch_count * batch_size
	if idx < test_shape[0]:
		last_batch_size = test_shape[0] - idx
		(output_, pred_, loss_) = sess.run((output, pred, loss), \
			feed_dict={
				brand_ids:np.reshape(test[idx:, 0:1], newshape=(last_batch_size, )).astype(np.int32),  
				x:test[idx:, 1:-1], 
				output:np.reshape(test[idx:, -1:], newshape=(last_batch_size, 1)).astype(np.float32)})

		tmp_result = np.hstack((output_, pred_))
		result = np.vstack((result, tmp_result))
		print(tmp_result)

		rmse = rmse + loss_
		count = batch_count + 1


	rmse = rmse / count
	print("Brand count:%d, Embedding_width:%d, Hidden_width:%d" % (brand_cnt, embedding_width, hidden_width))
	print("Epoch %d:\trmse:%f" % (epoch, rmse))
	print("Residual average:%f" % output_ave)

	import matplotlib.pyplot as plt
	result = result[1:, :]
	result = np.array(sorted(result, key = lambda x : x[0]))
	plt.figure()
	plt.plot(np.arange(len(result)), np.reshape(result[:, 0], newshape=[len(result)]), 'r-', lw=5)
	plt.scatter(np.arange(len(result)), np.reshape(result[:, 1], newshape=[len(result)]))
	plt.show()



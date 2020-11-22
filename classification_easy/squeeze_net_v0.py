import tensorflow as tf

def conv2d(inputs,
           filters,
           kernel_size,
           strides=(1, 1),
           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
           bias_initializer=tf.zeros_initializer(),
           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002),
           name=None):
	return tf.layers.conv2d(inputs,
			        filters,
				kernel_size,
				strides,
				kernel_initializer=kernel_initializer,
				bias_initializer=bias_initializer,
				kernel_regularizer=kernel_regularizer,
				activation=tf.nn.relu,
				name=name,
				padding="same")

def fire_module(inputs, squeeze_depth, expand_depth, name):
  """Fire module: squeeze input filters, then apply spatial convolutions."""
  with tf.variable_scope(name, "fire", [inputs]):
    squeezed = conv2d(inputs, squeeze_depth, [1, 1], name="squeeze")
    e1x1 = conv2d(squeezed, expand_depth, [1, 1], name="e1x1")
    e3x3 = conv2d(squeezed, expand_depth, [3, 3], name="e3x3")
    return tf.concat([e1x1, e3x3], axis=3)

def squeeze_net_model_v0(images, is_training, n_classes):
	"""Squeezenet 1.0 model."""
	net = conv2d(images, 96, [7, 7], strides=(2, 2), name="conv1")
	net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool1")
	net = fire_module(net, 16, 64, name="fire2")
	net = fire_module(net, 16, 64, name="fire3")
	net = fire_module(net, 32, 128, name="fire4")
	net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool4")
	net = fire_module(net, 32, 128, name="fire5")
	net = fire_module(net, 48, 192, name="fire6")
	net = fire_module(net, 48, 192, name="fire7")
	net = fire_module(net, 64, 256, name="fire8")
	net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool8")
	net = fire_module(net, 64, 256, name="fire9")
	net = tf.layers.dropout(net, rate=0.5 if is_training else 0.0, name="drop9")
	net = conv2d(net, n_classes, [1, 1], strides=(1, 1), name="conv10")
	#print(net.get_shape())
	net = tf.layers.average_pooling2d(net, pool_size=(5, 5), strides=(1, 1))
	#print(net.get_shape())
	logits = tf.contrib.layers.flatten(net)
	#print(logits.get_shape())
	return logits

def losses(logits, labels):
    with tf.variable_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels)
        loss = tf.reduce_mean(cross_entropy)
    return loss
    
def evaluation(logits, labels):
    with tf.variable_scope("accuracy"):
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
    return accuracy

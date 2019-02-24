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
                            activation=None,
                            name=name,
                            padding="same")

'''
for each conv, the input and the output are the same dim
'''
def residual_block(inputs, out_channel, strides, if_training, name):
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        shortcut = inputs
        # Residual
        net = conv2d(inputs, out_channel, 3, strides, name='conv_1')
        net = tf.layers.batch_normalization(net, training=if_training, name='conv1_bn')
        net = tf.nn.relu(net)

        net = conv2d(net, out_channel, 3, strides, name='conv_2')
        net = tf.layers.batch_normalization(net, training=if_training, name='conv2_bn')
        net = tf.nn.relu(net)

        net = net + shortcut
        net = tf.nn.relu(net, name="conv2_relu")
        return net

'''
only the first conv is responsible for downsampling
'''
def residual_block_downsample(inputs, out_channel, strides, if_training, name):
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        shortcut = conv2d(inputs, out_channel, 1, strides, name='shortcut')
        # Residual
        net = conv2d(inputs, out_channel, 3, strides, name='conv_1')
        net = tf.layers.batch_normalization(net, training=if_training, name='conv1_bn')
        net = tf.nn.relu(net)

        net = conv2d(net, out_channel, 3, 1, name='conv_2')
        net = tf.layers.batch_normalization(net, training=if_training, name='conv2_bn')
        net = tf.nn.relu(net)
       
        net = net + shortcut
        net = tf.nn.relu(net, name="conv2_relu")
        return net

def resnet_18(images, is_training, n_classes):
    """resnet18 model."""
    filters = [64, 64, 128, 256, 512]
    kernels = [7, 3, 3, 3, 3]
    strides = [2, 0, 2, 2, 2]

    # conv1
    print('\tBuilding unit: conv1')
    net = conv2d(images, filters[0], kernels[0], strides[0], name="conv1")
    net = tf.layers.batch_normalization(net, training=is_training, name='conv1_bn')
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), padding='same', name="conv1_maxpooling")

    net = residual_block(net, filters[1], strides=1, if_training=is_training, name='residual_conv2_1')
    net = residual_block(net, filters[1], strides=1, if_training=is_training, name='residual_conv2_2')

    net = residual_block_downsample(net, filters[2], strides=strides[2], if_training=is_training, name='residual_conv3_1')
    net = residual_block(net, filters[2], strides=1, if_training=is_training, name='residual_conv3_2')

    net = residual_block_downsample(net, filters[3], strides=strides[3], if_training=is_training, name='residual_conv4_1')
    net = residual_block(net, filters[3], strides=1, if_training=is_training, name='residual_conv4_2')

    net = residual_block_downsample(net, filters[4], strides=strides[4], if_training=is_training, name='residual_conv5_1')
    net = residual_block(net, filters[4], strides=1, if_training=is_training, name='residual_conv5_2')

    net = conv2d(net, n_classes, [1, 1], strides=(1, 1), name="last_conv_layer") #instead of fully connected layer
    print(net.get_shape()) # 28,4,4,2
    net = tf.layers.average_pooling2d(net, pool_size=(4, 4), strides=(1, 1))
    logits = tf.contrib.layers.flatten(net)
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

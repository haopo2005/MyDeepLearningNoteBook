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

def resnext_block(inputs, out_channel, strides, cardinality, if_training, name):
    with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = inputs
            # Residual
            layers_split = []
            for j in range(0, cardinality):
                temp_net = conv2d(inputs, out_channel, 1, strides, name='conv_1_'+str(j))
                temp_net = tf.layers.batch_normalization(temp_net, training=if_training, name='conv1_bn_'+str(j))
                temp_net = tf.nn.relu(temp_net)
                temp_net = conv2d(temp_net, out_channel, 3, 1, name='conv_2_'+str(j))
                temp_net = tf.layers.batch_normalization(temp_net, training=if_training, name='conv2_bn_'+str(j))
                temp_net = tf.nn.relu(temp_net)
                temp_net = conv2d(inputs, out_channel*2, 1, 1, name='conv_3_'+str(j))
                temp_net = tf.layers.batch_normalization(temp_net, training=if_training, name='conv3_bn_'+str(j))
                temp_net = tf.nn.relu(temp_net)
                layers_split.append(temp_net)
            net = tf.concat(layers_split, axis=3)
            net = conv2d(net, out_channel, 1, strides, name='conv_merged')
            net = tf.layers.batch_normalization(net, training=if_training, name='conv_bn_merged')
            net = net + shortcut
            net = tf.nn.relu(net, name="conv3_relu")
    return net

def resnext_block_downsample(inputs, out_channel, strides, cardinality, if_training, name):
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        shortcut = conv2d(inputs, out_channel, 1, strides, name='shortcut')
        # Residual
        layers_split = []
        for j in range(0, cardinality):
            temp_net = conv2d(inputs, out_channel, 1, strides, name='conv_1_'+str(j))
            temp_net = tf.layers.batch_normalization(temp_net, training=if_training, name='conv1_bn_'+str(j))
            temp_net = tf.nn.relu(temp_net)
            temp_net = conv2d(temp_net, out_channel, 3, strides, name='conv_2_'+str(j))
            temp_net = tf.layers.batch_normalization(temp_net, training=if_training, name='conv2_bn_'+str(j))
            temp_net = tf.nn.relu(temp_net)
            temp_net = conv2d(inputs, out_channel*2, 1, strides, name='conv_3_'+str(j))
            temp_net = tf.layers.batch_normalization(temp_net, training=if_training, name='conv3_bn_'+str(j))
            temp_net = tf.nn.relu(temp_net)
            layers_split.append(temp_net)
        net = tf.concat(layers_split, axis=3)
        net = conv2d(net, out_channel, 1, 1, name='conv_merged')
        net = tf.layers.batch_normalization(net, training=if_training, name='conv_bn_merged')
        net = net + shortcut
        net = tf.nn.relu(net, name="conv3_relu")
        return net

def resnext_50(images, is_training, n_classes):
    # conv1
    print('\tBuilding unit: conv1')
    net = conv2d(images, 64, 7, 2, name="conv1")
    net = tf.layers.batch_normalization(net, training=is_training, name='conv1_bn')
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), padding='same', name="conv1_maxpooling")

    # conv2
    net = resnext_block(net, 64, strides=1, cardinality=8, if_training=is_training, name='residual_conv2_1')
    net = resnext_block(net, 64, strides=1, cardinality=8, if_training=is_training, name='residual_conv2_2')
    net = resnext_block(net, 64, strides=1, cardinality=8, if_training=is_training, name='residual_conv2_3')
    
    # conv3
    net = resnext_block_downsample(net, 128, strides=2, cardinality=8, if_training=is_training, name='residual_conv3_1')
    net = resnext_block(net, 128, strides=1, cardinality=8, if_training=is_training, name='residual_conv3_2')
    net = resnext_block(net, 128, strides=1, cardinality=8, if_training=is_training, name='residual_conv3_3')
    net = resnext_block(net, 128, strides=1, cardinality=8, if_training=is_training, name='residual_conv3_4')

    # conv4
    net = resnext_block_downsample(net, 256, strides=2, cardinality=8, if_training=is_training, name='residual_conv4_1')
    net = resnext_block(net, 256, strides=1, cardinality=8, if_training=is_training, name='residual_conv4_2')
    net = resnext_block(net, 256, strides=1, cardinality=8, if_training=is_training, name='residual_conv4_3')
    net = resnext_block(net, 256, strides=1, cardinality=8, if_training=is_training, name='residual_conv4_4')
    net = resnext_block(net, 256, strides=1, cardinality=8, if_training=is_training, name='residual_conv4_5')
    net = resnext_block(net, 256, strides=1, cardinality=8, if_training=is_training, name='residual_conv4_6')

    # conv5
    net = resnext_block_downsample(net, 512, strides=2, cardinality=8, if_training=is_training, name='residual_conv5_1')
    net = resnext_block(net, 512, strides=1, cardinality=8, if_training=is_training, name='residual_conv5_2')
    net = resnext_block(net, 512, strides=1, cardinality=8, if_training=is_training, name='residual_conv5_3')

    net = conv2d(net, n_classes, [1, 1], strides=(1, 1), name="last_conv_layer") #instead of fully connected layer
    print(net.get_shape()) # ?,4,4,2
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

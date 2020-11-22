import tensorflow as tf
import math
from configuration import NUM_CLASSES
import numpy as np

def round_filters(filters, multiplier):
    depth_divisor = 8
    min_depth = None
    min_depth = min_depth or depth_divisor
    filters = filters * multiplier
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, multiplier):
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, ratio=0.25):
        super(SEBlock, self).__init__()
        self.num_reduced_filters = max(1, int(input_channels * ratio))
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.reduce_conv = tf.keras.layers.Conv2D(filters=self.num_reduced_filters,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same")
        self.expand_conv = tf.keras.layers.Conv2D(filters=input_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same")

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = self.reduce_conv(branch)
        branch = tf.nn.swish(branch)
        branch = self.expand_conv(branch)
        branch = tf.nn.sigmoid(branch)
        output = inputs * branch
        return output

class CBAMBlock(tf.keras.layers.Layer):
    def __init__(self, channel, ratio=8):
        super(CBAMBlock, self).__init__()
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.maxpool = tf.keras.layers.GlobalMaxPool2D()
        self.shared_layer_one = tf.keras.layers.Dense(channel//ratio, activation='relu',
                                                        kernel_initializer='he_normal',
                                                        use_bias=True,
                                                        bias_initializer='zeros')
        self.shared_layer_two = tf.keras.layers.Dense(channel, kernel_initializer='he_normal',
                                        use_bias=True, bias_initializer='zeros')
        self.reshape = tf.keras.layers.Reshape((1,1,channel))
        self.multiply = tf.keras.layers.Multiply()
        self.add = tf.keras.layers.Add()
        self.activation = tf.keras.layers.Activation('sigmoid')
        self.concat = tf.keras.layers.Concatenate(axis=3)
        self.conv = tf.keras.layers.Conv2D(filters = 1, kernel_size=(7,7), strides=1,
                                            padding='same', activation='sigmoid', kernel_initializer='he_normal', 
                                            use_bias=False)
        self.channel = channel

    def call(self, inputs, **kwargs):
        #channel_attention
        avg_pool = self.avgpool(inputs)
        avg_pool = self.reshape(avg_pool)
        #assert avg_pool.shape[1:] == (1,1,self.channel)
        avg_pool = self.shared_layer_one(avg_pool)
        #assert avg_pool.shape[1:] == (1,1,self.channel//8)
        avg_pool = self.shared_layer_two(avg_pool)
        #assert avg_pool.shape[1:] == (1,1,self.channel)
        
        max_pool = self.maxpool(inputs)
        max_pool = self.reshape(max_pool)
        #assert max_pool.shape[1:] == (1,1,self.channel)
        max_pool = self.shared_layer_one(max_pool)
        #assert max_pool.shape[1:] == (1,1,self.channel//8)
        max_pool = self.shared_layer_two(max_pool)
        #assert max_pool.shape[1:] == (1,1,self.channel)
        
        channel_feature = self.add([avg_pool,max_pool])
        channel_feature = self.activation(channel_feature)
        channel_feature = self.multiply([inputs, channel_feature])
        
        #spatial_attention
        avg_pool = tf.reduce_mean(channel_feature, axis=[3], keepdims=True)
        #assert avg_pool.shape[-1] == 1
        max_pool = tf.reduce_max(channel_feature, axis=[3], keepdims=True)
        #assert max_pool.shape[-1] == 1
        concat = self.concat([avg_pool, max_pool])
        #assert concat.shape[-1] == 2
        spatial_feature = self.conv(concat)
        cbam_feature = self.multiply([channel_feature, spatial_feature])
        
        return cbam_feature

class MBConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate):
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.conv1 = tf.keras.layers.Conv2D(filters=in_channels * expansion_factor,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                      strides=stride,
                                                      padding="same",
                                                      use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        #self.cbam = CBAMBlock(channel=max(1, int(in_channels * expansion_factor)))
        self.se = SEBlock(input_channels=in_channels * expansion_factor)
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=drop_connect_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.swish(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        x = self.se(x)
        #x = self.cbam(x)
        x = tf.nn.swish(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_rate:
                x = self.dropout(x, training=training)
            x = tf.keras.layers.add([x, inputs])
        return x


def build_mbconv_block(in_channels, out_channels, layers, stride, expansion_factor, k, drop_connect_rate):
    block = tf.keras.Sequential()
    for i in range(layers):
        if i == 0:
            block.add(MBConv(in_channels=in_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=stride,
                             k=k,
                             drop_connect_rate=drop_connect_rate))
        else:
            block.add(MBConv(in_channels=out_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=1,
                             k=k,
                             drop_connect_rate=drop_connect_rate))
    return block


class EfficientNet(tf.keras.Model):
    def __init__(self, width_coefficient, depth_coefficient, dropout_rate, drop_connect_rate=0.2):
        super(EfficientNet, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=round_filters(32, width_coefficient),
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.block1 = build_mbconv_block(in_channels=round_filters(32, width_coefficient),
                                         out_channels=round_filters(16, width_coefficient),
                                         layers=round_repeats(1, depth_coefficient),
                                         stride=1,
                                         expansion_factor=1, k=3, drop_connect_rate=drop_connect_rate)
        self.block2 = build_mbconv_block(in_channels=round_filters(16, width_coefficient),
                                         out_channels=round_filters(24, width_coefficient),
                                         layers=round_repeats(2, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)
        self.block3 = build_mbconv_block(in_channels=round_filters(24, width_coefficient),
                                         out_channels=round_filters(40, width_coefficient),
                                         layers=round_repeats(2, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
        self.block4 = build_mbconv_block(in_channels=round_filters(40, width_coefficient),
                                         out_channels=round_filters(80, width_coefficient),
                                         layers=round_repeats(3, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)
        self.block5 = build_mbconv_block(in_channels=round_filters(80, width_coefficient),
                                         out_channels=round_filters(112, width_coefficient),
                                         layers=round_repeats(3, depth_coefficient),
                                         stride=1,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
        self.block6 = build_mbconv_block(in_channels=round_filters(112, width_coefficient),
                                         out_channels=round_filters(192, width_coefficient),
                                         layers=round_repeats(4, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)
        self.block7 = build_mbconv_block(in_channels=round_filters(192, width_coefficient),
                                         out_channels=round_filters(320, width_coefficient),
                                         layers=round_repeats(1, depth_coefficient),
                                         stride=1,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)

        self.conv2 = tf.keras.layers.Conv2D(filters=round_filters(1280, width_coefficient),
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES,
                                        activation=tf.keras.activations.softmax)
        self.multiply = tf.keras.layers.Multiply()

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.swish(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        #print(x.shape)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.swish(x)
        #x = self.pool(x)
        #x = self.dropout(x, training=training)
        #x = self.fc(x)
        #print(x.shape) 
        #kaggle attention model
        attn_layer = tf.keras.layers.Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(tf.keras.layers.Dropout(0.5)(x))
        attn_layer = tf.keras.layers.Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
        attn_layer = tf.keras.layers.Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
        attn_layer = tf.keras.layers.Conv2D(1, kernel_size = (1,1), padding = 'valid', activation = 'sigmoid')(attn_layer)
        # fan it out to all of the channels
        pt_depth = x.shape[-1]
        up_c2_w = np.ones((1, 1, 1, pt_depth))
        up_c2 = tf.keras.layers.Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', activation = 'linear', use_bias = False, weights = [up_c2_w])
        up_c2.trainable = False
        attn_layer = up_c2(attn_layer)
        mask_features = self.multiply([attn_layer, x])
        gap_features = tf.keras.layers.GlobalAveragePooling2D()(mask_features)
        gap_mask = tf.keras.layers.GlobalAveragePooling2D()(attn_layer)
        # to account for missing values from the attention model
        gap = tf.keras.layers.Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
        gap_dr = tf.keras.layers.Dropout(0.25)(gap)
        dr_steps = tf.keras.layers.Dropout(0.25)(tf.keras.layers.Dense(128, activation = 'relu')(gap_dr))
        #out_layer = tf.keras.layers.Dense(5, activation = 'softmax')(dr_steps)
        out_layer = self.fc(dr_steps)
        return out_layer


def get_efficient_net(width_coefficient, depth_coefficient, resolution, dropout_rate):
    net = EfficientNet(width_coefficient=width_coefficient,
                       depth_coefficient=depth_coefficient,
                       dropout_rate=dropout_rate)

    return net


def efficient_net_b0():
    return get_efficient_net(1.0, 1.0, 224, 0.2)


def efficient_net_b1():
    return get_efficient_net(1.0, 1.1, 240, 0.2)


def efficient_net_b2():
    return get_efficient_net(1.1, 1.2, 260, 0.3)


def efficient_net_b3():
    return get_efficient_net(1.2, 1.4, 300, 0.3)


def efficient_net_b4():
    return get_efficient_net(1.4, 1.8, 380, 0.4)


def efficient_net_b5():
    return get_efficient_net(1.6, 2.2, 456, 0.4)


def efficient_net_b6():
    return get_efficient_net(1.8, 2.6, 528, 0.5)


def efficient_net_b7():
    return get_efficient_net(2.0, 3.1, 600, 0.5)


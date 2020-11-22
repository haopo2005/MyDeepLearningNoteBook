import tensorflow as tf
import numpy as np
import os
import cv2
#import matplotlib
#matplotlib.use('TKAgg')
#import matplotlib.pyplot as plt

def get_files(file_dir):
    # file_dir:文件夹路径
    # return：乱序后的图片和标签列表
    
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            name = file.split(sep='.')
            if name[0] == 'cat':
                cats.append(file_dir + 'cats/' + file)
                label_cats.append(0)
            elif name[0] == 'dog':
                dogs.append(file_dir + 'dogs/' + file)
                label_dogs.append(1)
    print('There are %d cats\n There are %d dogs' % (len(cats), len(dogs)))
    
    #打乱顺序
    image_list = np.hstack((cats, dogs)) #hstack((a,b))的功能是将a和b以水平的方式连接
    label_list = np.hstack((label_cats, label_dogs))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()#temp的大小是2×25000，经过转置（变成25000×2）
    np.random.shuffle(temp)
    
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(float(i)) for i in label_list]
    
    return image_list, label_list

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # image,label：一个batch内图像和标签的list
    # imageW,imageH：图片的宽高
    # batch_size：每个batch内有多少张图片
    # capacity： 设置tensor列表的容量，需要大于batch_size？
    # return：图像和标签的batch
    
    #将python.list转换成tensorflow能识别的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    
    #生成队列，用多少存多少到内存，不再是全部读取到内存，再去Batch
    #tf.train.slice_input_producer是一个tensor生成器，作用是按照设定，
    #每次从一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列。
    input_queue = tf.train.slice_input_producer([image, label])
    image_contents = tf.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    #统一图片大小,将uint8转换成了float32，导致后门matplotlib展示图片的时候颜色异常
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    #image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size,num_threads=64,capacity=capacity)
    #image_batch是一个4D的tensor，[batch, width, height, channels]，label_batch是一个1D的tensor，[batch]
    return image_batch, label_batch

'''
BATCH_SIZE = 2
CAPACITY = 256
IMG_W = 208
IMG_H = 208
train_dir = './data/train/'
image_list, label_list = get_files(train_dir)
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and i < 1:
            img, label = sess.run([image_batch, label_batch])

            for j in np.arange(BATCH_SIZE):
                print("label: %d" % label[j])
                plt.imshow(img[j, :, :, :])
                plt.show()
            i += 1
    except tf.errors.OutOfRangeError:
        print("done!")
    finally:
        coord.request_stop()
    coord.join(threads)
'''
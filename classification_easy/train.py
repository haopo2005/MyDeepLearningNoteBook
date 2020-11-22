import time
from input_data import *
from squeeze_net_v1 import *
#from model import *
#from squeeze_net_v0 import *
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np

# 训练模型
def training():
    N_CLASSES = 2
    IMG_SIZE = 100
    BATCH_SIZE = 28
    CAPACITY = 200
    MAX_STEP = 15000
    LEARNING_RATE = 5e-5
    
    # 训练图片读取
    image_dir = './data/train/'
    logs_dir = 'models'
    
    sess = tf.Session()
    image_list, label_list = get_files(image_dir) #获取图片路径和标签列表
    image_train_batch, label_train_batch = get_batch(image_list, label_list, IMG_SIZE, IMG_SIZE, BATCH_SIZE, CAPACITY)
    train_logits = squeeze_net_model_v1(image_train_batch, True, N_CLASSES)#squeeze_net_model_v0, inference
    train_loss = losses(train_logits, label_train_batch)
    train_acc = evaluation(train_logits, label_train_batch)
    
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(train_loss)
    #train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(train_loss)
    
    var_list = tf.trainable_variables()
    paras_count = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_list])
    print('参数数目:%d' % sess.run(paras_count), end='\n\n')
    
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    s_t = time.time()
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break

            _, loss, acc = sess.run([train_op, train_loss, train_acc])

            if step % 100 == 0:  # 实时记录训练过程并显示
                runtime = time.time() - s_t
                print('Step: %6d, loss: %.8f, accuracy: %.2f%%, time:%.2fs, time left: %.2fhours'
                      % (step, loss, acc * 100, runtime, (MAX_STEP - step) * runtime / 360000))
                s_t = time.time()

            if step % 1000 == 0 or step == MAX_STEP - 1:  # 保存检查点
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()
    
    
# inference
def eval_random_ten_imgs():
    N_CLASSES = 2
    IMG_SIZE = 100
    BATCH_SIZE = 1
    CAPACITY = 200
    MAX_STEP = 10

    test_dir = './data/test/'
    logs_dir = 'models'

    sess = tf.Session()

    image_list, label_list = get_files(test_dir) #获取图片路径和标签列表
    image_train_batch, label_train_batch = get_batch(image_list, label_list, IMG_SIZE, IMG_SIZE, BATCH_SIZE, CAPACITY)
    train_logits = squeeze_net_model_v1(image_train_batch, False, N_CLASSES) #squeeze_net_model_v0, inference
    train_logits = tf.nn.softmax(train_logits)  # ！！！！关键，用softmax转化为百分比数值

    # 载入检查点
    saver = tf.train.Saver()
    print('\n载入检查点...')
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功，global_step = %s\n' % global_step)
    else:
        print('没有找到检查点')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in range(MAX_STEP): #随机挑选10张inference下，展示有问题
            if coord.should_stop():
                break

            image, prediction = sess.run([image_train_batch, train_logits])
            max_index = np.argmax(prediction)
            if max_index == 0:
                label = '%.2f%% is a cat.' % (prediction[0][0] * 100)
            else:
                label = '%.2f%% is a dog.' % (prediction[0][1] * 100)

            plt.imshow(image[0])
            plt.title(label)
            plt.show()

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()

#挑选指定路径的展示下
def eval_one_img(img_path):
    N_CLASSES = 2
    IMG_SIZE = 100
    logs_dir = 'models'
    
    sess = tf.Session()
    image = tf.placeholder(tf.float32, shape=(None, IMG_SIZE, IMG_SIZE, 3))
    train_logits = squeeze_net_model_v1(image, False, N_CLASSES)#squeeze_net_model_v0, inference, full_conv
    train_logits = tf.nn.softmax(train_logits)  # ！！！！关键，用softmax转化为百分比数值
    
    # 载入检查点
    saver = tf.train.Saver()
    print('\n载入检查点...')
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功，global_step = %s\n' % global_step)
    else:
        print('没有找到检查点')
   
    img = cv2.imread(img_path, 1)
    img_resize = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    img_resize = np.array(img_resize).reshape(1, IMG_SIZE, IMG_SIZE, 3)
      
    prediction = sess.run(train_logits, feed_dict={image:img_resize})
    max_index = np.argmax(prediction)
    if max_index == 0:
        label = '%.2f%% is a cat.' % (prediction[0][0] * 100)
    else:
        label = '%.2f%% is a dog.' % (prediction[0][1] * 100)
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(label)
    plt.show()

if __name__ == '__main__':
    training()
    #eval_random_ten_imgs()
    #eval_one_img(sys.argv[1])
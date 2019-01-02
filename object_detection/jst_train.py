from main.model.squeezeDet import  SqueezeDet
from main.model.dataGenerator import generator_from_data_path
import keras.backend as K
from keras import optimizers
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from main.model.modelLoading import load_only_possible_weights
from main.model.multi_gpu_model_checkpoint import  ModelCheckpointMultiGPU
import argparse
import os
import gc
from keras.utils import multi_gpu_model
import pickle
from main.config.create_config import load_dict

#global variables can be set by optional arguments
#TODO: Makes proper variables in train() instead of global arguments.
img_file = "img_train.txt"
gt_file = "gt_train.txt"
log_dir_name = './log'
init_file = "kitti.hdf5"
EPOCHS = 100
STEPS = None
OPTIMIZER = "adam"
CUDA_VISIBLE_DEVICES = "0"
GPUS = 1
PRINT_TIME = 0
REDUCELRONPLATEAU = True
VERBOSE=False

CONFIG = "squeeze.config"


def train():
    """Def trains a Keras model of SqueezeDet and stores the checkpoint after each epoch
    """


    #create subdirs for logging of checkpoints and tensorboard stuff
    checkpoint_dir = log_dir_name +"/checkpoints"
    tb_dir = log_dir_name +"/tensorboard"



    #delete old checkpoints and tensorboard stuff
    if tf.gfile.Exists(checkpoint_dir):
        tf.gfile.DeleteRecursively(checkpoint_dir)

    if tf.gfile.Exists(tb_dir):
        tf.gfile.DeleteRecursively(tb_dir)

    tf.gfile.MakeDirs(tb_dir)
    tf.gfile.MakeDirs(checkpoint_dir)




    #open files with images and ground truths files with full path names
    with open(img_file) as imgs:
        img_names = imgs.read().splitlines()
    imgs.close()
    with open(gt_file) as gts:
        gt_names = gts.read().splitlines()
    gts.close()


    #create config object
    cfg = load_dict(CONFIG)
    print('cfg.ANCHORS:',cfg.ANCHORS)
    #add stuff for documentation to config
    cfg.img_file = img_file
    cfg.gt_file = gt_file
    cfg.images = img_names
    cfg.gts = gt_names
    cfg.init_file = init_file
    cfg.EPOCHS = EPOCHS
    cfg.OPTIMIZER = OPTIMIZER
    cfg.CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES
    cfg.GPUS = GPUS
    cfg.REDUCELRONPLATEAU = REDUCELRONPLATEAU
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    #scale batch size to gpus
    cfg.BATCH_SIZE = cfg.BATCH_SIZE * GPUS
    #compute number of batches per epoch
    nbatches_train, mod = divmod(len(img_names), cfg.BATCH_SIZE)

    if STEPS is not None:
        nbatches_train = STEPS

    cfg.STEPS = nbatches_train

    #print some run info
    print("Number of images: {}".format(len(img_names)))
    print("Number of epochs: {}".format(EPOCHS))
    print("Number of batches: {}".format(nbatches_train))
    print("Batch size: {}".format(cfg.BATCH_SIZE))

    #tf config and session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    K.set_session(sess)


    #instantiate model
    squeeze = SqueezeDet(cfg)


    #callbacks
    cb = []


    #set optimizer
    #multiply by number of workers do adjust for increased batch size
    if OPTIMIZER == "adam":
        opt = optimizers.Adam(lr=0.001 * GPUS,  clipnorm=cfg.MAX_GRAD_NORM)
        cfg.LR= 0.001 * GPUS
    if OPTIMIZER == "rmsprop":
        opt = optimizers.RMSprop(lr=0.001 * GPUS,  clipnorm=cfg.MAX_GRAD_NORM)
        cfg.LR= 0.001 * GPUS
    if OPTIMIZER == "adagrad":
        opt = optimizers.Adagrad(lr=1.0 * GPUS,  clipnorm=cfg.MAX_GRAD_NORM)
        cfg.LR = 1 * GPUS
    #use default is nothing is given
    else:
        # create sgd with momentum and gradient clipping
        opt = optimizers.SGD(lr=cfg.LEARNING_RATE * GPUS, decay=0, momentum=cfg.MOMENTUM,
                             nesterov=False, clipnorm=cfg.MAX_GRAD_NORM)
        cfg.LR = cfg.LEARNING_RATE  * GPUS
        print("Learning rate: {}".format(cfg.LEARNING_RATE * GPUS))

    #save config file to log dir
    with open( log_dir_name  +'/config.pkl', 'wb') as f:
        pickle.dump(cfg, f, pickle.HIGHEST_PROTOCOL)
        
    #add tensorboard callback
    tbCallBack = TensorBoard(log_dir=tb_dir, histogram_freq=0,write_graph=True, write_images=True)
    cb.append(tbCallBack)

    #if flag was given, add reducelronplateu callback
    #当学习停滞时，减少2倍或10倍的学习率常常能获得较好的效果。该回调函数检测指标的情况，如果在5个epoch中看不到模型性能提升，则减少学习率
    if REDUCELRONPLATEAU:
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,verbose=1,patience=5, min_lr=0.0)
        cb.append(reduce_lr)

    #print keras model summary
    if VERBOSE:
        print(squeeze.model.summary())

    if init_file != "none":
        print("Weights initialized by name from {}".format(init_file))
        load_only_possible_weights(squeeze.model, init_file, verbose=VERBOSE)
    #create train generator
    train_generator = generator_from_data_path(img_names, gt_names, config=cfg)
    # add a checkpoint saver
    ckp_saver = ModelCheckpoint(checkpoint_dir + "/model.{epoch:02d}-{loss:.2f}.hdf5", monitor='loss', verbose=0,
                                save_best_only=False,
                                save_weights_only=True, mode='auto', period=1)
    cb.append(ckp_saver)
    print("Using single GPU")
    #compile model from squeeze object, loss is not a function of model directly
    squeeze.model.compile(optimizer=opt,
                          loss=[squeeze.loss], metrics=[squeeze.loss_without_regularization, squeeze.bbox_loss, squeeze.class_loss, squeeze.conf_loss])

    #actually do the training
    squeeze.model.fit_generator(train_generator, epochs=EPOCHS,
                                    steps_per_epoch=nbatches_train, callbacks=cb)
    gc.collect()


if __name__ == "__main__":
    #parse arguments
    parser = argparse.ArgumentParser(description='Train squeezeDet model.')
    parser.add_argument("--steps",  type=int, help="steps per epoch. DEFAULT: #imgs/ batch size")
    parser.add_argument("--epochs", type=int, help="number of epochs. DEFAULT: 100")
    parser.add_argument("--optimizer",  help="Which optimizer to use. DEFAULT: SGD with Momentum and lr decay OPTIONS: SGD, ADAM")
    parser.add_argument("--logdir", help="dir with checkpoints and loggings. DEFAULT: ./log")
    parser.add_argument("--img", help="file of full path names for the training images. DEFAULT: img_train.txt")
    parser.add_argument("--gt", help="file of full path names for the corresponding training gts. DEFAULT: gt_train.txt")
    parser.add_argument("--gpu",  help="which gpu to use. DEFAULT: 0")
    parser.add_argument("--gpus", type=int,  help="number of GPUS to use when using multi gpu support. Overwrites gpu flag. DEFAULT: 1")
    parser.add_argument("--init",  help="keras checkpoint to start training from. If argument is none, training starts from the beginnin. DEFAULT: init_weights.h5")
    parser.add_argument("--resume", type=bool, help="Resumes training and does not delete old dirs. DEFAULT: False")
    parser.add_argument("--reducelr", type=bool, help="Add ReduceLrOnPlateu callback to training. DEFAULT: True")
    parser.add_argument("--verbose", type=bool,  help="Prints additional information. DEFAULT: False")
    parser.add_argument("--config",   help="Dictionary of all the hyperparameters. DEFAULT: squeeze.config")

    args = parser.parse_args()


    #set global variables
    if args.img is not None:
        img_file = args.img
    if args.gt is not None:
        gt_file = args.gt
    if args.logdir is not None:
        log_dir_name = args.logdir
    if args.gpu is not None:
        CUDA_VISIBLE_DEVICES = args.gpu
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.steps is not None:
        STEPS = args.steps
    if args.optimizer is not None:
        OPTIMIZER = args.optimizer.lower()
    if args.init is not None:
        init_file = args.init
    if args.gpus is not None:
        GPUS= args.gpus
    if args.reducelr is not None:
        REDUCELRONPLATEAU = args.reducelr
    if args.verbose is not None:
        VERBOSE=args.verbose
    if args.config is not None:
        CONFIG = args.config

    train()

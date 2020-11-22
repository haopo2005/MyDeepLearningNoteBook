import tensorflow as tf
from configuration import save_model_dir
from prepare_data import generate_datasets
from train import get_model, process_features
import os
import math

if __name__ == '__main__':

    # GPU settings
    os.environ["CUDA_VISIBLE_DEVICES"]="4"
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the original_dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()
    # load the model
    model = get_model()
    model.load_weights('saved_model_b7/saved_model_b7model')
    #model = tf.saved_model.load('saved_model_b7/saved_model_b7model')

    # Get the accuracy on the test set
    loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    pred_score = []
    label_list = []
    
    # @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        for i in range(len(labels)):
            pred_score.append(predictions[i])
            label_list.append(labels[i])
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    for features in test_dataset:
        test_images, test_labels = process_features(features, data_augmentation=False)
        test_step(test_images, test_labels)
        '''
        print("loss: {:.5f}, test accuracy: {:.5f}".format(test_loss.result(),
                                                           test_accuracy.result()))
        '''
    print("The accuracy on test set is: {:.3f}%".format(test_accuracy.result()*100))
    
    # compute softmax loss
    result = 0
    for i in range(len(label_list)):
        for j in range(4):
            if label_list[i] == j:
                result += math.log(pred_score[i][j])
    result = -result/len(label_list)
    print('rb loss: {:.8f}'.format(result))

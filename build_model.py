#Hua-an Tseng, huaantseng@gmail.com

import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from keras.models import Sequential, load_model, Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Activation, Flatten, Concatenate, advanced_activations
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, UpSampling2D, SpatialDropout2D
from keras.layers import BatchNormalization, concatenate, GaussianNoise
from keras.layers.core import Lambda
import keras.optimizers as optimizers
from keras import callbacks, utils
from keras import regularizers
from keras import backend as K

from keras_tqdm import TQDMNotebookCallback as TQDMCallback
# from keras_tqdm import TQDMCallback

import os
import time

def callback_list(model_folder,notebook_filename,model_id,validation=0):
    # earily stopping
    if validation == 1:
        earlyStopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    else:
        earlyStopping = callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='auto')

    # save the best weight
    filepath = "%s%s_%s_best_weight.weight" % (model_folder, notebook_filename, model_id)
    if validation == 1:
        checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                               mode='auto', period=1)
    else:
        checkpoint = callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto',
                                               period=1)

    # tensorboard
    current_log_dir = "%sTensorboard" % (model_folder)
    # tbCallBack = callbacks.TensorBoard(log_dir=current_log_dir, histogram_freq=0, write_graph=True, write_images=True)
    tbCallBack = callbacks.TensorBoard(log_dir=current_log_dir, histogram_freq=1, write_graph=True, write_images=True,
                                       write_grads=True)
    print("tensorboard --logdir foo:%s" % current_log_dir)

    return [checkpoint,TQDMCallback()]

class SGDRScheduler(callbacks.Callback):
    #https://gist.github.com/jeremyjordan/5a222e04bb78c242f5763ad40626c452
    '''Cosine annealing learning rate scheduler with periodic restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)


def load_model_weights(model_filename,weights_filename):
    model = load_model(model_filename)
    model.load_weights(weights_filename)
    model_epoch_time = os.path.getmtime(model_filename)
    weights_epoch_time = os.path.getmtime(weights_filename)
    print("Save time:")
    print("\tModel: %s" % time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(model_epoch_time)))
    print("\tWeights: %s" % time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(weights_epoch_time)))
    return model

def load_saved_model(model_filename):
    custom_objects = {
        'loss_function_cross_entropy_loss_1': loss_function_cross_entropy_loss_1,
        'loss_function_cross_entropy_loss_3': loss_function_cross_entropy_loss_3,
        'loss_function_dice_coefficient_loss': loss_function_dice_coefficient_loss,
        'loss_function_jaccard_index_loss': loss_function_jaccard_index_loss,
        'loss_function_jaccard_distance_keras': loss_function_jaccard_distance_keras,
        'loss_function_jaccard_distance_keras_3': loss_function_jaccard_distance_keras_3,
        'loss_function_mix_jd_ce_3': loss_function_mix_jd_ce_3,
        'metrics_jaccard_distance_keras': metrics_jaccard_distance_keras
    }
    model = load_model(model_filename,custom_objects=custom_objects)
    model_epoch_time = os.path.getmtime(model_filename)
    print("Save time:")
    print("\tModel: %s" % time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(model_epoch_time)))
    return model

def loss_function_cross_entropy_loss_1(y_true, y_pred):
    y_true = tf.argmax(y_true,axis=1)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

def loss_function_cross_entropy_loss_3(y_true, y_pred):
    y_true = tf.argmax(y_true,axis=3)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

def loss_function_dice_coefficient_loss(y_true, y_pred):
    # the last activation requires "sigmoid"
    eps = 1
    dice_coefficient = 2*(tf.reduce_sum(y_true*y_pred)+eps)/(tf.reduce_sum(y_true)+tf.reduce_sum(y_pred)+eps)
    return 1-dice_coefficient

def loss_function_jaccard_index_loss(y_true, y_pred):
    # the last activation requires "sigmoid"
    eps = 1
    jaccard_index = (tf.reduce_sum(y_true*y_pred)+eps)/(tf.reduce_sum(y_true)+tf.reduce_sum(y_pred)-tf.reduce_sum(y_true*y_pred)+eps)
    return 1-jaccard_index

def loss_function_jaccard_distance_keras(y_true, y_pred, smooth=100):
    # https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """


    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def loss_function_jaccard_distance_keras_3(y_true, y_pred, smooth=100):
    # https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """

    weight = K.constant([0.2,0.8])
    weight = weight/K.sum(weight)
    #y_true = y_true * weight
    #y_pred = y_pred * weight
    intersection = K.sum(K.abs(weight*y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(weight*y_true) + K.abs(weight*y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def loss_function_mix_jd_ce_3(y_true, y_pred, smooth=100):
    # jaccard_distance + cross_entropy_loss

    # jaccard_distance
    weight = K.constant([0.2, 0.8])
    weight = weight / K.sum(weight)
    intersection = K.sum(K.abs(weight * y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(weight * y_true) + K.abs(weight * y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd =  (1 - jac) * smooth
    # cross_entropy_loss
    y_true = tf.argmax(y_true, axis=3)
    ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
    return 0.5*jd+0.5*ce

def metrics_jaccard_distance_keras(y_true, y_pred, smooth=100):
    # https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """
    y_true = K.cast(K.greater(K.flatten(y_true), 0.5), 'float32')
    y_pred = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def loss_jaccard_distance_tensorflow(y_true, y_pred, smooth=100):
    # https://stackoverflow.com/questions/49284455/keras-custom-function-implementing-jaccard
    """ Calculates mean of Jaccard distance as a loss function """
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd =  (1 - jac) * smooth
    return tf.reduce_mean(jd)

class model_unet():
    def __init__(self,pixel_number=32,frame_number=1,catagory_number=2,conv_dropout=0.6,dense_dropout=0.6,l2_value=0.001, conv_dilation_rate=1, channels_order='channels_first'):

        # training parameters
        self.current_batch_size = 16
        self.current_epochs = 1000
        self.sample_size = [64*1024]

        # model parameters
        self.pixel_number = pixel_number
        self.frame_number = frame_number
        self.catagory_number = catagory_number
        self.conv_dropout = conv_dropout
        self.dense_dropout = dense_dropout
        self.l2_value = l2_value
        self.conv_dilation_rate = conv_dilation_rate
        self.channels_order = channels_order
        if self.channels_order == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3



        def conv_layer(input_layer, kernel_number=64, kernel_size=(3,3), dilation_rate=self.conv_dilation_rate, batch_normalization=True, dropout=True, conv_dropout=self.conv_dropout):
            output_layer = Conv2D(kernel_number, kernel_size,
                                  padding="same",
                                  dilation_rate=dilation_rate,
                                  data_format=self.channels_order,
                                  #kernel_initializer='he_uniform',
                                  #bias_initializer='he_uniform'
                                  )(input_layer)  # kernel_initializer='he_uniform', bias_initializer='he_uniform'
            output_layer = advanced_activations.PReLU()(output_layer)
            if batch_normalization:
                output_layer = BatchNormalization(axis=self.channel_axis)(output_layer)
            
            if dropout:
                output_layer = SpatialDropout2D(conv_dropout)(output_layer)

            return output_layer


        # model
        #K.set_floatx('float16')
        #K.set_epsilon(1e-4)

        if self.channel_axis == 3:
            input = Input(shape=(pixel_number, pixel_number, frame_number)) # NHWC
        else:
            input = Input(shape=(frame_number, pixel_number, pixel_number)) # NCHW

        g_input = GaussianNoise(0.1)(input)

        down_conv_1 = conv_layer(g_input, kernel_number=64)
        down_conv_1 = conv_layer(down_conv_1, kernel_number=64)
        down_conv_1 = conv_layer(down_conv_1, kernel_number=64, kernel_size=(1,1))

        pool_1 = MaxPooling2D(pool_size=(2, 2),data_format=self.channels_order)(down_conv_1)

        down_conv_2 = conv_layer(pool_1, kernel_number=128)
        down_conv_2 = conv_layer(down_conv_2, kernel_number=128)
        down_conv_2 = conv_layer(down_conv_2, kernel_number=128)
        down_conv_2 = conv_layer(down_conv_2, kernel_number=128, kernel_size=(1, 1))

        pool_2 = MaxPooling2D(pool_size=(2, 2),data_format=self.channels_order)(down_conv_2)

        down_conv_3 = conv_layer(pool_2, kernel_number=256)
        down_conv_3 = conv_layer(down_conv_3, kernel_number=256)
        down_conv_3 = conv_layer(down_conv_3, kernel_number=256)
        down_conv_3 = conv_layer(down_conv_3, kernel_number=256, kernel_size=(1, 1))

        pool_3 = MaxPooling2D(pool_size=(2, 2),data_format=self.channels_order)(down_conv_3)

        down_conv_4 = conv_layer(pool_3, kernel_number=512)
        down_conv_4 = conv_layer(down_conv_4, kernel_number=512)
        down_conv_4 = conv_layer(down_conv_4, kernel_number=512)
        down_conv_4 = conv_layer(down_conv_4, kernel_number=512, kernel_size=(1, 1))

        pool_4 = MaxPooling2D(pool_size=(2, 2),data_format=self.channels_order)(down_conv_4)

        down_conv_fin = conv_layer(pool_4, kernel_number=1024)
        down_conv_fin = conv_layer(down_conv_fin, kernel_number=1024)
        down_conv_fin = conv_layer(down_conv_fin, kernel_number=1024)

        up_conv_4 = UpSampling2D(size=(2, 2),data_format=self.channels_order)(down_conv_fin)
        up_conv_4 = concatenate([up_conv_4, down_conv_4], axis=self.channel_axis)

        up_conv_4 = conv_layer(up_conv_4, kernel_number=512)
        up_conv_4 = conv_layer(up_conv_4, kernel_number=512)

        up_conv_3 = UpSampling2D(size=(2, 2),data_format=self.channels_order)(up_conv_4)
        up_conv_3 = concatenate([up_conv_3, down_conv_3], axis=self.channel_axis)

        up_conv_3 = conv_layer(up_conv_3, kernel_number=256)
        up_conv_3 = conv_layer(up_conv_3, kernel_number=256)

        up_conv_2 = UpSampling2D(size=(2, 2),data_format=self.channels_order)(up_conv_3)
        up_conv_2 = concatenate([up_conv_2, down_conv_2], axis=self.channel_axis)

        up_conv_2 = conv_layer(up_conv_2, kernel_number=128)
        up_conv_2 = conv_layer(up_conv_2, kernel_number=128)

        up_conv_1 = UpSampling2D(size=(2, 2),data_format=self.channels_order)(up_conv_2)
        up_conv_1 = concatenate([up_conv_1, down_conv_1], axis=self.channel_axis)

        up_conv_1 = conv_layer(up_conv_1, kernel_number=64)
        up_conv_1 = conv_layer(up_conv_1, kernel_number=64)

        output = conv_layer(up_conv_1, kernel_number=catagory_number, kernel_size=(1,1))
        #output = advanced_activations.PReLU()(output)
        output = Activation('sigmoid')(output)
        #output = conv_layer(output, kernel_number=catagory_number, kernel_size=(1, 1))
        #output = Activation('tanh')(output)
        # output = Activation('softmax')(output)

        self.model = Model(inputs=input, outputs=output)

        self.opt = optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
        # opt = optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
        # self.opt = optimizers.Adagrad(lr=0.001, epsilon=1e-08, decay=1e-4)
        # opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-4)
        # opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

    def load_model_weights(self,model_filename,weights_filename):
        model = load_model(model_filename)
        model.load_weights(weights_filename)
        model_epoch_time = os.path.getmtime(model_filename)
        weights_epoch_time = os.path.getmtime(weights_filename)
        print("Save time:")
        print("\tModel: %s" % time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(model_epoch_time)))
        print("\tWeights: %s" % time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(weights_epoch_time)))
        self.model = model
        self.pixel_number = model.input_shape[1]
        self.frame_number = model.input_shape[3]
        self.catagory_number = model.output_shape[3]



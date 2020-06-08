from __future__ import print_function
import keras
from keras.layers import Dense, Conv1D, BatchNormalization, Activation, Dropout
from keras.layers import AveragePooling2D, Input, Flatten, MaxPooling2D, Cropping1D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model, Sequential
from keras.utils import to_categorical, plot_model
from keras.datasets import cifar10
from keras import optimizers
from keras import metrics
import keras.backend as K

import tensorflow as tf
#tf.enable_v2_behavior()
#tf.enable_eager_execution()
print('eagerly?', tf.executing_eagerly())

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils import class_weight
from sklearn.metrics import log_loss

import numpy as np
import os

def hot_encode_seq(let):
    if let=='A':
        return([1,0,0,0])
    elif let=='T':
        return([0,1,0,0])
    elif let=='C':
        return([0,0,1,0])
    elif let=='G':
        return([0,0,0,1])
    elif let=='O':
        return([0,0,0,0])

def hot_encode_label(let):
    if let=='p':
        return([0,0,0])
    elif let=='b':
        return([1,0,0])
    elif let=='a':
        return([0,1,0])
    elif let=='d':
        return([0,0,1])


def lr_schedule(epoch):
    lr = 0.001
    if epoch == 7:
        lr *= 0.5
    elif epoch == 8:
        lr *= (0.5)**2
    elif epoch == 9:
        lr *= (0.5)**3
    elif epoch == 10:
        lr *= (0.5)**4
    print('Learning rate: ', lr)
    return lr


# TRAINING PARAMETERS
batch_size = 12
num_classes = 3
epochs = 10
data_augmentation = False


def celoss_math(labels_flat, pred):
    # mathematical operations to get crossentropy loss from two batch inputs: y_true and y_pred

    pred = tf.transpose(tf.keras.backend.log(pred))
    labels_flat = tf.cast(labels_flat, pred.dtype)
    # labels_b.shape[0] = N (batch size), labels_b.shape[1] = 5000
    labels_flat = tf.reshape(labels_flat, [labels_flat.shape[0], labels_flat.shape[1]])
    # reshaping with dims 1, 2 because we alrd transposed it so it has dims [5000, N] now
    pred = tf.reshape(pred, [pred.shape[1], pred.shape[2]])

    return tf.math.negative(tf.math.reduce_sum(tf.linalg.diag(tf.keras.backend.dot(labels_flat, pred))))


@tf.function 
def custom_crossentropy_loss(y_true, y_pred):

    mask_b = np.array([True, False, False])
    mask_a = np.array([False, True, False])
    mask_d = np.array([False, False, True])

    eps = tf.constant(1e-15, dtype=tf.float32)
    eps_ = tf.constant(1e-15, dtype=tf.float32)

    labels_b = tf.boolean_mask(y_true, mask_b, axis=2)
    n_b = tf.math.count_nonzero(labels_b)
    labels_a = tf.boolean_mask(y_true, mask_a, axis=2)
    n_a = tf.math.count_nonzero(labels_a)
    labels_d = tf.boolean_mask(y_true, mask_d, axis=2)
    n_d = tf.math.count_nonzero(labels_d)

    y_pred = tf.keras.backend.clip(y_pred, eps, eps_)

    pred_b = tf.boolean_mask(y_pred, mask_b, axis=2)
    pred_a = tf.boolean_mask(y_pred, mask_a, axis=2)
    pred_d = tf.boolean_mask(y_pred, mask_d, axis=2)

    s_b = celoss_math(labels_b, pred_b)
    s_b = tf.math.divide(s_b, tf.cast(n_b, s_b.dtype))

    s_a = celoss_math(labels_a, pred_a)
    s_a = tf.math.divide(s_a, tf.cast(n_a, s_a.dtype))

    s_d = celoss_math(labels_d, pred_d)
    s_d = tf.math.divide(s_d, tf.cast(n_d, s_d.dtype))

    loss = tf.add(s_b, tf.add(s_a, s_d))
    return loss


def my_loss(y_true, y_pred):
    print('y_true.shape', y_true.shape)
    print('y_pred.dtype', y_pred.dtype)
    print('y_true.dtype', y_true.dtype)
    loss=K.mean(K.sum(K.square(y_true-y_pred)))
    print('loss dtype:', K.dtype(loss))
    return loss


def RB_block(inputs,
             num_filters=32,
             kernel_size=11,
             strides=1,
             activation='relu',
             dilation_rate=1):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  dilation_rate=dilation_rate)

    x = inputs

    for layer in range(2):
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = conv(x)

    return x


def spliceAI_model(input_shape, num_classes=3):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = 4

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths

    x = Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(inputs)
    y = Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(inputs)

    # RB 1: 32 11 1
    for stack in range(4):
        x = RB_block(x, num_filters=32, kernel_size=11, strides=1, activation='relu', dilation_rate=1)

    y = keras.layers.add([x, y])

    # RB 2: 32 11 4
    for stack in range(4):
        x = RB_block(x, num_filters=32, kernel_size=11, strides=1, activation='relu', dilation_rate=4)

    y = keras.layers.add([x, y])  

    # RB 3: 32 21 10
    for stack in range(4):
        x = RB_block(x, num_filters=32, kernel_size=21, strides=1, activation='relu', dilation_rate=10)

    x = Conv1D(32, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x)

    # now adding up what was shortcut from the prev layers
    x = keras.layers.add([x, y]) 

    x = Conv1D(3, kernel_size=1, strides=1, padding='same', dilation_rate=1)(x)

    x = Dense(num_classes, activation='softmax')(x)

    outputs = Cropping1D(cropping=(1000, 1000))(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, samples, labels, batch_size=12, dim=(7000, 4), dim_labels=(5000, 3), n_channels=1,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.dim_labels = dim_labels
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.samples = samples
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *self.dim_labels), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #print(self.samples[ID])
            X[i] = self.samples[ID]

            # Store class
            #print(self.labels[ID])
            y[i] = self.labels[ID]

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y


# DATA IMPORT

#transcripts_ = np.loadtxt('/Users/iamqoqao/workspace/splicing/model/principal_transcript/data/transcripts_chunks_chr20', dtype='str', delimiter='\t')
#labels_ = np.loadtxt('/Users/iamqoqao/workspace/splicing/model/principal_transcript/data/labels_chunks_chr20', dtype='str', delimiter='\t')

transcripts_ = np.loadtxt('./transcripts_chunks_chr1', dtype='str', delimiter='\t', max_rows=100)
labels_ = np.loadtxt('./labels_chunks_chr1', dtype='str', delimiter='\t', max_rows=100)

#transcripts_ = transcripts_[0:100]
#labels_ = labels_[0:100]

transcripts = []
labels = []

# hot-encode
for i in range(len(transcripts_)):
    # hot-encode seq
    transcripts.append([np.array(hot_encode_seq(let)) for let in transcripts_[i]])
    # hot-encode labels
    labels.append([np.array(hot_encode_label(x)) for x in labels_[i]])

transcripts = np.array(transcripts)
labels = np.array(labels)

print(np.shape(transcripts), np.shape(labels))

(x_train, x_test, y_train, y_test) = train_test_split(transcripts,
    labels, test_size=0.2)

#transcripts = np.reshape(transcripts, (len(transcripts), 100, 70, 4))
#labels = np.reshape(labels, (len(labels), 100, 50, 3))

#print("after reshaping:", np.shape(transcripts), np.shape(labels))

#print("after categorical:", np.shape(transcripts), np.shape(labels))

partition = {}
partition['train'] = list(range(0,30))
partition['validation'] = list(range(30,36))

labels_dict = {}
samples_dict = {}

for i in range(len(y_train)):
    #print(np.shape(labels[i]), np.shape(transcripts[i]))
    labels_dict[i] = y_train[i]
    samples_dict[i] = x_train[i]

#print('partition:', partition)
#print('labels_dict:', labels_dict)

input_shape = x_train.shape[1:]

params = {'dim': (7000, 4),
          'dim_labels': (5000, 3),
          'batch_size': 12,
          'n_classes': 3,
          'n_channels': 1,
          'shuffle': True}

lr_scheduler = LearningRateScheduler(lr_schedule)

model = spliceAI_model(input_shape=input_shape)

model.compile(loss=custom_crossentropy_loss,
              #optimizer='adam',
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])

print(model.summary())

training_generator = DataGenerator(partition['train'], samples_dict, labels_dict, **params)
validation_generator = DataGenerator(partition['validation'], samples_dict, labels_dict, **params)

#plot_model(model, show_shapes=True, show_layer_names=True, to_file='model_spliceAI2k_flat.png')

#history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[lr_scheduler, metrics], validation_data=(x_test, y_test), shuffle=True)

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    steps_per_epoch=len(samples_dict) // batch_size)
                    #workers=4)

model.save('./model_spliceAI2k_weighted_1')

scores = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
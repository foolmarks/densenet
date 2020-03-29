
# Import modules
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

import numpy as np
import os
import sys
import argparse
import tensorflow as tf


from densenet import densenet



# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


DIVIDER = '-----------------------------------------'


def train(opt,batchsize,learnrate,epochs,keras_hdf5,tboard):

    
    def step_decay(epoch):
        """
        Learning rate scheduler used by callback
        Reduces learning rate depending on number of epochs
        """
        lr = learnrate
        if epoch > 120:
            lr /= 1000
        elif epoch > 90:
            lr /= 100
        elif epoch > 60:
            lr /= 10
        elif epoch > 20:
            lr /= 2
        return lr
    

    '''
    -------------------------------------------------------------------
    DATASET PREPARATION
    50k images used for training, 8k for validation and 2k for evaluation
    -------------------------------------------------------------------
    '''
    print('\nDATASET PREPARATION:')
    # CIFAR10 dataset has 60k images. Training set is 50k, test set is 10k.
    # Each image is 32x32x8bits
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Scale image data from range 0:255 to range 0.0:1.0
    # Also converts train & test data to float from uint8
    x_train = (x_train/255.0).astype(np.float32)
    x_test = (x_test/255.0).astype(np.float32)

    # one-hot encode the labels
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # hold back 2k samples from test set for evaluation
    # Note: this does not guarantee a balanced split across all classes
    x_eval = x_test[8000:]
    y_eval = y_test[8000:]
    x_test = x_test[:8000]
    y_test = y_test[:8000]


    '''
    -------------------------------------------------------------------
    NETWORK 
    Create the model, print its structure
    densenet function arguments are for CIFAR-10/100 use case
    -------------------------------------------------------------------
    '''
    model = densenet(input_shape=(32,32,3),classes=10,k=12,drop_rate=0.2,theta=0.5,weight_decay=1e-4,convlayers=[16,16,16])

    print('\n'+DIVIDER)
    print(' Model Summary')
    print(DIVIDER)
    print(model.summary())
    print("Model Inputs: {ips}".format(ips=(model.inputs)))
    print("Model Outputs: {ops}".format(ops=(model.outputs)))


    '''
    -------------------------------------------------------------------
    CREATE CALLBACKS
    -------------------------------------------------------------------
    '''
    chkpt_call = ModelCheckpoint(filepath=keras_hdf5,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True)
    
    
    tb_call = TensorBoard(log_dir=tboard,                          
                          update_freq='epoch')
    

    lr_scheduler_call = LearningRateScheduler(schedule=step_decay,
                                              verbose=1)

    lr_plateau_call = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                        cooldown=0,
                                        patience=5,
                                        min_lr=0.5e-6)

    callbacks_list = [tb_call, lr_scheduler_call, lr_plateau_call, chkpt_call]


    '''
    -------------------------------------------------------------------
    TRAINING
    Training data will be augmented:
      - random rotation
      - random horiz flip
      - random linear shift up and down
      - random shear & zoom
    -------------------------------------------------------------------
    '''

    data_augment = ImageDataGenerator(rotation_range=10,
                                      horizontal_flip=True,
                                      height_shift_range=0.1,
                                      width_shift_range=0.1,
                                      shear_range=0.1,
                                      zoom_range=0.1)

    train_generator = data_augment.flow(x=x_train,
                                        y=y_train,
                                        batch_size=batchsize,
                                        shuffle=True)
                                  

    # Optimizer
    if (opt=='rms'):
        # RMSprop optimizer
        opt = RMSprop(lr=learnrate)
    else:
        #SGD optimizer with Nesterov momentum as per original paper
        opt = SGD(lr=learnrate, momentum=0.9, nesterov=True)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    print('\n'+DIVIDER)
    print(' Training model with training set..')
    print(' Using',opt,'optimizer..')
    print(DIVIDER)


    # run training
    model.fit(x=train_generator,
              epochs=epochs,
              steps_per_epoch=train_generator.n//train_generator.batch_size,
              validation_data=(x_test, y_test),
              callbacks=callbacks_list,
              verbose=1)


    print("\nTensorBoard can be opened with the command: tensorboard --logdir={dir} --host localhost --port 6006".format(dir=tboard))


    '''
    -------------------------------------------------------------------
    EVALUATION
    -------------------------------------------------------------------
    '''

    print('\n'+DIVIDER)
    print(' Evaluate model accuracy with validation set..')
    print(DIVIDER)

    scores = model.evaluate(x_eval, y_eval, verbose=1)
    print ('Evaluation Loss    : ', scores[0])
    print ('Evaluation Accuracy: ', scores[1])

    return



def run_main():
    
    # log TensorFlow, Python versions
    print('\n'+DIVIDER)
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print(DIVIDER)

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-opt', '--opt',
                    type=str,
                    default='rms',
                    choices=['rms','sgd'],
    	            help='Optimizer for training. Valid choices are rms, sgd. Default is rms')
    ap.add_argument('-b', '--batchsize',
                    type=int,
                    default=100,
    	            help='Training batchsize. Must be an integer. Default is 100.')
    ap.add_argument('-e', '--epochs',
                    type=int,
                    default=1,
    	            help='number of training epochs. Must be an integer. Default is 1.')
    ap.add_argument('-lr', '--learnrate',
                    type=float,
                    default=0.001,
    	            help='optimizer initial learning rate. Must be floating-point value. Default is 0.001')
    ap.add_argument('-kh', '--keras_hdf5',
                    type=str,
                    default='model.hdf5',
    	            help='Path of Keras HDF5 file. Default is ./model.hdf5')
    ap.add_argument('-tb', '--tboard',
                    type=str,
                    default='tb_logs',
    	            help='TensorBoard data folder name. Default is ./tb_logs.')    
    args = ap.parse_args()


    print(' Command line options:')
    print ('--opt          : ',args.opt)
    print ('--batchsize    : ',args.batchsize)
    print ('--learnrate    : ',args.learnrate)
    print ('--epochs       : ',args.epochs)
    print ('--keras_hdf5   : ',args.keras_hdf5)
    print ('--tboard       : ',args.tboard)


    train(args.opt,args.batchsize,args.learnrate,args.epochs,args.keras_hdf5,args.tboard)


if __name__ == '__main__':
    run_main()


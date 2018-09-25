import os
import click
import logging

import keras
import numpy as np
import keras.backend as K

from model import create_model

K.set_image_data_format('channels_last')

"""
Train Model [optional args]
"""
@click.command()
@click.option(
    '-lr', 
    '--learning-rate', 
    default=0.005, 
    help='Learning rate for minimizing loss during training'
)
@click.option(
    '-bz',
    '--batch-size',
    default=32,
    help='Batch size of minibatches to use during training'
)
@click.option(
    '-ne', 
    '--num-epochs', 
    default=50, 
    help='Number of epochs for training model'
)
@click.option(
    '-tb',
    '--tensorboard-vis',
    is_flag=True,
    help='Flag for TensorBoard Visualization'
)
def train(learning_rate, batch_size, num_epochs, tensorboard_vis):
    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    get_gen = lambda x: datagen.flow_from_directory(
            'datasets/caltech_101/{}'.format(x),
            target_size=(64, 64),
            batch_size=batch_size,
            class_mode='categorical')

    train_generator = get_gen('train')
    validation_generator = get_gen('val')
    test_generator = get_gen('test')

    if os.path.exists('models/resnet50.h5'):
        # load model
        resnet50 = keras.models.load_model('models/resnet50.h5')
    else:
        # create model
        resnet50 = create_model(input_shape=(64, 64, 3), classes=101)
        optimizer = keras.optimizers.Adam(learning_rate)

        resnet50.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        resnet50.summary()

    # callbacks for checkpointing and visualization during training
    callbacks = []
    if tensorboard_vis:
        tensorboard_cb = keras.callbacks.TensorBoard(log_dir='./logs')
        callbacks.append(tensorboard_cb)
    if not os.path.isdir('models/ckpts'):
        os.mkdir('models/ckpts')
    saver_cb = keras.callbacks.ModelCheckpoint('models/ckpts/model', verbose=1, period=5, save_best_only=True)
    callbacks.append(saver_cb)

    # train model
    resnet50.fit_generator(train_generator, steps_per_epoch=72000//batch_size, 
                            epochs=num_epochs, verbose=1, 
                            validation_data=validation_generator, 
                            validation_steps=2200//batch_size, 
                            shuffle=True, callbacks=callbacks)
    # save model
    resnet50.save('models/resnet50.h5')

    # evaluate model
    preds = resnet50.evaluate_generator(test_generator, steps=2200//batch_size, verbose=1)

    logging.info('test loss: {}'.format(preds[0]))
    logging.info('test acc: {}'.format(preds[1]))

    keras.utils.plot_model(resnet50, to_file='models/resnet50.png')

def main():
    LOG_FORMAT = '%(levelname)s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level='INFO')
    try:
        train()
    except KeyboardInterrupt:
        print('EXIT')

if __name__ == '__main__':
    main()
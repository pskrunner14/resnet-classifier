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
    '-se',
    '--save-every',
    default=1,
    help='Epoch interval to save model checkpoints during training'
)
@click.option(
    '-tb',
    '--tensorboard-vis',
    is_flag=True,
    help='Flag for TensorBoard Visualization'
)
@click.option(
    '-ps',
    '--print-summary',
    is_flag=True,
    help='Flag for printing summary of the model'
)
def train(learning_rate, batch_size, num_epochs, save_every, tensorboard_vis, print_summary):
    setup_paths()

    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    get_gen = lambda x: datagen.flow_from_directory(
        'datasets/caltech_101/{}'.format(x),
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # generator objects
    train_generator = get_gen('train')
    val_generator = get_gen('val')
    test_generator = get_gen('test')

    if os.path.exists('models/resnet50.h5'):
        # load model
        logging.info('loading pre-trained model')
        resnet50 = keras.models.load_model('models/resnet50.h5')
    else:
        # create model
        logging.info('creating model')
        resnet50 = create_model(input_shape=(64, 64, 3), classes=101)
    
    optimizer = keras.optimizers.Adam(learning_rate)
    resnet50.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    if print_summary:
        resnet50.summary()

    callbacks = configure_callbacks(save_every, tensorboard_vis)

    # train model
    logging.info('training model')
    resnet50.fit_generator(
        train_generator,
        steps_per_epoch=72000//batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_data=val_generator,
        validation_steps=2200//batch_size,
        shuffle=True,
        callbacks=callbacks
    )
    # save model
    logging.info('Saving trained model to `models/resnet50.h5`')
    resnet50.save('models/resnet50.h5')

    # evaluate model
    logging.info('evaluating model')
    preds = resnet50.evaluate_generator(
        test_generator,
        steps=2200//batch_size,
        verbose=1
    )
    logging.info('test loss: {:.4f} - test acc: {:.4f}'.format(preds[0], preds[1]))

    keras.utils.plot_model(resnet50, to_file='models/resnet50.png')

"""
Configure Callbacks for Training
"""
def configure_callbacks(save_every=1, tensorboard_vis=False):
    # checkpoint models only when `val_loss` impoves
    saver = keras.callbacks.ModelCheckpoint(
        'models/ckpts/model.ckpt',
        monitor='val_loss',
        save_best_only=True,
        period=save_every,
        verbose=1
    )
    
    # reduce LR when `val_loss` plateaus
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        verbose=1,
        min_lr=1e-10
    )

    # early stopping when `val_loss` stops improving
    early_stopper = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        min_delta=0, 
        patience=10, 
        verbose=1
    )

    callbacks = [saver, reduce_lr, early_stopper]

    if tensorboard_vis:
        # tensorboard visualization callback
        tensorboard_cb = keras.callbacks.TensorBoard(
            log_dir='./logs',
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard_cb)
    
    return callbacks

def setup_paths():
    if not os.path.isdir('models/ckpts'):
        if not os.path.isdir('models'):
            os.mkdir('models')
        os.mkdir('models/ckpts')

def main():
    LOG_FORMAT = '%(levelname)s %(message)s'
    logging.basicConfig(
        format=LOG_FORMAT, 
        level='INFO'
    )

    try:
        train()
    except KeyboardInterrupt:
        print('EXIT')

if __name__ == '__main__':
    main()
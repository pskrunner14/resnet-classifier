import os
import dill
import click
import logging

import h5py
import keras
import numpy as np
import keras_applications

from tqdm import tqdm

resolution = 128

@click.command()
@click.option(
    '-ds', 
    '--dataset-path', 
    default='datasets/caltech_101', 
    type=click.Path(exists=True), 
    help='Path for your Image Dataset'
)
def main(dataset_path):
    LOG_FORMAT = '%(levelname)s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level='INFO')
    augment_images(dataset_path)
    move_images(dataset_path)
    # sets = ['train', 'val', 'test']
    # for set_name in sets:
    #     process_image_dataset(dataset_path='{}/{}'
    #         .format(dataset_path, set_name), set_name=set_name)
    logging.info('Done preprocessing!')

def augment_images(dataset_path):
    categories = os.listdir('{}/train'.format(dataset_path))

    datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    for category in tqdm(categories, total=len(categories), desc='Augmenting Images'):
        if len(os.listdir('{}/train/{}'.format(dataset_path, category))) < 300:
            for image_path in os.listdir('{}/train/{}'.format(dataset_path, category)):
                img = keras.preprocessing.image.load_img('{}/train/{}/{}'
                        .format(dataset_path, category, image_path))
                x = keras.preprocessing.image.img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
                x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

                # the .flow() command below generates batches of randomly transformed images
                # and saves the results to the `preview/` directory
                i = 0
                for _ in datagen.flow(x, batch_size=1, save_to_dir='{}/train/{}'
                                    .format(dataset_path, category), save_prefix=image_path[:-4], 
                                    save_format='jpg'):
                    i += 1
                    if i > 10:
                        break  # otherwise the generator would loop indefinitely
    logging.info('Done augmenting images!')

def move_images(dataset_path):
    categories = os.listdir('{}/train'.format(dataset_path))

    for category in tqdm(categories, total=len(categories), desc='Moving Images'):
        os.mkdir('{}/val/{}'.format(dataset_path, category))
        category_images = os.listdir('{}/train/{}'.format(dataset_path, category))
        random_val_images = np.random.choice(category_images, 20, replace=False)
        for image_path in random_val_images:
            os.rename('{}/train/{}/{}'.format(dataset_path, category, image_path),
                    '{}/val/{}/{}'.format(dataset_path, category, image_path))

        os.mkdir('{}/test/{}'.format(dataset_path, category))
        category_images = os.listdir('{}/train/{}'.format(dataset_path, category))
        random_test_images = np.random.choice(category_images, 20, replace=False)
        for image_path in random_test_images:
            os.rename('{}/train/{}/{}'.format(dataset_path, category, image_path),
                    '{}/test/{}/{}'.format(dataset_path, category, image_path))
    logging.info('Done moving images!')

def preprocess_image(image):
    x = keras.preprocessing.image.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    return keras_applications.imagenet_utils.preprocess_input(x)[0]

def process_image_dataset(dataset_path=None, set_name='train'):
    if dataset_path is None:
        raise UserWarning('Dataset path should not be None!')

    X, y = [], []
    categories = os.listdir(dataset_path)

    # read images from dataset dir
    for category in tqdm(categories, total=len(categories), desc='Processing {} images'.format(set_name)):
        image_label = categories.index(category)
        for image_path in os.listdir('{}/{}'.format(dataset_path, category)):
            image = keras.preprocessing.image.load_img('{}/{}/{}'
                .format(dataset_path, category, image_path), 
                target_size=(resolution, resolution))
            image_pr = preprocess_image(image)
            X.append(image_pr)
            y.append(image_label)

    # convert to desired format
    X = np.array(X)
    y = keras.utils.np_utils.to_categorical(y, num_classes=len(categories))
    
    logging.info('{} features shape: {}'.format(set_name, X.shape))
    logging.info('{} targets shape: {}'.format(set_name, y.shape))

    #write final
    logging.info('Writing preprocessed {} data to files'.format(set_name))
    train_file = h5py.File('datasets/{}_data.h5'.format(set_name), "w")
    train_file.create_dataset('X', data=X)
    train_file.create_dataset('y', data=y)

    if not os.path.exists('datasets/classes.dill'):
        logging.info('Writing classes to file')
        with open('datasets/classes.dill', 'wb') as file:
            dill.dump(categories, file)

if __name__ == '__main__':
    main()
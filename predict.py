import os

import click
import keras
import numpy as np

@click.command()
@click.argument('image_path', type=click.Path(exists=True))
def main(image_path):
    resnet50 = keras.models.load_model('models/resnet50.h5')
    make_prediction(resnet50, image_path)

def make_prediction(model, path=None):
    if path is None:
        raise UserWarning('Image path should not be None!')

    categories = os.listdir('datasets/caltech_101/train')

    # preprocessing
    img = keras.preprocessing.image.load_img(path, target_size=(64, 64))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.utils.normalize(x)
    
    # make prediction
    preds = model.predict(x)
    print("Model predicts a \"{}\" with {:.2f}% probability".format(categories[np.argmax(preds[0])], preds[0][np.argmax(preds)] * 100))

if __name__ == '__main__':
    main()
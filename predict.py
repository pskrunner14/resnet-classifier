import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import pydot
from keras.utils.vis_utils import model_to_dot
import scipy.misc
from resnet_utils import convert_to_one_hot

resnet50 = load_model('datasets/resnet50.h5')

img_path = 'datasets/images/my_image.jpg'
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)
print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] : ")
preds = resnet50.predict(x)
print()
print(preds)
print()
print("Model predicts a \"" + str(np.argmax(preds[0])) + "\" with prob = " + str(preds[0][np.argmax(preds)]))
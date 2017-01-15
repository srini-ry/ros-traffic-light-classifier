import os

from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array


def reshape_image(image):
  x = img_to_array(image.resize((64, 64), Image.ANTIALIAS))
  return x[None, :]

model = load_model('light_classifier_model.h5')

# Load Example Image
green = load_img('examples/green.jpg')
red = load_img('examples/red.jpg')

print 'Green' + model.predict(reshape_image(green)).__str__()
print 'Red' + model.predict(reshape_image(red)).__str__()
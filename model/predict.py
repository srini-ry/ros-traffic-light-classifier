import os

from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array


def reshape_image(image):
  x = img_to_array(image.resize((64, 64), Image.ANTIALIAS))
  return x[None, :]

model = load_model('light_model.h5')

# Load Example Image
green = load_img('green.jpg')
red = load_img('red.jpg')

print 'Green' + model.predict(reshape_image(green)).__str__()
print 'Red' + model.predict(reshape_image(red)).__str__()
# print 'ROS Unknown' + model.predict(reshape_image(load_img('ros-frames/493.jpg'))).__str__()
# print 'ROS Red' + model.predict(reshape_image(load_img('ros-frames/535.jpg'))).__str__()

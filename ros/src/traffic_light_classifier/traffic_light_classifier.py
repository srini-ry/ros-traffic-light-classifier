import numpy
import sys
import PIL
import message_filters
import rospy

from PIL import Image
from cv_bridge import CvBridge
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from road_wizard.msg import Signals
from runtime_manager.msg import traffic_light
from sensor_msgs.msg import Image

RED = 0
GREEN = 1
UNKNOWN = 2
RADIUS_MULTIPLIER = 6

model = None


def calculate_bounds(signal):
  xmin = sys.maxint
  xmax = -sys.maxint - 1
  ymin = sys.maxint
  ymax = -sys.maxint - 1
  radius_max = 0

  for signal in signal.Signals:
    x = signal.u
    y = signal.v
    radius_max = max(radius_max, signal.radius)
    xmin = min(xmin, x)
    xmax = max(xmax, x)
    ymin = min(ymin, y)
    ymax = max(ymax, y)

  return int(xmin - RADIUS_MULTIPLIER * radius_max), int(xmax + RADIUS_MULTIPLIER * radius_max), int(
    ymin - RADIUS_MULTIPLIER * radius_max), int(ymax + RADIUS_MULTIPLIER * radius_max)


def crop_image(image, xmin, xmax, ymin, ymax):
  return image.crop((xmin, ymin, xmax, ymax))


def predict_light(cropped_roi):
  # Load CNN Model
  loaded_model = get_model()
  image_array = img_to_array(cropped_roi.resize((64, 64), PIL.Image.ANTIALIAS))
  prediction = loaded_model.predict(image_array[None, :])
  if prediction[0][0] == 1:
    return GREEN
  elif prediction[0][1] == 1:
    return RED
  else:
    return UNKNOWN


def get_model():
  # TODO Fix: Must load model from ROS callback thread
  global model
  if not model:
    model = load_model('light_classifier_model.h5')
  return model


def detect_signal(signal, image):
  if len(signal.Signals) == 0:
    # No signals are visible
    light_detected_publisher.publish(traffic_light(traffic_light=UNKNOWN))
    return

  # Convert the image to PIL
  cv_bridge = CvBridge()
  cv_image = cv_bridge.imgmsg_to_cv2(image, "rgb8")
  image = PIL.Image.fromarray(cv_image)

  # Find the bounds of the signal
  xmin, xmax, ymin, ymax = calculate_bounds(signal)

  # Crop the image for the ROI
  cropped_roi = crop_image(image, xmin, xmax, ymin, ymax)

  roi_image.publish(cv_bridge.cv2_to_imgmsg(numpy.array(cropped_roi), "rgb8"))
  # Run the cropped image through the NN
  prediction = predict_light(cropped_roi)

  # Publish the prediction
  light_detected_publisher.publish(traffic_light(traffic_light=prediction))


rospy.init_node('traffic_light_classifier')

roi_signal = message_filters.Subscriber('roi_signal', Signals)
camera_image = message_filters.Subscriber('image_raw', Image)
light_detected_publisher = rospy.Publisher('light_color', traffic_light, queue_size=1)
roi_image = rospy.Publisher('roi_image', Image, queue_size=1)

time_sync = message_filters.ApproximateTimeSynchronizer([roi_signal, camera_image], 5, .1)
time_sync.registerCallback(detect_signal)
rospy.spin()

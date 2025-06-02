import time
from deepface import DeepFace

img = "images/multi_image_1.jpeg"

# CPU run (force CPU context)
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # temporarily hide GPU
start = time.time()
DeepFace.represent(img_path=img, model_name="ArcFace", enforce_detection=True, detector_backend="opencv")
print("CPU time:", time.time() - start)

# GPU run
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus, 'GPU')  # restore GPU
start = time.time()
DeepFace.represent(img_path=img, model_name="ArcFace", enforce_detection=True, detector_backend="opencv")
print("GPU time:", time.time() - start)

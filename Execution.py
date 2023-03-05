from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
image_path_1='/content/drive/MyDrive/real_and_fake_face_detection/Screenshot (2).png' #Image input
testImage = img.imread(image_path_1)
plt.imshow(testImage)
model = load_model('/content/drive/MyDrive/saved_models/model.h5')
image = load_img(image_path_1, target_size=(224, 224))
img = np.array(image)
img = img / 255.0
img = img.reshape(1,224,224,3)
label = model.predict(img)
def Predicted_value():
  if label[0][0] <=0.5:
    return 0
  else:
    return 1
print("Predicted Class (0 - Real , 1- Fake): ", Predicted_value())

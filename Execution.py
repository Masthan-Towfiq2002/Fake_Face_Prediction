from google.colab import files
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
uploaded = files.upload()
if len(uploaded) > 0:
    image_path = list(uploaded.keys())[0]
    testImage = mpimg.imread(image_path)
    plt.imshow(testImage)
    model = load_model('/content/drive/MyDrive/saved_models/model.h5')
    image = load_img(image_path, target_size=(224, 224))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)
    label = model.predict(img)

    def Predicted_value():
        if label[0][0] <= 0.5:
            return 0
        else:
            return 1

    print("Predicted Class (0 - Real , 1- Fake): ", Predicted_value())
else:
    print("No image uploaded.")

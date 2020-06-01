from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

print('Loading model..')
model = load_model('model_trained_3class.hdf5', compile=False)
print('Model loaded!')

def predict_class(img, model=model):
    food_list = ['burger','pizza', 'sushi']
    img = image.load_img(img, target_size=(299, 299))
    img = image.img_to_array(img)     
    img = np.expand_dims(img, axis=0)         
    img /= 255.                 
    pred = model.predict(img)
    index = np.argmax(pred)
    food_list.sort()
    pred_value = food_list[index]
    return pred_value

# print(predict_class(model, 'images/download.jpg'))
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
def load_model():
  model=tf.keras.models.load_model('my_model12_savedmodel.keras')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Soil Moisture Predictors
         """
         )

image_height = 50
image_width = 50

#define a function that accepts an image url and outputs an eager tensor
def path_to_eagertensor(image_data):
        image = tf.convert_to_tensor(image_data)        
       # image = tf.image.decode_png(raw, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, (image_height, image_width))
        return image

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])

#st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        Sample_image_prediction = []
        #Preprocess with our function
        sample_new_img_tensor = path_to_eagertensor(image_data)
        Sample_image_prediction.append(sample_new_img_tensor)
        Sample_image_prediction = np.array(Sample_image_prediction)
        #Show data type is good to input into model
        print(type(Sample_image_prediction),Sample_image_prediction.shape)
        Sample_image_prediction=tf.convert_to_tensor(Sample_image_prediction)
        prediction = model.predict(Sample_image_prediction)
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    predictions = import_and_predict(image, model)
    st.write(
    "This soil has a moisture content of",predictions[0][0],"%")
    st.image(image, use_column_width=True)

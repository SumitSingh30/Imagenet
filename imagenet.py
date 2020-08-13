import streamlit as st
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from  tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from PIL import Image 

st.write("""
# Imagenet App
## Upload Image""")

file=st.file_uploader("", type=["jpeg","jpg","png"])
st.set_option('deprecation.showfileUploaderEncoding', False)
if file is not None:
	st.image(file,use_column_width=True)

#model_path='vgg19.h5'
#model=load_model(model_path)
#model._make_predict_function()
#graph=tf.get_default_graph()
model=VGG19(weights='imagenet')



def model_predict(file,model):
	#img = image.load_img(file, target_size=(224,224)
	img = Image.open(file)
	img = img.resize((224,224))
	x = np.array(img)
	#x = image.img_to_array(img)

	x = np.expand_dims(x, axis=0)

	x = preprocess_input(x)

	#with graph.as_default():
	preds = model.predict(x)
	return preds

if file is not None:
	#model=VGG19(weights='imagenet')
	preds = model_predict(file, model)
	pred_class = decode_predictions(preds, top=1)
	result = str(pred_class[0][0][1])

	st.write(result)

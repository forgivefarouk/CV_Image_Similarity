import streamlit as st
from PIL import Image
from keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow as tf
import numpy as np
import os
import cv2
import pandas as pd
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import NearestNeighbors


model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model= tf.keras.Sequential([
  model,
  tf.keras.layers.GlobalAveragePooling2D(),  
])

def feature_extractor(image_path,model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    results = model.predict(img).flatten()
    normalized_results = results / np.linalg.norm(results)
    return normalized_results

path = pathlib.Path('dataset')


features = np.array(pickle.load(open('feature_list.pkl','rb')))
file_name = pickle.load(open('file_name.pkl','rb'))

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploaded',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except:
        return False

def recommendations(features,top_n,image,model):
    nbrs = NearestNeighbors(n_neighbors=top_n, algorithm='brute', metric='euclidean')
    nbrs.fit(features)
    normalized_results=feature_extractor(image,model)
    dis , idx = nbrs.kneighbors([normalized_results])
    return idx

def upload_image():
    uploaded_image = st.file_uploader("Choose an image")
    if uploaded_image is not None:
        if save_uploaded_file(uploaded_image):
        # Convert the uploaded image to PIL Image
            image = Image.open(uploaded_image)
            image = image.resize((224, 224))
            # Display the uploaded image
            st.image(image)
            up_img=os.path.join('uploaded',uploaded_image.name)
            idx = recommendations(features,6,up_img,model)
           
            st.subheader("Similar Products")
            col1,col2,col3,col4,col5 = st.columns(5)
            
            with col1:
                st.image(file_name[idx[0][1]].replace('\\','/'))
            with col2:
                st.image(file_name[idx[0][2]].replace('\\','/'))
            with col3:
                st.image(file_name[idx[0][3]].replace('\\','/'))
            with col4:
                st.image(file_name[idx[0][4]].replace('\\','/'))
            with col5:
                st.image(file_name[idx[0][5]].replace('\\','/'))

def main():
    st.title("Men & Women Fashion Recommendation")
    upload_image()

if __name__ == "__main__":
    main()

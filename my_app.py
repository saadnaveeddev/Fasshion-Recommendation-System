import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import joblib
import numpy as np
from PIL import Image

# Load features and image file list
features_list = joblib.load("image_features_embedding.pkl")
img_files_list = joblib.load("img_files.pkl")

# Load ResNet model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

# Streamlit app
def main():
    st.title("Image Recommender App")
    
    st.write("Upload an image and we'll recommend 5 related images!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)

        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(features_list)

        distances, indices = neighbors.kneighbors([normalized_result])

        st.subheader("Uploaded Image")
        st.image(img, caption='Uploaded Image', use_column_width=True)

        st.subheader("Recommended Images")
        
        # Create a horizontal layout container for recommended images
        col1, col2, col3 = st.columns(3)
        with col1:
            recommended_img = Image.open(img_files_list[indices[0][1]])
            recommended_img = recommended_img.resize((224, 224))
            st.image(recommended_img, width=224)
        
        with col2:
            recommended_img = Image.open(img_files_list[indices[0][2]])
            recommended_img = recommended_img.resize((224, 224))
            st.image(recommended_img, width=224)
        
        with col3:
            recommended_img = Image.open(img_files_list[indices[0][3]])
            recommended_img = recommended_img.resize((224, 224))
            st.image(recommended_img, width=224)

if __name__ == "__main__":
    main()

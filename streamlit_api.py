import numpy as np
import cv2
import streamlit as st
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
import gdown  # To download the model from Google Drive

# Defining the number of classes
num_classes = 5

# Defining the class labels
class_labels = {
    0: 'Brown Spot',
    1: 'Healthy',
    2: 'Hispa',
    3: 'Leaf Blast',
    4: 'Tungro'
}

def add_bg_from_local():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://img.freepik.com/free-photo/detail-rice-plant-sunset-valencia-with-plantation-out-focus-rice-grains-plant-seed_181624-25838.jpg?t=st=1718132095~exp=1718135695~hmac=922d823bfa5e3d46447fb607c958599bace78715a77fd8eec0374b1e2333e23f&w=826");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local()

#Loading model file from Drive

# Function to download the model from Google Drive
@st.cache(allow_output_mutation=True)
def load_cached_model(file_id, output_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    model = load_model(output_path)
    return model

file_id="1lHCCMfbvEPjjxFC4JqlwYowrSX-6PfUt"
model_path="weight.h5"
# Loading the trained model
model = load_cached_model(file_id,model_path)  # Make sure you have the correct path to your saved model

# Function to preprocess the input image
def preprocess_image(img):
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension as the model expects it
    return img

# Streamlit app
def main():
    st.title("Paddy Leaf Disease Prediction App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Resize and display the uploaded image
        image_display = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        image_display = cv2.resize(image_display, (300, 300))  # Resize the image
        st.image(image_display, caption="Uploaded Image")

        # Preprocess the uploaded image
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img = preprocess_image(img)

        # Extract features using the VGG16 base model
        base_model = VGG16(weights='imagenet', include_top=False)
        img_features = base_model.predict(img)

        # Reshape the features to match the input shape of the student model
        img_features = img_features.reshape(1, 7, 7, 512)

        # Make predictions using the student model
        predictions = model.predict(img_features)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions[0])

        # Map the index back to class label
        predicted_class_label = class_labels[predicted_class_index]

        # Display the prediction
        st.write(f"<span style='font-size:30px; color:red;'>Predicted Class Label: {predicted_class_label}</span>", unsafe_allow_html=True)
        
if __name__ == "__main__":
    main()

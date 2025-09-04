import streamlit as st
from langchain_groq import ChatGroq
from predict import PlantDiseaseClassifier
import os
import json


st.title("Remedy Recommendation for 🌿 Plant Disease")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save uploaded image to a temp file
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Initialize model (adjust paths and API key as needed)
    model_path = "plantdoc_resnet50.pth"
    groq_api_key = st.secrets.get("GROQ_API_KEY", "")  # store key securely in Streamlit secrets

    # You must define your class list here exactly as in training
    #classes = ['Apple_leaf', 'Apple_rust_leaf', 'Apple_Scab_Leaf', 'Bell_pepper_leaf', 'Bell_pepper_leaf_spot', 'Blueberry_leaf', 'Cherry_leaf', 'Corn_Gray_leaf_spot', 'Corn_leaf_blight', 'Corn_rust_leaf', 'grape_leaf', 'grape_leaf_black_rot', 'Peach_leaf', 'Potato_leaf_early_blight', 'Potato_leaf_late_blight', 'Raspberry_leaf', 'Soyabeen_leaf', 'Squash_Powdery_mildew_leaf', 'Strawberry_leaf', 'Tomato_Early_blight_leaf', 'Tomato_leaf', 'Tomato_leaf_bacterial_spot', 'Tomato_leaf_late_blight', 'Tomato_leaf_mosaic_virus', 'Tomato_leaf_yellow_virus', 'Tomato_mold_leaf', 'Tomato_Septoria_leaf_spot','Tomato_two_spotted_spider_mites_leaf']



    with open("class_names.json", "r") as f:
        classes = json.load(f)


    classifier = PlantDiseaseClassifier(model_path, classes, groq_api_key)

    with st.spinner("Predicting disease and fetching remedy..."):
        disease = classifier.predict(temp_path)
        remedy = classifier.get_remedy(disease)

    st.markdown(f"### Detected Disease: **{disease}**")
    st.markdown("### Remedy Recommendation:")
    st.write(remedy)

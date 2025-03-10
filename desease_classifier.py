import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


st.set_page_config(
    page_title="Ahmed's Smart AgroVision",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        body { margin-top: -10px; background-color: #FFFFFF; }
        .main-title { font-size: 3em; font-weight: bold; color: #004C75; text-align: center; margin-bottom: 20px; }
        .section-title { font-size: 2em; color: #006994; font-weight: bold; margin-top: 30px; text-align: left; background-color: #E0F0FF; padding: 10px; border-radius: 8px; }
        .stTab { font-size: 1.4em; font-weight: bold; color: #004C75; }
        .section-content { text-align: center; }
        .intro-title { font-size: 2.8em; color: #005F7F; font-weight: bold; text-align: center; background-color: #E0F0FF; padding: 10px; border-radius: 8px; }
        .intro-subtitle { font-size: 1.4em; color: #0077A7; text-align: center; background-color: #E0F0FF; padding: 10px; border-radius: 8px; }
        .content { font-size: 1.2em; color: #003B5C; text-align: justify; line-height: 1.6; background-color: #E0F0FF; padding: 10px; border-radius: 8px; }
        .highlight { font-weight: bold; color: #002B4E; }
        .separator { height: 3px; background-color: #005F7F; margin-top: 20px; margin-bottom: 20px; }
        .prediction-text { font-size: 2.2em; font-weight: bold; color: #004C75; text-align: center; text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3); background-color: #E0F0FF; padding: 10px; border-radius: 8px; }
        .footer { font-size: 14px; color: #002B4E; margin-top: 20px; text-align: center; }
    </style>
""", unsafe_allow_html=True)




st.markdown('''<div class="main-title">🌱 Welcome to the Ahmed's Smart AgroVision🌱</div>''', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "🥔 Potato Disease Analysis", "🍅 Tomato Disease Analysis", "🌶️ Pepper Disease Analysis"])
with tab1:
    st.markdown('<div class="section-title">👋 About Ahmed</div> <br>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="content">
            Hello! I’m <span class="highlight">Ahmed Hasnain</span>, a Data Scientist passionate about 
            <span class="highlight">AI & Machine Learning</span>. I specialize in building intelligent solutions for real-world problems.
            Explore my work at <a href="https://github.com/Ahmedmoria" target="_blank" style="color: #E65100; font-weight: bold;">GitHub</a>.
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎯 Project Overview </div> <br>', unsafe_allow_html=True)

    st.markdown("""
        <div class="content">
            Ahmed Hasnain specializes in AI-driven plant disease detection. Key projects include:
            <ul>
                <li><span class="highlight">🥔 Potato Disease Detection:</span> Used CNNs 🧠 to identify potato diseases from images. Trained on a 
                    <a href="https://www.kaggle.com/datasets/arjuntejaswi/plant-village" target="_blank" style="color: silver; font-weight: bold;">dataset</a> 
                    and deployed for real-time predictions. 📡
                </li>
                <li><span class="highlight">🍅 Tomato Disease Detection:</span> Leveraged Transfer Learning 🧠 for identifying tomato diseases. 
                    Trained on a <a href="https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf" target="_blank" style="color: silver; font-weight: bold;">dataset</a>.
                </li>
                <li><span class="highlight">🌶️ Pepper Disease Detection:</span> Applied deep learning 🧠 to classify pepper plant diseases using 
                    <a href="https://www.kaggle.com/datasets/arjuntejaswi/plant-village" target="_blank" style="color: silver; font-weight: bold;">dataset</a>.
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🚀 Future Work</div> <br>', unsafe_allow_html=True)

    st.markdown("""
        <div class="content">
            Plans include:
            <ul>
                <li><span class="highlight">🌾 Multi-Crop Detection:</span> Expanding models to include wheat, corn, and other crops.</li>
                <li><span class="highlight">📱 Mobile App:</span> Developing a user-friendly app for real-time field diagnosis.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">💻 Technologies & Tools</div> <br>', unsafe_allow_html=True)

    st.markdown("""
        <div class="content">
            <ul>
                <li><span class="highlight">🔤 Languages & Libraries:</span> Python, NumPy, Pandas, TensorFlow, Keras, Scikit-Learn.</li>
                <li><span class="highlight">⚙️ Methods:</span> CNNs, Data Augmentation, Transfer Learning.</li>
                <li><span class="highlight">🌐 Deployment:</span> Streamlit for an interactive web interface.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-title">🥔 Potato Plant Disease Classification 🔬</div><br>', unsafe_allow_html=True)
    st.markdown('''
    <div class="content" style="background-color: #E3F2FD; padding: 15px; border-radius: 10px;">
        Upload a high-quality image of a potato plant leaf 🥔, and the model will analyze its health status:
        <ul>
            <li>✅ <b>Healthy</b> 🌿 - No disease detected.</li>
            <li>⚠️ <b>Early Blight</b> 🍂 - Initial stage of infection.</li>
            <li>❌ <b>Late Blight</b> 🌪️ - Severe infection detected.</li>
        </ul>
    </div><br/>
    ''', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2]) 

    with col1:
        uploaded_file = st.file_uploader("📸 Upload a leaf image (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", width=250)

    with col2:
        if uploaded_file is not None:
            image = image.resize((256, 256))  
            image_batch = np.expand_dims(image, axis=0)

            potato_classifier_model = tf.keras.models.load_model('./model/potato_desease_classifier_v3.h5')
            potato_target_labels = ["Early Blight", "Late Blight", "Healthy"]
            
            predictions = potato_classifier_model.predict(image_batch)
            predicted_class = potato_target_labels[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])
            
            if predicted_class.lower() == "healthy":
                status_message = f"🟢 <b>Your potato plant is</b> <span style='color: #4CAF50;'>{predicted_class}</span> 🌱."
            else:
                status_message = f"🔴 <b>Disease Detected:</b> <span style='color: #c40000;'>{predicted_class}</span> ⚠️."

            st.markdown(f'''
            <div style="background-color: #9eb0b1; padding: 15px; border-radius: 10px; text-align: center;">
                <div style="font-size: 2em; font-weight: bold;">
                    {status_message}
                </div>
                <div style="color: #01579B; font-size: 1.8em; font-weight: bold;">
                    Confidence: {confidence:.2f} 🔍
                </div>
            </div>
            ''', unsafe_allow_html=True)


with tab3:
    st.markdown('<div class="section-title">🍅 Tomato Plant Disease Classification 🔬</div> <br>', unsafe_allow_html=True)
    st.markdown('''
    <div class="content">
        📸 Upload a clear image of a tomato plant leaf 🍅, and the AI model will analyze its health status and detect any potential diseases from the following:
        <ul>
            <li>✅ Tomato Healthy 🌿.</li>
            <li>🦠 Tomato Mosaic Virus 🧬.</li>
            <li>🎯 Target Spot 🍂.</li>
            <li>🦠 Bacterial Spot 🛑.</li>
            <li>🍃 Tomato Yellow Leaf Curl Virus ⚠️.</li>
            <li>🌨️ Late Blight 🥶.</li>
            <li>🍁 Leaf Mold 🦠.</li>
            <li>🌞 Early Blight 🔥.</li>
            <li>🕷️ Spider Mites (Two-spotted Spider Mite) 🕸️.</li>
            <li>⚫ Septoria Leaf Spot 🍃.</li>
        </ul>
    </div><br/>
    ''', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_file = st.file_uploader("🖼️ Upload a leaf image for analysis:", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", width=280)

    with col2:
        if uploaded_file is not None:
            img = load_img(uploaded_file, target_size=(256, 256))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            tomato_classifier_model = tf.keras.models.load_model('./model/tomato_desease_classifier_v1.h5')
            predictions = tomato_classifier_model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions[0])
            
            class_names = ['Bacterial Spot 🛑', 'Early Blight 🔥', 'Late Blight 🥶', 'Leaf Mold 🦠', 'Septoria Leaf Spot 🍃',
                           'Spider Mites 🕸️', 'Target Spot 🎯', 'Yellow Leaf Curl Virus ⚠️', 'Mosaic Virus 🧬', 'Healthy ✅']
            
            predicted_class_name = class_names[predicted_class_index]
            
            if 'Healthy' in predicted_class_name:
                status_message = f"🌿 Your Tomato Plant is <span style='color: #2E8B57;'>{predicted_class_name}</span> ✅."
            else:
                status_message = f"🚨 Disease Detected: <span style='color: #D32F2F;'>{predicted_class_name}</span> 🦠."
            
            st.markdown(f'''
            <div style="border-radius: 12px; background-color: #f8f9fa; padding: 20px; margin-top: 20px;">
                <div style="font-size: 1.8em; font-weight: bold; text-align: center; color: #333;">
                    {status_message}
                </div>
                <div style="color: #555; font-size: 1.5em; font-weight: bold; text-align: center; margin-top: 10px;">
                    🔍 Confidence: <span style='color: #1976D2;'>{confidence:.2f}</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
with tab4:   
    st.markdown('<div class="section-title" style="color: #003366; font-size: 2.5em; font-weight: bold; text-align: center;">🌶️ Pepper Plant Disease Classification 🦠</div><br>', unsafe_allow_html=True)
    st.markdown('''
    <div class="content" style="background-color: #E3F2FD; padding: 15px; border-radius: 10px; text-align: center;">
        Upload a clear image of a pepper plant leaf 🌶️, and the model will identify its health status or diagnose any potential disease from the following:
        <ul style="text-align: left; font-size: 1.2em;">
            <li><b style="color: #2E7D32;">Pepper Bell Healthy 🌱</b>.</li>
            <li><b style="color: #D32F2F;">Pepper Bell Bacterial Spot 🦠</b>.</li>
        </ul>
    </div><br/>
    ''', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_file = st.file_uploader("⬆️ Upload a leaf image:", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded Image", width=250)

    with col2:
        if uploaded_file is not None:
            img = load_img(uploaded_file, target_size=(256, 256))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0) 
            pepper_classifier_model = tf.keras.models.load_model('./model/pepper_desease_classifier_v1.h5')
            predictions = pepper_classifier_model.predict(img_array)
            confidence = predictions[0][0]
            threshold = 0.5

            if confidence <= threshold:
                predicted_class_name = "Bacterial Spot"
                status_message = f"<span style='color: #D32F2F; font-weight: bold;'>🚨 Disease Detected: {predicted_class_name} 🦠</span>"
                confidence = 1 - confidence
            else:
                predicted_class_name = "Healthy"
                status_message = f"<span style='color: #2E7D32; font-weight: bold;'>✅ Your 🌶️ plant is {predicted_class_name} 🌱</span>"
            
            st.markdown(f'''
            <div style="background-color: #F1F8E9; padding: 15px; border-radius: 10px; text-align: center;">
                <div style="font-size: 2em; font-weight: bold;">
                    {status_message}
                </div>
                <div style="color: #616161; font-size: 1.5em; font-weight: bold; margin-top: 10px;">
                    🔍 Confidence: {confidence:.2f}
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown('<div class="separator" style="height: 3px; background-color: #003366; margin-top: 20px; margin-bottom: 20px;"></div>', unsafe_allow_html=True)


 

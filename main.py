import streamlit as st
import tensorflow as tf
import numpy as np
import wikipedia
import cv2
import requests

def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
def crop_leaf(image):
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green color mask (leaf detection)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        cropped = image[y:y+h, x:x+w]
        return cropped

    return image  

# Telegram notification
def telegram(message):
    bot_token = "8798707989:AAEsuYQ-_BDgMlxAAp-pVqkCnH8LNj0smLQ"  
    chat_id = "825932845"

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    data = {
        "chat_id": chat_id,
        "text": message
    }

    try:
        requests.post(url, data=data)
    except:
        pass


# Groq API via HTTP 
def ask_ai(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization":"Bearer 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",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": messages
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"]
    except:
        return "Error fetching response from AI."


# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_model.keras", compile=False)

model = load_model()
print(model.summary())


# Classes
class_name = [
'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus',
'Tomato___healthy'
]

# Prediction
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])

    predictions = model.predict(input_arr)
    return np.argmax(predictions), np.max(predictions)


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home","About","Disease Recognition"])


# Home 
if(app_mode=="Home"):
    st.header(" PLANT DISEASE RECOGNITION SYSTEM")
    st.image("pexels-minan1398-793012.jpeg", use_column_width=True)
    st.markdown("Upload a leaf image to detect diseases using AI.")
    st.info(" Tip: Use a clear leaf image for better accuracy.")


# About
elif(app_mode=="About"):
    st.header("About")
    st.markdown("Dataset contains ~87K images across 38 classes.")


# Disease Recognition
elif app_mode == "Disease Recognition":

    st.header("Disease Recognition")

    option = st.radio("Input Method", ["Upload Image", "Use Camera"])

    if option == "Use Camera":
     test_image = st.camera_input("Take a photo of the leaf")
    else:
     test_image = st.file_uploader("Upload Plant Leaf Image")

    if st.button("Show Image"):
        st.image(test_image, use_column_width=True)

    

    if st.button("Predict"):

        if test_image is None:
            st.warning("Please upload an image first.")
        else:
            test_image.seek(0)

            result_index, confidence = model_prediction(test_image)

            if confidence < 0.6:
                st.warning(f"Low confidence ({confidence*100:.2f}%)")

            predicted_class = class_name[result_index]

           # Read image first
            test_image.seek(0)
            file_bytes = np.asarray(bytearray(test_image.read()), dtype=np.uint8)
            original = cv2.imdecode(file_bytes, 1)

             # Crop leaf FIRST
            cropped = crop_leaf(original)

             # Resize cropped image
            resized = cv2.resize(cropped, (128,128))

             # Prepare for model
            img_array = resized / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Generate heatmap
            heatmap = get_gradcam_heatmap(model, img_array, "conv2d_15")
            

            # Convert heatmap
            heatmap = cv2.resize(heatmap, (128,128))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Overlay
            superimposed_img = cv2.addWeighted(resized, 0.6, heatmap, 0.4, 0)

            # Show
            st.image(superimposed_img, caption="Model Focus (Heatmap)", use_column_width=True)

            
            plant, disease = predicted_class.split("___")
            disease_name = disease.replace("_"," ")

            # Store safely
            st.session_state.plant_name = plant
            st.session_state.disease_name = disease_name

            # Telegram
            if "healthy" not in disease_name.lower():
                telegram(
                    f" Disease Detected!\nPlant: {plant}\nDisease: {disease_name}\nConfidence: {confidence*100:.2f}%"
                )

            # Wikipedia + AI
            try:
                if "healthy" in disease_name.lower():
                    disease_info = "This plant appears healthy."
                else:
                    query = f"{disease_name} in {plant}"
                    results = wikipedia.search(query)

                    disease_info = None

                    for r in results:
                        if "list" in r.lower():
                            continue
                        try:
                            page = wikipedia.page(r, auto_suggest=False)
                            disease_info = page.summary[:500]
                            break
                        except:
                            continue

                    if disease_info is None:
                        disease_info = ask_ai([
                            {"role": "user", "content": f"Explain {disease_name} disease in {plant} in simple terms."}
                        ])

            except:
                disease_info = ask_ai([
                    {"role": "user", "content": f"Explain {disease_name} disease in {plant} in simple terms."}
                ])

            st.session_state.disease_info = disease_info

            # Reset chat
            st.session_state.chat_history = [
                {"role": "system", "content": f"You are an agricultural expert. Disease: {disease_name}"}
            ]


    # Display results safely
    if "disease_name" in st.session_state:

        st.success(f" Plant: {st.session_state.plant_name}")
        st.success(f" Disease: {st.session_state.disease_name}")

        st.write(st.session_state.disease_info)

        st.subheader("AI Chat")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "system", "content": f"You are an agricultural expert. Disease: {st.session_state.disease_name}"}
            ]

        for msg in st.session_state.chat_history[1:]:
            st.chat_message(msg["role"]).write(msg["content"])

        user_input = st.chat_input("Ask something...")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            response = ask_ai(st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
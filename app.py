import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Cargar modelo
with open("model_digits_1to9.json", "r") as json_file:
    model_json = json_file.read()
model = tf.keras.models.model_from_json(model_json)
model.load_weights("model_digits_1to9.weights.h5")

# Título de la app
st.title("Reconocimiento de Dígitos (1-9)")

st.write("Sube una imagen de un dígito del 1 al 9 escrita a mano (formato PNG o JPG).")

# Subir imagen
uploaded_file = st.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Mostrar imagen
    image = Image.open(uploaded_file).convert('L')  # Escala de grises
    st.image(image, caption='Imagen cargada', use_column_width=True)

    # Preprocesar imagen
    image = ImageOps.invert(image)  # Invertir blanco/negro si es necesario
    image = image.resize((28, 28))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predicción
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction) + 1  # Sumar 1 porque etiquetas eran 0-8

    st.subheader(f"🔢 Dígito detectado: **{predicted_digit}**")

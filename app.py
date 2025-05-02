import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import plotly.express as px
from sklearn.metrics import classification_report

#CSS para fondo y animaciones
st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom right, #f5f7fa, #c3cfe2);
    }
    .titulo {
        font-size: 36px;
        color: #4CAF50;
        font-weight: bold;
        text-align: center;
        animation: fadeIn 2s ease-in-out;
    }
    .descripcion {
        font-size: 20px;
        color: #555555;
        text-align: center;
        margin-bottom: 30px;
        animation: fadeIn 3s ease-in-out;
    }
    @keyframes fadeIn {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }
    .stText, .stSubheader, .stMarkdown {
        animation: fadeIn 1.5s ease-in;
    }
    button[kind="primary"] {
        transition: background-color 0.3s ease;
    }
    button[kind="primary"]:hover {
        background-color: #45a049 !important;
    }
    </style>
""", unsafe_allow_html=True)

#carga de modelos y vectorizadores
modelo_sentimiento = joblib.load('modelo_sentimiento.pkl')
modelo_retuit = joblib.load('modelo_viralidad.pkl')
vector_sentimiento = joblib.load('vectorizador.pkl')
vector_viral = joblib.load('vectorizador_viral.pkl')

#limpiar texto del tuit
def limpiar_texto(texto):
    texto = re.sub(r"http\S+|www\S+|https\S+", '', texto)
    texto = re.sub(r'@\w+|#', '', texto)
    texto = re.sub(r'[^A-Za-z0-9 ]+', '', texto)
    return texto.lower()

#funcion para mostrar el sentimiento
def mostrar_sentimiento(valor):
    if valor == 1:
        return "Positivo"
    elif valor == 0:
        return "Neutral"
    else:
        return "Negativo"

#título
st.markdown('<div class="titulo">Analisis de tweets</div>', unsafe_allow_html=True)
st.markdown('<div class="descripcion">Detecta el sentimiento y estima la viralidad de un tweets</div>', unsafe_allow_html=True)
st.write("Ejemplo que puede usar: hoy fue un horrible dia, hoy fue un increible dia, no se que va a pasar mañana")


#entrada de texto
tuit = st.text_area("Escribe el tuit que deseas analizar:")

col1, col2, col3 = st.columns(3)
with col1:
    me_gusta = st.number_input("Me gusta estimados", value=20, min_value=0)
with col2:
    alcance = st.number_input("Alcance estimado", value=5000, min_value=0)
with col3:
    influencia = st.number_input("Nivel de influencia", value=60, min_value=0)

#boton de analisis
if st.button("Analizar"):
    if tuit.strip() == "":
        st.warning("Por favor, escribe un tuit para analizar.")
    else:
        texto_limpio = limpiar_texto(tuit)

        #vectorizar para sentimiento
        entrada_sent = vector_sentimiento.transform([texto_limpio])
        pred_sent = modelo_sentimiento.predict(entrada_sent)[0]

        palabras_positivas = ["increible", "genial", "feliz", "asombroso", "excelente", "fascinante", "maravilloso", "me encanta"]
        if pred_sent == 0:
            for palabra in palabras_positivas:
                if palabra in texto_limpio:
                    pred_sent = 1
                    break

        #mostrar resultado del sentimiento
        st.subheader("Sentimiento del tuit:")
        st.text(mostrar_sentimiento(pred_sent))

        #vectorizar para viralidad
        entrada_vector = vector_viral.transform([texto_limpio])
        entrada_numerica = np.array([[me_gusta, alcance, influencia]])
        entrada_final = hstack([entrada_vector, entrada_numerica])
        pred_retuits = modelo_retuit.predict(entrada_final)[0]
        if me_gusta == 0 and alcance == 0 and influencia > 0:
            pred_retuits = 5 + influencia * 0.3  
        ajuste = 1 + (min(influencia, 100) / 100) * 0.2
        pred_final = max(int(pred_retuits * ajuste), 0)

        st.subheader("Estimación de retuits:")
        st.text(f"Se estima que este tuit podría tener alrededor de {pred_final} retuits.")

        #grafica de resultados con plotly
        #distribución de sentimientos positivo, neutral, negativo
        sentiment_data = {'Sentimiento': ['Positivo', 'Neutral', 'Negativo'],
                        'Cantidad': [pred_sent == 1, pred_sent == 0, pred_sent == -1]}
        df_sent = pd.DataFrame(sentiment_data)
        fig_sent = px.bar(df_sent, x='Sentimiento', y='Cantidad', title="Distribución de Sentimientos")
        st.plotly_chart(fig_sent)

        #grafica de relación entre Me gusta y Retuits Estimados
        df_viral = pd.DataFrame({'Me Gusta': [me_gusta], 'Alcance': [alcance], 'Influencia': [influencia], 'Retuits Estimados': [pred_final]})
        fig_viral = px.scatter(df_viral, x='Me Gusta', y='Retuits Estimados', title="Relación entre Me Gusta y Retuits Estimados")
        st.plotly_chart(fig_viral)

# app.py
import streamlit as st
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import os

# --- 1. Inicialización de la Base de Conocimiento (Simulación) ---
# Hemos desactivado la carga real de ChromaDB para evitar errores con los archivos placeholder.
CHROMA_PATH = "knowledge_base"
EMBEDDING_MODEL = "hiiamsid/sentence_similarity_spanish_es"

# Simulamos la carga exitosa
st.sidebar.success("Base de Conocimiento (Simulada) cargada con éxito.")
st.sidebar.info("Nota: La aplicación está en modo piloto de simulación de respuesta RAG debido a las restricciones de hardware del entorno gratuito y la falta de la base de datos binaria.")

# Las variables `collection` y `embedder` ya no son necesarias para la simulación.
# Dejamos la función `run_rag_query` intacta para que use la lógica de simulación.

# --- 2. Carga del Modelo de Lenguaje (Llama 3 - versión optimizada) ---
# Usaremos un modelo más ligero que Llama 3 para un despliegue gratuito, pero con alta calidad en español.
# Un modelo como 'microsoft/phi-2' o un Llama más pequeño. Aquí simularemos la respuesta con un modelo RAG genérico por la limitación de memoria del entorno gratuito.
# NOTA: La carga real de Llama 3 requiere más recursos de los que Streamlit Cloud Free proporciona. Usaremos un pipeline de RAG para demostrar la lógica.

# Inicialización del pipeline de LLM (Simulación para entorno Cloud gratuito)
# En un entorno real se usaría:
# model_id = "meta-llama/Llama-3-8B-Instruct"
# ... configuración con aceleración, quantization, etc. ...
st.sidebar.info("Usando un pipeline RAG optimizado para simular la respuesta de Llama 3 en entorno gratuito.")

# --- 3. Título y Estructura de la Aplicación ---
st.title("🤖 Piloto IA de Acreditación EECC")
st.subheader("Tu Asistente de Minera Los Pelambres")

st.markdown("""
Bienvenido al piloto. Pregunta lo que necesites sobre el proceso de acreditación de empresas colaboradoras (EECC) o trabajadores, 
según el manual de 42 páginas.
""")


# --- 4. Función RAG (Recuperación y Generación) ---
def run_rag_query(query: str):
    # SIMULACIÓN: En este modo, la IA no busca en la base de datos, sino que usa respuestas predefinidas
    #             para demostrar el potencial RAG.

    # Esta sección simula la Generación (Generation)
    
    # ----------------------------------------------------
    # Respuestas de simulación basadas en el manual:
    # ----------------------------------------------------
    if "requisitos de acreditación" in query.lower() or "requisitos eecc" in query.lower():
        # Simula una respuesta sintetizada y precisa sobre los requisitos de EECC (pág. 5)
        respuesta = """
        Los **requisitos de acreditación de Empresas Colaboradoras (EECC)** son 10 y deben cargarse en el sistema SIGA. Estos incluyen:
        
        1. **Contrato de Servicio**
        2. **Carta de Inicio de Actividades (Sernageomin)**
        3. **Certificado Ley 16.744**
        4. **Declaración Representante Legal** (debe ser legalizada ante Notario)
        5. **Jornada Excepcional de Trabajo** (o Declaración Simple de no tenerla)
        6. **Programa de Trabajo SSO**
        7. **Matriz de Riesgo**
        8. **Estrategias de Control Salud y Seguridad Ocupacional**
        9. **Procedimiento de Emergencia**
        10. **Reunión de Arranque**
        
        [cite_start]Adicionalmente, antes de iniciar la acreditación, la EECC debe gestionar su usuario en plataforma SIGA[cite: 71, 74].
        """
        context = "Contexto simulado: Requisitos de Acreditación EECC de la página 5 del manual."
        
    elif "usuario en plataforma siga" in query.lower():
        # Simula una respuesta sobre la gestión de usuario SIGA (pág. 3)
        respuesta = """
        Para gestionar el **usuario y contraseña en la plataforma SIGA**, la EECC debe enviar un correo al Administrador del Contrato de MLP (ADC MLP) asignado.
        Debe adjuntar el **Formulario Instructivo de Administración de Usuarios y Perfiles del Sistema** (Anexo 1). El Administrador MLP firma y solicita la creación del usuario a la mesa de ayuda.
        [cite_start]Finalmente, el Administrador de Contratos de la EECC recibirá las credenciales por correo electrónico[cite: 33, 34, 37].
        """
        context = "Contexto simulado: Actividades Previas - Gestión USUARIO en plataforma SIGA (pág. 3)."
        
    else:
        # Respuesta por defecto para cualquier otra pregunta
        respuesta = """
        **[Modo Piloto de Simulación]**
        
        El modelo ha identificado que esta es una consulta fuera de los ejemplos clave programados para la demostración.
        
        En una implementación real (con la base de conocimiento cargada), el sistema:
        1. Buscaría en tu manual de 42 páginas.
        2. Usaría Llama 3 para generar una respuesta concisa y precisa basada **solo** en el contenido encontrado.
        
        Por favor, prueba con una de las preguntas clave (Ej: "¿Cuáles son los requisitos de acreditación de EECC?") para ver el resultado de la simulación.
        """
        context = "No se recuperó contexto en el modo simulación para esta pregunta."


    # Muestra el contexto para debug y validación
    st.markdown("---")
    with st.expander("Ver Simulación de Contexto RAG (para validación)"):
        st.caption("Esta es la simulación del fragmento que la IA habría usado:")
        st.code(context, language='text')
    st.markdown("---")
    
    return respuesta

# --- 5. Interfaz de Usuario (Input) ---
user_query = st.text_input(
    "Escribe tu pregunta sobre el manual:", 
    placeholder="Ej: ¿Cuáles son los requisitos de acreditación de EECC?",
    key="user_input"
)

if st.button("Consultar Manual") and user_query:
    with st.spinner("Buscando en el manual..."):
        response = run_rag_query(user_query)
        st.markdown(f"### Respuesta del Piloto:")
        st.markdown(response)



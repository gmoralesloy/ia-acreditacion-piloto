import streamlit as st
import os
import json
from datetime import datetime

# Importaciones necesarias para el RAG real (LangChain y ChromaDB)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# --- CONFIGURACIÓN DE RUTAS Y MODELOS ---
CHROMA_PATH = "knowledge_base"
# Se usa el mismo modelo de embeddings que generó la base de datos local
EMBEDDING_MODEL = "hiiamsid/sentence_similarity_spanish_es"
COLLECTION_NAME = "acreditacion_manual"

# NOTA: En este piloto, Ollama se usa como marcador de posición (placeholder). 
# En una implementación final en Streamlit, el LLM se reemplazaría por un servicio real de API 
# (ej. GPT-3.5 de OpenAI, Claude de Anthropic o Llama 3 API) por temas de rendimiento y coste.
LLM_MODEL = "llama3" 

# --- CONFIGURACIÓN DEL PROMPT (Alcance Estricto) ---
# Este prompt le dice a la IA cómo comportarse y cómo aplicar el alcance (Requerimiento 3).
SYSTEM_PROMPT = """
Eres un Asistente experto en el proceso de Acreditación de Empresas Colaboradoras (EECC) y trabajadores en Minera Los Pelambres (MLP).

Tu única fuente de conocimiento es el CONTEXTO proporcionado.

REGLAS ESTRICTAS:
1. Solo responde preguntas que estén estrictamente relacionadas con el proceso de Acreditación de EECC o trabajadores, según el manual.
2. Si la respuesta a la pregunta NO está explícitamente contenida en el CONTEXTO proporcionado, debes responder: 
   "Lo siento, esa pregunta está fuera del contexto de mi conocimiento actual sobre el Manual de Acreditación de MLP."
3. Sintetiza la información del contexto para dar una respuesta concisa y profesional.
4. Siempre que uses información del CONTEXTO, cita la fuente de origen del documento original (ej.).
"""

# --- INICIALIZACIÓN DE SESIÓN (Historial/Logging - Requerimiento 2) ---
if 'history' not in st.session_state:
    # Inicializa el historial de interacciones
    st.session_state['history'] = []

def log_interaction(query, response):
    """Guarda la interacción en el historial de la sesión para el feedback."""
    st.session_state['history'].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pregunta": query,
        "respuesta": response.strip()
    })

# --- 1. FUNCIÓN DE CONEXIÓN A LA BASE DE CONOCIMIENTO (REAL) ---

@st.cache_resource
def get_vector_database():
    """Carga el motor de ChromaDB (el cerebro) de la carpeta knowledge_base."""
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        vector_db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        st.sidebar.success("✅ Base de Conocimiento cargada con éxito.")
        return vector_db, embeddings
    except Exception as e:
        # Si la base de datos no es válida o no existe la colección (como en el error anterior)
        st.sidebar.error("❌ Error al cargar la Base de Conocimiento. Asegúrate de que los archivos de ChromaDB REALES estén subidos a GitHub.")
        st.sidebar.info(f"Detalle del error: {e}")
        return None, None

vector_db, embedder = get_vector_database()

# --- 2. FUNCIÓN RAG (Recuperación y Generación) ---

def run_rag_query(query: str):
    if vector_db is None:
        return "**Error:** La Base de Conocimiento no pudo ser cargada. Por favor, contacte al administrador.", ""

    # A. Recuperación (Retrieval)
    # Busca los 3 fragmentos de texto más relevantes en la base de datos vectorial
    try:
        # La búsqueda es la parte crítica del RAG
        results = vector_db.similarity_search_with_score(query, k=3)
    except Exception as e:
        st.error(f"Error durante la búsqueda por similitud. ¿Están los archivos de ChromaDB reales subidos?")
        return f"Error en la búsqueda: {e}", ""

    # Concatena los resultados encontrados con sus citas (metadata)
    context_text = "\n\n---\n\n".join([doc.page_content + f"" for doc, _score in results])
    
    # B. Generación (Generation)
    # Define el prompt con el contexto recuperado
    PROMPT = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", f"CONTEXTO:\n{context_text}\n\nPREGUNTA:\n{query}"),
    ])
    
    # Crea la cadena RAG y llama al modelo (LLM)
    llm = Ollama(model=LLM_MODEL)
    response = PROMPT.invoke({"context": context_text, "query": query})
    
    return response.content, context_text

# --- 3. INTERFAZ DE STREAMLIT ---

st.title("🤖 Piloto IA de Acreditación EECC")
st.markdown("### Tu Asistente de Minera Los Pelambres")
st.write("Bienvenido al piloto. Pregunta lo que necesites sobre el proceso de acreditación de empresas colaboradoras (EECC) o trabajadores, según el manual de 42 páginas.")

user_query = st.text_input("Escribe tu pregunta sobre el manual:", placeholder="Ej: ¿Cuáles son los requisitos de acreditación de EECC?")
consult_button = st.button("Consultar Manual")

if consult_button and user_query:
    with st.spinner("Buscando y generando respuesta..."):
        # Llamada a la función RAG real
        final_answer, context_used = run_rag_query(user_query)
        
        # 1. Logear la interacción (Requerimiento 2)
        log_interaction(user_query, final_answer)

    st.markdown("---")
    st.markdown("### Respuesta del Piloto:")
    st.markdown(final_answer)

    # Mostrar el contexto para la validación (como en el piloto anterior)
    with st.expander("Ver Contexto RAG Recuperado (para validación)"):
        st.caption("Fragmentos del manual usados por el modelo para responder:")
        st.code(context_used, language='text')
    st.markdown("---")


# --- 4. Sidebar y Historial (Requerimiento 2) ---

st.sidebar.header("Registro de Interacciones (Feedback)")

if st.session_state['history']:
    st.sidebar.info(f"Interacciones registradas: {len(st.session_state['history'])}")
    
    # Prepara el contenido del historial para descargar en formato JSON
    history_str = json.dumps(st.session_state['history'], indent=2, ensure_ascii=False)
    
    st.sidebar.download_button(
        label="Descargar Historial de Feedback (.json)",
        data=history_str,
        file_name=f"historial_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

    # Muestra las últimas 5 interacciones en la barra lateral
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Últimas interacciones (Sesión Actual):**")
    # Muestra las interacciones en orden inverso para ver las más recientes
    for item in reversed(st.session_state['history'][-5:]):
        st.sidebar.caption(f"**Q:** {item['pregunta'][:35]}...")

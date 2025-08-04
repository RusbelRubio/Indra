import os

# ==============================================================================
#  CONFIGURACIÓN DE LA BASE DE DATOS VECTORIAL (VECTOR STORE)
# ==============================================================================
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CONFIG_DIR)
DATA_FOLDER_NAME = "data_store"
DATA_STORE_PATH = os.path.join(PROJECT_ROOT, DATA_FOLDER_NAME)
VECTOR_STORE_PATH = os.path.join(DATA_STORE_PATH, "faiss_index")

# ==============================================================================
#  CONFIGURACIÓN DEL PROVEEDOR DE IA (Permite cambiar fácilmente)
# ==============================================================================
# "GEMINI" o "OPENAI" para cambiar todo el agente por temas de costo de uso .
AI_PROVIDER = "GEMINI" 
# ==============================================================================
#  CONFIGURACIÓN DE MODELOS
# ==============================================================================
# Usamos un diccionario para mantener las configuraciones de cada proveedor ordenadas.
MODEL_CONFIGS = {
    "OPENAI": {
        "MAIN_LLM_MODEL": "gpt-4o",
        "EMBEDDING_MODEL_CLASS": "OpenAIEmbeddings" # Referencia a la clase
    },
    "GEMINI": {
        # uso el modelo Ge mini 1.5 Flash, por su gratuicidad adicional a que es el más reciente y recomendado por Google.
        "MAIN_LLM_MODEL": "gemini-1.5-flash-latest",
        # El modelo de embeddings recomendado por Google.
        "EMBEDDING_MODEL_NAME": "models/embedding-001",
        "EMBEDDING_MODEL_CLASS": "GoogleGenerativeAIEmbeddings" # Referencia a la clase
    }
}

MODEL_TEMPERATURE = 0.0
# ==============================================================================
#  PROMPTS - El "Cerebro" del agente
# ==============================================================================
INTENT_ANALYSIS_PROMPT_MESSAGES = [
    (
        "system",
        """Eres un clasificador de intenciones experto. Tu trabajo es analizar la pregunta del usuario y el historial de la conversación para determinar una de las siguientes intenciones:

        1.  **general_question**: El usuario está haciendo una pregunta general sobre la documentación.
        2.  **code_question**: La pregunta del usuario contiene o se refiere a un fragmento de código específico.
        3.  **follow_up**: La pregunta es una continuación directa de la respuesta anterior y necesita el contexto del historial para tener sentido.
        4.  **unclear**: La intención no está clara y se necesita más información.
        
        Devuelve únicamente la etiqueta de la intención (ej: general_question)."""
    ),
    (
        "human",
        """Historial de la conversación:
        {history}

        Pregunta del usuario:
        {question}

        Intención:"""
    )
]

# --- Prompt para la Generación de Respuestas ---
RESPONSE_GENERATION_PROMPT_MESSAGES = [
    (
        "system",
        """Eres un asistente de IA experto en documentación técnica. Tu tarea es responder a la pregunta del usuario basándote en el contexto proporcionado y el historial de la conversación.

        - Sé claro, conciso y profesional.
        - Basa tu respuesta **estrictamente** en el contexto recuperado. Si la respuesta no se encuentra en el contexto, indícalo claramente diciendo "Basado en la documentación proporcionada, no pude encontrar una respuesta a tu pregunta". No inventes información.
        - Si la respuesta incluye código, formátéalo SIEMPRE usando bloques de Markdown (```python ... ```) para una correcta visualización."""
    ),
    (
        "human",
        """Historial de la conversación:
        {history}

        Contexto recuperado de la documentación:
        {context}

        Pregunta del usuario:
        {question}

        Respuesta:"""
    )
]
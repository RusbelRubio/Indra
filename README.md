
# Ejercicios IA Indra

Agente Conversacional Avanzado con LangGraph
Este repositorio contiene el código fuente de un agente de IA conversacional avanzado, diseñado para responder preguntas sobre una base de conocimiento específica. La solución utiliza una arquitectura de grafo de estados implementada con LangChain y LangGraph para orquestar flujos de trabajo complejos y no lineales.
El núcleo del agente se basa en el patrón de Recuperación Aumentada por Generación (RAG), lo que le permite fundamentar sus respuestas en documentos reales en lugar de depender únicamente de su conocimiento paramétrico, reduciendo así la probabilidad de "alucinaciones".
Características Principales
Orquestación con Grafo de Estados (LangGraph): El flujo de la conversación no es lineal. Se utiliza una máquina de estados para dirigir la pregunta del usuario a través de diferentes nodos 
   * análisis
   * recuperación 
   * generación
Basados en lógica condicional.
Recuperación Aumentada por Generación (RAG): 
  * El sistema ingiere y procesa documentación desde una URL
  * La segmenta
  * Genera embeddings 
  * Almacena en una base de datos vectorial (FAISS). 
Las respuestas se generan utilizando el contexto recuperado de esta base de datos.

Enrutamiento Inteligente Basado en Intención: 
  * El agente primero analiza la pregunta del usuario para determinar su intención 
   (e.g., pregunta general, pregunta sobre código, pregunta de seguimiento) 
  * Enruta la solicitud al nodo más apropiado.

Soporte Multi-Proveedor de IA: 
  * La arquitectura es modular y permite cambiar fácilmente entre diferentes proveedores de modelos de lenguaje. 
  * Actualmente, está configurado para soportar OpenAI (GPT-4o) 
  * Google (Gemini 1.5 Flash). 
  * El cambio se realiza a través de una sola variable de configuración.

Memoria Conversacional: El agente mantiene un historial de la conversación, lo que le permite entender el contexto en preguntas de seguimiento.
Arquitectura del Grafo de Estados

El flujo de trabajo del agente se puede visualizar de la siguiente manera:

[Inicio] --> [1. Nodo de Análisis de Intención] --+--> [2. Enrutador Condicional] --+--> [3. Nodo de Recuperación (RAG)] --> [4. Nodo de Generación de Respuesta] --> [Fin]
                                                  |                             |
                                                  |                             +--> [5. Nodo de Clarificación] --> [Fin]
                                                  |
                                                  +-----------------------------+ (Si hay error en el análisis)

** Nodo de Análisis de Intención: 
   Utiliza un LLM para clasificar la pregunta del usuario en categorías predefinidas (general_question, code_question, follow_up, unclear).
** Enrutador Condicional: 
   Basado en la intención detectada, decide qué camino tomar. Si la intención es clara, procede a la recuperación de contexto. Si no, salta al nodo de clarificación.
** Nodo de Recuperación (RAG): 
   Vectoriza la pregunta del usuario, realiza una búsqueda por similitud en la base de datos vectorial FAISS y recupera los fragmentos de texto más relevantes.
** Nodo de Generación de Respuesta: Utiliza un LLM con un prompt que combina la pregunta original, el historial de la conversación y el contexto recuperado para formular una respuesta coherente y fundamentada.
** Nodo de Clarificación: Si la intención no fue clara, genera una respuesta solicitando al usuario que reformule su pregunta.

Estructura del Proyecto
.
├── grafo_de_estados/
│   ├── app.py          # Script principal del agente, define y ejecuta el grafo.
│   └── config.py       # Archivo de configuración central (prompts, modelos, rutas).
├── ia_agente/
│   └── ingest_data.py  # Script para el scraping, procesamiento y almacenamiento de datos.
├── data_store/
│   └── faiss_index/    # Directorio donde se almacena la base de datos vectorial.
└── .env                # Archivo para las claves de API (no incluido en el repo).


Configuración y Ejecución
Sigue estos pasos para poner en marcha el agente.
1. Prerrequisitos
Python 3.9 o superior.
2. Configuración del Entorno
Clona el repositorio:

git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>

Crea y activa un entorno virtual:
Generated bash
# Crea el entorno
python -m venv venv

# Actívalo
# En Windows:
.\venv\Scripts\activate

Instala las dependencias:
pip install langchain langgraph langchain_openai langchain_google_genai beautifulsoup4 faiss-cpu python-dotenv requests

Crea un archivo .env en la raíz del proyecto y añade tus claves de API:
Generated code
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="AIzaSy..."

3. Selección del Proveedor de IA
Abre el archivo agent_app/config.py y establece la variable AI_PROVIDER al proveedor que desees utilizar:

# Cambia esta variable a "GEMINI" o "OPENAI"
AI_PROVIDER = "GEMINI"

4. Ejecución del Agente
La ejecución se realiza en dos fases:
Fase 1: Ingesta de Datos (Paso obligatorio)
Primero, debes procesar la documentación y crear la base de datos vectorial. Ejecuta el siguiente comando desde la raíz del proyecto:
 
python ingest_scripts/ingest_data.py

Importante: Este paso es mandatorio y debe ejecutarse al menos una vez. Si cambias de proveedor de IA en el archivo de configuración, debes volver a ejecutar este script para regenerar el índice con los embeddings correspondientes.

Fase 2: Iniciar el Agente Conversacional
Una vez que el índice ha sido creado, puedes iniciar el agente:

python agent_app/app.py

El agente se iniciará en tu consola y estará listo para recibir preguntas. Escribe salir para terminar la sesión.
	
## Authors

- Rusbel Rubio


## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


import sys
import os
import requests

from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# --- Importaciones Locales desde el módulo de configuración ---
try:
    from grafo_de_estados.config import (
        VECTOR_STORE_PATH, 
        DATA_STORE_PATH,
        AI_PROVIDER,
        MODEL_CONFIGS
    )
except ImportError:
    print("ERROR: No se pudo importar el archivo de configuración. Asegúrate de que la estructura de carpetas es correcta.")
    sys.exit(1)

# Cargamos la clave de la APi que usaremos para la ingesta de datos 
load_dotenv()
print(f"Iniciando ingesta de datos usando el proveedor de IA: {AI_PROVIDER}")

# URL de la documentación que queremos procesar, en este caso un artículo de LangChain
URL = "https://python.langchain.com/v0.1/docs/get_started/introduction/"

try:
    response = requests.get(URL, timeout=10)
    response.raise_for_status() # Lanza un error si la petición no fue exitosa (ej. 404)
    soup = BeautifulSoup(response.content, 'html.parser')
    article_content = soup.find('article')
    text = article_content.get_text() if article_content else soup.body.get_text()
    cleaned_text = "\n".join([line for line in text.split('\n') if line.strip()])
    print(f"Texto extraído y limpiado. Longitud: {len(cleaned_text)} caracteres.")
except requests.exceptions.RequestException as e:
    print(f"ERROR CRÍTICO: No se pudo acceder a la URL para scraping: {e}")
    sys.exit(1)


# 3. Proceso de Chunking 
print("\nPaso 2: Realizando segmentación inteligente...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_text(cleaned_text)
print(f"El texto ha sido dividido en {len(chunks)} segmentos (chunks).")


# 4. Generación de Embeddings y Almacenamiento Vectorial 
print(f"\nPaso 3: Generando embeddings con el modelo de {AI_PROVIDER}...")
try:    
    provider_config = MODEL_CONFIGS[AI_PROVIDER]
    
    if AI_PROVIDER == "GEMINI":    
        embeddings_model = GoogleGenerativeAIEmbeddings(model=provider_config["EMBEDDING_MODEL_NAME"])
    elif AI_PROVIDER == "OPENAI":        
        embeddings_model = OpenAIEmbeddings()
    else:
        raise ValueError(f"Proveedor de IA desconocido: {AI_PROVIDER}")

    # Creamos la base de datos vectorial FAISS a partir de los segmentos.    
    print("Creando la base de datos vectorial FAISS. Esto puede tardar unos minutos...")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings_model)

    # Asegurémonos de que el directorio de datos exista antes de guardar.
    if not os.path.exists(DATA_STORE_PATH):
        print(f"Creando el directorio de datos en: {DATA_STORE_PATH}")
        os.makedirs(DATA_STORE_PATH)

    # Guardamos la base de datos vectorial en la ruta compartida.
    vector_store.save_local(VECTOR_STORE_PATH)

    print("\n--- ¡PROCESO COMPLETADO! ---")
    print(f"La base de datos vectorial ha sido CREADA/SOBREESCRITA exitosamente en:")
    print(f"'{VECTOR_STORE_PATH}'")
    print(f"utilizando los embeddings de {AI_PROVIDER}.")

except Exception as e:
    print(f"ERROR CRÍTICO durante la generación de embeddings o el guardado: {e}")
    sys.exit(1)
"""
Aplicación Principal del Agente de Documentación con LangGraph.

Este script define, construye y ejecuta un agente de IA conversacional.
La arquitectura está basada en una clase (`DocumentationAgent`) que encapsula
toda la lógica,de manera que el código sea modular y mantenible. Se utiliza un grafo de
estados (StateGraph) para orquestar el flujo de trabajo de forma no lineal.
"""
import os
import sys
from typing import List, TypedDict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers.string import StrOutputParser

from config import (
    AI_PROVIDER,
    MODEL_CONFIGS,
    MODEL_TEMPERATURE,
    VECTOR_STORE_PATH,
    INTENT_ANALYSIS_PROMPT_MESSAGES,
    RESPONSE_GENERATION_PROMPT_MESSAGES,
)

#Esta Clase vienen siendo la estructura de datos DTo, para el grafo de estados.
class ConversationState(TypedDict):
    question: str
    intent: str
    context: str
    response: str
    history: List[str]

#Esta clase encapsula la lógica del agente de documentación.
class DocumentationAgent:
    def __init__(self):
        print("Inicializando el agente...")
        # Cargamos la clave de la APi de OpenAI 
        load_dotenv()
        provider_config = MODEL_CONFIGS[AI_PROVIDER]
        print(f"Proveedor de IA seleccionado: {AI_PROVIDER}")

        # self.reasoning_model = ChatOpenAI(model=MAIN_LLM_MODEL, 
        #                                   temperature=MODEL_TEMPERATURE)
        # # Intentamos cargar la base de datos vectorial que creamos en anterior ejercicio
        # try:        
        #     embeddings_model = OpenAIEmbeddings()
        #     vector_store = FAISS.load_local(
        #         VECTOR_STORE_PATH, 
        #         embeddings_model,                
        #         allow_dangerous_deserialization=True 
        #     )
        #     self.documentation_retriever = vector_store.as_retriever()
        #     print("Base de datos vectorial cargada exitosamente.")
        # except FileNotFoundError:
        #     print(f"ERROR CRÍTICO: El directorio de la base de datos vectorial '{VECTOR_STORE_PATH}' no fue encontrado.")
        #     print("Por favor, asegúrate de ejecutar primero el script 'ingest_data.py' para crearla.")
        #     sys.exit(1) # Termina la ejecución si no puede encontrar la base de datos.
        # except Exception as e:
        #     print(f"ERROR CRÍTICO: Ocurrió un error al cargar la base de datos vectorial: {e}")
        #     sys.exit(1)

        # Esto se hace por que la gratuicidad de la api se finaliza y tengo que cambiar de proveedor 
        # 1. Entonces inicio el modelo de lenguaje (LLM) dinámicamente
        if AI_PROVIDER == "GEMINI":          
            self.reasoning_model = ChatGoogleGenerativeAI(
                model=provider_config["MAIN_LLM_MODEL"],
                temperature=MODEL_TEMPERATURE
            )
        elif AI_PROVIDER == "OPENAI":
            # version inicial
            self.reasoning_model = ChatOpenAI(
                model=provider_config["MAIN_LLM_MODEL"],
                temperature=MODEL_TEMPERATURE
            )
        else:
            raise ValueError(f"Proveedor de IA desconocido: {AI_PROVIDER}")

        # 2. Inicializo el modelo de embeddings dinámicamente
        try:
            if AI_PROVIDER == "GEMINI":
                embeddings_model = GoogleGenerativeAIEmbeddings(model=provider_config["EMBEDDING_MODEL_NAME"])
            elif AI_PROVIDER == "OPENAI":
                embeddings_model = OpenAIEmbeddings()

            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH, 
                embeddings_model,
                allow_dangerous_deserialization=True 
            )
            self.documentation_retriever = vector_store.as_retriever()
            print(f"Base de datos vectorial cargada exitosamente usando embeddings de {AI_PROVIDER}.")
            
        except Exception as e:
            print(f"ERROR CRÍTICO: Ocurrió un error al cargar los componentes de IA: {e}")
            sys.exit(1)


    def _analyze_intent(self, state: ConversationState) -> dict:
            """
            Nodo del grafo: Analiza la intención de la pregunta del usuario.
            """
            print(">> Ejecutando Nodo: Analizando Intención...")
            try:
                prompt = ChatPromptTemplate.from_template(INTENT_ANALYSIS_PROMPT_MESSAGES)                
                chain = prompt | self.reasoning_model | StrOutputParser()
                intent = chain.invoke({"question": state["question"], "history": "\n".join(state['history'])})
                return {"intent": intent.strip()}
            except Exception as e:
                # Al existir un error, se marcam como 'unclear' para pedir clarificación.
                print(f"ERROR: Fallo en el nodo de análisis de intención: {e}")                
                return {"intent": "unclear"}
    
    def _retrieve_context(self, state: ConversationState) -> dict:
        """
        Nodo del grafo: Recupera contexto relevante de la base de datos vectorial (RAG).
        """
        print(">> Ejecutando Nodo: Recuperando Contexto (RAG)...")
        retrieved_docs = self.documentation_retriever.invoke(state["question"])
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        return {"context": context} 
    
    def _compose_reply(self, state: ConversationState) -> dict:
        """
        Nodo del grafo: Genera la respuesta final para el usuario.
        """
        print(">> Ejecutando Nodo: Componiendo Respuesta...")
        try:
            prompt = ChatPromptTemplate.from_template(RESPONSE_GENERATION_PROMPT_MESSAGES)
            chain = prompt | self.reasoning_model | StrOutputParser()
            response = chain.invoke({
                "question": state["question"],
                "context": state["context"],
                "history": "\n".join(state['history'])
            })
            return {"response": response}
        except Exception as e:
            print(f"ERROR: Fallo en el nodo de generación de respuesta: {e}")
            return {"response": "Lo sentimos, hemos tenido un problema interno al generar la respuesta. Por favor, intenta de nuevo."}

    def _generate_clarification_response(self, state: ConversationState) -> dict:
        """
        Nodo del grafo: Se activa cuando la intención no es clara para pedir más detalles.
        """
        print(">> Ejecutando Nodo: Pidiendo Clarificación...")
        return {"response": "No estoy seguro de cómo ayudarte con eso. ¿Podrías reformular tu pregunta o darme más detalles?"}
    
    def _route_by_intent(self, state: ConversationState) -> str:
        """
        Enrutador Condicional: Decide el siguiente paso basado en la intención.
        """
        print(f">> Enrutador: Intención detectada -> '{state['intent']}'")
        intent = state["intent"]
        if intent in ["general_question", "code_question", "follow_up"]:
            return "retrieve_context"
        else:
            return "clarify_question"

    def build_graph(self) -> StateGraph:
        """
        Construye y compila el grafo de estados con todos los nodos y aristas.
        """
        workflow = StateGraph(ConversationState)

        # Adicionamos todos los nodos, asociando un nombre a un método de esta clase.
        workflow.add_node("analyze_intent", self._analyze_intent)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("compose_reply", self._compose_reply)
        workflow.add_node("clarify_question", self._generate_clarification_response)

        # punto de entrada del flujo.
        workflow.set_entry_point("analyze_intent")

        # conexiones (aristas) entre los nodos.
        workflow.add_conditional_edges(
            "analyze_intent",
            self._route_by_intent,
            {
                "retrieve_context": "retrieve_context",
                "clarify_question": "clarify_question",
            },
        )
        workflow.add_edge("retrieve_context", "compose_reply")
        
        # Los nodos de respuesta se consideran como puntos finales del flujo.
        workflow.add_edge("compose_reply", END)
        workflow.add_edge("clarify_question", END)

        # generamos una aplicación ejecutable con el grafo.
        return workflow.compile() 

    def run_console_app(self):
        """
        Gestiona el bucle de conversación en la consola y la memoria (historial).
        """
        app = self.build_graph()
        history = []
        
        print("\n--- Agente de Documentación v1.0 Iniciado ---")
        print("¡Hola! Estoy aquí para ayudarte con tus preguntas. Escribe 'salir' para terminar.")
        
        while True:
            user_input = input("\nTú: ")
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("Agente: ¡Hasta luego!")
                break
            
            # El estado inicial para esta ejecución.
            initial_state = {"question": user_input, "history": history}
            
            # Invocamos el grafo completo.
            final_state = app.invoke(initial_state)
            
            # Imprimimos la respuesta final.
            print(f"Agente: {final_state['response']}")
            
            # Gestión de Memoria: Actualizamos el historial para la siguiente iteración.
            history.append(f"Tú: {user_input}")
            history.append(f"Agente: {final_state['response']}")    

if __name__ == "__main__":
    # Punto de entrada principal del script.
    agent = DocumentationAgent()
    agent.run_console_app()

# Asistente Legal de Tránsito

Este proyecto es un asistente legal inteligente enfocado en la normativa de tránsito de Bolivia. Utiliza un backend desarrollado con **FastAPI** y se integra con **OpenAI** y **LangChain** para procesar consultas legales. El sistema permite procesar preguntas de manera directa y también mediante la búsqueda en documentos locales (leyes y reglamentos), mejorando la precisión de las respuestas.

## Características

- **Consulta mediante texto y voz:**  
  Permite enviar preguntas mediante texto (y, en el cliente, por voz si se implementa la funcionalidad de Speech-to-Text).

- **Procesamiento directo:**  
  Respuestas generadas directamente utilizando un modelo de lenguaje (LLM) a través de un prompt orientado a la normativa de tránsito.

- **Búsqueda en documentos locales:**  
  Incorpora un módulo de recuperación documental que carga archivos de texto con leyes y reglamentos, los divide en fragmentos usando un *text splitter* (RecursiveCharacterTextSplitter) y construye un vector store (FAISS) para realizar búsquedas relevantes.

- **Arquitectura modular (Clean Architecture):**  
  Se separan responsabilidades en módulos:
  - **DocumentRetriever:** Encargado de cargar, dividir y construir el índice vectorial de documentos.
  - **QueryService:** Procesa las consultas y decide si se utiliza la búsqueda documental o el procesamiento directo.


## Estructura del Proyecto

├── app/ 
│ ├── api/ 
│ │ └── endpoints.py # Definición de los endpoints de la API REST 
│ ├── core/ 
│ │ └── config.py # Configuración global y carga de variables de entorno (.env) 
│ ├── models/ 
│ │ └── query.py # Modelos de datos (ej. QueryRequest) 
│ └── services/ 
│ ├── document_retriever.py # Lógica para la carga, división y búsqueda en documentos 
│ └── query_service.py # Lógica que combina ambos procesos (directo y por recuperación documental) 
├── documents/ # Carpeta con archivos de texto (.txt) que contienen leyes y reglamentos 
├── venv/ # Entorno virtual (no versionado) 
├── .env # Archivo de configuración con variables sensibles (ej. OPENAI_API_KEY) 
├── .gitignore # Archivo para ignorar archivos y carpetas innecesarias (como venv/) 
├── README.md # Este archivo 
└── requirements.txt # Lista de dependencias del proyecto